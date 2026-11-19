"""Single-node data parallelism. All ranks hold a model; only diffusion gradients are reduced.

The loss calls functional layers directly, so wrapping model.forward in DDP would not suffice.
"""

import logging
import os
from dataclasses import dataclass
from datetime import timedelta

import torch
import torch.distributed as dist

log = logging.getLogger(__name__)


@dataclass
class Distributed:
    rank: int
    world_size: int
    device: torch.device

    @property
    def primary(self):
        return self.rank == 0

    def primary_call(self, action):
        """Run a file/remote action once; propagate its result or failure to every rank."""
        status = [None]
        if self.primary:
            try:
                status[0] = (True, action())
            except Exception as error:
                log.exception("rank-zero action failed")
                status[0] = (False, f"{type(error).__name__}: {error}")
        if self.world_size > 1:
            dist.broadcast_object_list(status, src=0)
        ok, result = status[0]
        if not ok:
            raise RuntimeError(f"rank-zero action failed: {result}")
        return result

    def each_call(self, action):
        """Keep model/checkpoint load errors from leaving peers waiting at the next collective."""
        result, error = None, None
        try:
            result = action()
        except Exception as exc:
            error = f"rank {self.rank}: {type(exc).__name__}: {exc}"
        errors = [None] * self.world_size
        if self.world_size > 1:
            dist.all_gather_object(errors, error)
        else:
            errors[0] = error
        failures = [e for e in errors if e]
        if failures:
            raise RuntimeError("; ".join(failures))
        return result

    def maximum(self, value):
        tensor = torch.tensor(float(value), device=self.device, dtype=torch.float64)
        if self.world_size > 1:
            dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
        return tensor.item()

    def stop_reason(self, reason):
        flag = torch.tensor(bool(reason), device=self.device, dtype=torch.int32)
        if self.world_size > 1:
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        if not flag.item():
            return ""
        reasons = [None] * self.world_size
        if self.world_size > 1:
            dist.all_gather_object(reasons, reason)
        else:
            reasons[0] = reason
        return next(r for r in reasons if r)


def initialize():
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if int(os.environ.get("LOCAL_WORLD_SIZE", world)) != world:
        raise ValueError("only a single node with shared data/output directories is supported")
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group(
            backend="nccl" if device.type == "cuda" else "gloo", timeout=timedelta(hours=2)
        )
    return Distributed(rank, world, device)


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def close():
    if dist.is_initialized():
        dist.destroy_process_group()


def average_gradients(parameters, world_size: int, bucket_mb: int = 32):
    """Average accumulated gradients in bounded FP32 buffers, including rank-local None grads.

    All-unused parameters retain grad=None. A rank-local unused parameter contributes zero.
    The presence collective guarantees that all ranks run identical bucket collectives.
    """
    if world_size == 1:
        return
    parameters = list(parameters)
    if not parameters:
        return
    device = parameters[0].device
    presence = torch.tensor(
        [p.grad is not None for p in parameters], device=device, dtype=torch.int32
    )
    dist.all_reduce(presence, op=dist.ReduceOp.SUM)
    active = presence.cpu().tolist()
    limit = bucket_mb * 1024 * 1024 // 4
    pending, size = [], 0

    def flush():
        if not pending:
            return
        buffer = torch.empty(sum(p.numel() for p in pending), device=device, dtype=torch.float32)
        offset = 0
        for param in pending:
            part = buffer[offset : offset + param.numel()]
            if param.grad is None:
                part.zero_()
            else:
                part.copy_(param.grad.detach().reshape(-1))
            offset += param.numel()
        dist.all_reduce(buffer, op=dist.ReduceOp.SUM)
        buffer.div_(world_size)
        offset = 0
        for param in pending:
            grad = buffer[offset : offset + param.numel()].view_as(param)
            if param.grad is None:
                param.grad = grad.to(param.dtype).clone()
            else:
                param.grad.copy_(grad)
            offset += param.numel()

    for param, used in zip(parameters, active, strict=True):
        if not used:
            continue
        if pending and size + param.numel() > limit:
            flush()
            pending, size = [], 0
        pending.append(param)
        size += param.numel()
    flush()
