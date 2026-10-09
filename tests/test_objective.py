import torch
import torch.nn.functional as F

from orthrus.objective import (
    FusedLinearForwardKL,
    model_states,
    orthrus_loss,
    reference_states,
    sample_anchors,
    supervision_mask,
)


def test_fused_kl_matches_autograd_reference():
    torch.manual_seed(0)
    student = torch.randn(37, 16, requires_grad=True)
    teacher = torch.randn(11, 16)
    index = torch.randint(0, 11, (37,)).sort().values  # rows share teacher positions
    weight = torch.randn(50, 16)
    total, agree = FusedLinearForwardKL.apply(student, teacher, index, weight, 8, False)
    total.backward()

    ref_student = student.detach().clone().requires_grad_(True)
    log_ps = F.log_softmax(ref_student @ weight.t(), dim=-1)
    log_pt = F.log_softmax(teacher[index] @ weight.t(), dim=-1)
    reference = F.kl_div(log_ps, log_pt, log_target=True, reduction="sum")  # KL(teacher||student)
    reference.backward()
    assert torch.allclose(total, reference, rtol=1e-5, atol=1e-5)
    assert torch.allclose(student.grad, ref_student.grad, rtol=1e-4, atol=1e-6)
    assert torch.equal(agree, log_ps.argmax(-1) == log_pt.argmax(-1))


def _batch(model, rows=2, seq_len=48, slots=6, seed=0):
    torch.manual_seed(seed)
    ids = torch.randint(0, 500, (rows, seq_len))
    mask = torch.zeros(rows, seq_len, dtype=torch.bool)
    mask[:, 12:] = True
    mask[1, 30:36] = False  # a second prompt inside the packed row
    gen = torch.Generator().manual_seed(seed)
    anchors, valid = sample_anchors(mask, model.config.block_size, slots, gen)
    return ids, mask, anchors, valid


def test_functional_views_match_module_path(tiny_model):
    ids, _, anchors, _ = _batch(tiny_model)
    student, teacher = model_states(tiny_model, ids, anchors)
    ref_student, ref_teacher = reference_states(tiny_model, ids, anchors)
    assert torch.allclose(teacher, ref_teacher, atol=1e-5)
    assert torch.allclose(student, ref_student, atol=1e-5)


def test_loss_matches_undeduplicated_reference(tiny_model):
    """Shared teacher rows and sorted chunks change nothing: same loss and gradients as the plain
    per-position KL over the module path."""
    ids, mask, anchors, valid = _batch(tiny_model, slots=12)  # many overlapping blocks
    k = tiny_model.config.block_size
    loss, stats = orthrus_loss(tiny_model, ids, mask, anchors, valid, kl_chunk_size=16)
    loss.backward()
    grads = {n: p.grad.clone() for n, p in tiny_model.named_parameters() if p.grad is not None}
    tiny_model.zero_grad(set_to_none=True)

    student, teacher = reference_states(tiny_model, ids, anchors)
    student = student.view(ids.shape[0], -1, k, student.shape[-1])[:, :, :-1]
    targets = anchors[:, :, None] + torch.arange(k - 1)
    teacher = teacher.gather(1, targets.flatten(1)[..., None].expand(-1, -1, teacher.shape[-1]))
    keep = supervision_mask(mask, anchors, valid, k)
    weight = tiny_model.lm_head.weight
    log_ps = F.log_softmax(student[keep] @ weight.t(), dim=-1)
    log_pt = F.log_softmax(teacher.view_as(student)[keep] @ weight.t(), dim=-1)
    reference = F.kl_div(log_ps, log_pt, log_target=True, reduction="sum") / keep.sum()
    reference.backward()
    assert stats["tokens"] == keep.sum()
    assert torch.allclose(loss, reference, rtol=1e-5, atol=1e-6)
    assert stats["agree"] == (log_ps.argmax(-1) == log_pt.argmax(-1)).sum()
    for name, param in tiny_model.named_parameters():
        if param.grad is not None:
            assert torch.allclose(grads[name], param.grad, rtol=1e-4, atol=1e-6), name


def test_activation_checkpointing_keeps_gradients(tiny_model):
    ids, mask, anchors, valid = _batch(tiny_model)
    grads = []
    for checkpointing in (False, True):
        if checkpointing:
            tiny_model.gradient_checkpointing_enable({"use_reentrant": False})
        tiny_model.zero_grad(set_to_none=True)
        loss, _ = orthrus_loss(tiny_model, ids, mask, anchors, valid, kl_chunk_size=64)
        loss.backward()
        grads.append({n: p.grad for n, p in tiny_model.named_parameters() if p.grad is not None})
    assert grads[0].keys() == grads[1].keys()
    assert all(torch.allclose(grads[0][n], grads[1][n], atol=1e-6) for n in grads[0])


def test_anchors_lie_on_assistant_tokens():
    mask = torch.zeros(3, 64, dtype=torch.bool)
    mask[0, 10:50] = True
    mask[1, 5:9] = True  # fewer candidates than slots
    gen = torch.Generator().manual_seed(1)
    anchors, valid = sample_anchors(mask, block_size=8, num_anchors=16, generator=gen)
    assert anchors.shape == (3, 16)
    first = anchors[0]
    assert valid[0].all() and mask[0, first].all() and (first[1:] > first[:-1]).all()
    assert valid[1].sum() == 4 and mask[1, anchors[1, :4]].all()
    assert not valid[2].any()  # no assistant tokens: arbitrary anchors, nothing supervised
    assert (anchors >= 1).all() and (anchors <= 64 - 8).all()


def test_supervision_targets_are_next_tokens():
    mask = torch.zeros(1, 20, dtype=torch.bool)
    mask[0, 6:12] = True
    anchors = torch.tensor([[5, 10]])
    keep = supervision_mask(mask, anchors, torch.tensor([[True, True]]), block_size=4)
    # block at 5 predicts tokens 6,7,8; block at 10 predicts 11,12,13
    assert keep.tolist() == [[[True, True, True], [True, False, False]]]


def test_loss_trains_only_diffusion_parameters(tiny_model):
    torch.manual_seed(0)
    ids = torch.randint(0, 500, (2, 48))
    mask = torch.zeros(2, 48, dtype=torch.bool)
    mask[:, 12:] = True
    gen = torch.Generator().manual_seed(0)
    anchors, valid = sample_anchors(mask, tiny_model.config.block_size, 6, gen)
    tiny_model.train()
    loss, stats = orthrus_loss(tiny_model, ids, mask, anchors, valid, kl_chunk_size=64)
    loss.backward()
    grads = {n: p.grad for n, p in tiny_model.named_parameters() if p.grad is not None}
    assert torch.isfinite(loss) and loss > 0
    assert grads and all("_diff." in n for n in grads)
    assert all(torch.isfinite(g).all() for g in grads.values())
    assert stats["tokens"] == stats["tokens_by_offset"].sum()
    assert 0 <= stats["agree"] <= stats["tokens"]
