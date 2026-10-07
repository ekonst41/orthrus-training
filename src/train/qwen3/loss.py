"""Memory-bounded full-vocabulary forward KL distillation."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def chunked_forward_kl(
    student_hidden, teacher_hidden, lm_head, *, chunk_size=32, temperature=1.0
):
    def project_kl(student, teacher):
        student_logp = F.log_softmax(
            F.linear(student, lm_head.weight).float() / temperature, dim=-1
        )
        with torch.no_grad():
            teacher_logp = F.log_softmax(
                F.linear(teacher, lm_head.weight).float() / temperature, dim=-1
            )
        return (
            F.kl_div(student_logp, teacher_logp, log_target=True, reduction="sum")
            * temperature**2
        )

    total = student_hidden.sum().float() * 0.0
    for begin in range(0, student_hidden.shape[0], chunk_size):
        student = student_hidden[begin : begin + chunk_size]
        teacher = teacher_hidden[begin : begin + chunk_size]
        total = total + checkpoint(project_kl, student, teacher, use_reentrant=False)
    return total
