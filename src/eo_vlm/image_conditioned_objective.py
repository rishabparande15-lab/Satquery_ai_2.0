"""Auditable sequence-level image-conditioned objective for Phase 3O.14."""
from __future__ import annotations

import torch
from torch.nn import functional as F


def target_sequence_log_probability(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Sum causal log probabilities of every non-masked authoritative target token.

    `labels` uses the existing CausalLM convention: token *t* is scored by
    logits at *t - 1*.  This works for one-token answers, multi-token answers,
    captions, and an appended EOS without making tokenization assumptions.
    """
    if logits.ndim != 3 or labels.shape != logits.shape[:2]:
        raise ValueError("logits must be [batch, sequence, vocab] and labels [batch, sequence]")
    shifted_logits, shifted_labels = logits[:, :-1].float(), labels[:, 1:]
    active = shifted_labels.ne(-100)
    if not bool(active.any()):
        raise ValueError("at least one target token is required")
    safe_labels = shifted_labels.masked_fill(~active, 0)
    token_log_probs = F.log_softmax(shifted_logits, dim=-1).gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    return (token_log_probs * active).sum(dim=-1)


def image_conditioned_loss(
    correct_logits: torch.Tensor,
    shuffled_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    base_lm_loss: torch.Tensor,
    contrastive_weight: float,
    margin: float,
) -> dict[str, torch.Tensor]:
    """LM loss plus hinge loss enforcing score(correct) >= score(shuffled)+margin."""
    if contrastive_weight < 0 or margin < 0:
        raise ValueError("contrastive_weight and margin must be nonnegative")
    correct_score = target_sequence_log_probability(correct_logits, labels)
    shuffled_score = target_sequence_log_probability(shuffled_logits, labels)
    score_margin = correct_score - shuffled_score
    contrastive = F.relu(float(margin) - score_margin).mean()
    total = base_lm_loss + float(contrastive_weight) * contrastive
    return {"total_loss": total, "base_lm_loss": base_lm_loss, "contrastive_loss": contrastive,
            "correct_score": correct_score.mean(), "shuffled_score": shuffled_score.mean(), "score_margin": score_margin.mean()}


def image_conditioned_penalty(score_margin: torch.Tensor, *, kind: str, weight: float, margin: float = 0.0) -> torch.Tensor:
    """Return a nonnegative contrastive penalty with a fixed favorable direction.

    Both forms decrease as `score_correct - score_shuffled` increases.  The
    smooth form is `weight * softplus(-score_margin)`, so it has no hinge
    threshold while retaining the same sign convention.
    """
    if weight < 0 or margin < 0:
        raise ValueError("weight and margin must be nonnegative")
    if kind == "none":
        return score_margin.sum() * 0
    if kind == "hinge":
        return float(weight) * F.relu(float(margin) - score_margin).mean()
    if kind == "softplus":
        return float(weight) * F.softplus(-score_margin).mean()
    raise ValueError(f"unknown image-conditioned penalty: {kind}")
