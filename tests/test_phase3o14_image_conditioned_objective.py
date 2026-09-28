import torch

from src.eo_vlm.image_conditioned_objective import image_conditioned_loss, image_conditioned_penalty, target_sequence_log_probability


def test_multitoken_target_sequence_score_uses_all_unmasked_causal_targets():
    logits = torch.zeros(1, 5, 4)
    labels = torch.tensor([[-100, -100, 1, 2, 3]])
    score = target_sequence_log_probability(logits, labels)
    assert torch.allclose(score, torch.tensor([-3 * torch.log(torch.tensor(4.0))]))


def test_margin_loss_prefers_correct_image_and_is_finite_with_projector_gradients():
    projector = torch.nn.Parameter(torch.tensor(0.0))
    labels = torch.tensor([[-100, 1, 2]])
    correct = torch.zeros(1, 3, 3); shuffled = torch.zeros(1, 3, 3)
    correct[:, 0, 1] = projector + 2; correct[:, 1, 2] = projector + 2
    shuffled[:, 0, 1] = -projector; shuffled[:, 1, 2] = -projector
    result = image_conditioned_loss(correct, shuffled, labels, base_lm_loss=correct.sum() * 0, contrastive_weight=1.0, margin=1.0)
    assert torch.isfinite(result["total_loss"]) and result["score_margin"] > 0
    result["total_loss"].backward()
    assert projector.grad is not None and torch.isfinite(projector.grad)


def test_swapping_conditions_reverses_score_margin_and_changes_hinge():
    labels = torch.tensor([[-100, 1]])
    good = torch.tensor([[[0.0, 4.0], [0.0, 0.0]]], requires_grad=True)
    bad = torch.tensor([[[0.0, -4.0], [0.0, 0.0]]], requires_grad=True)
    forward = image_conditioned_loss(good, bad, labels, base_lm_loss=good.sum() * 0, contrastive_weight=1.0, margin=1.0)
    swapped = image_conditioned_loss(bad, good, labels, base_lm_loss=bad.sum() * 0, contrastive_weight=1.0, margin=1.0)
    assert forward["score_margin"] > 0 > swapped["score_margin"]
    assert forward["contrastive_loss"] < swapped["contrastive_loss"]


def test_smooth_penalty_has_the_same_favorable_score_direction_as_hinge():
    weak = image_conditioned_penalty(torch.tensor([-1.0]), kind="softplus", weight=0.25)
    strong = image_conditioned_penalty(torch.tensor([1.0]), kind="softplus", weight=0.25)
    assert strong < weak and torch.isfinite(strong)
