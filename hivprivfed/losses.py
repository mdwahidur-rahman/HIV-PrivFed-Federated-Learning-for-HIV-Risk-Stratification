"""
Loss functions:
  - Class-weighted binary cross-entropy (Eq. wbce), weight w_k = n_k^- / n_k^+
  - Focal-loss reweighting of the above with gamma = 2 (Eq. focal)
  - FedProx proximal term, mu = 0.01 (Eq. fedprox)

All operate per-example (not batch-reduced) where needed so that Opacus can
compute correct per-sample gradients for DP-SGD.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import Dict

from .config import LOSS


def class_weight_from_counts(n_pos: int, n_neg: int) -> float:
    """w_k = n_k^- / n_k^+, per Section 'Client-Level Objective'."""
    if n_pos == 0:
        return 1.0
    return n_neg / n_pos


def per_example_weighted_bce(logits: torch.Tensor, targets: torch.Tensor, w_pos: float) -> torch.Tensor:
    """
    Eq. wbce, computed per-example (reduction='none') so callers can apply the
    focal-loss reweighting and FedProx term before any batch reduction.
    logits, targets: shape (N,) or (N, 1).
    """
    logits = logits.view(-1)
    targets = targets.view(-1).float()
    pos_weight = torch.tensor(w_pos, dtype=logits.dtype, device=logits.device)
    return F.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=pos_weight, reduction="none"
    )


def focal_reweight(
    logits: torch.Tensor,
    targets: torch.Tensor,
    w_pos: float,
    gamma: float = LOSS.focal_gamma,
) -> torch.Tensor:
    """
    Per-example focal loss: (1 - p_t)^gamma * weighted_BCE(z, y)  (Eq. pt, Eq. focal).
    Returns a per-example tensor, shape (N,).
    """
    logits_flat = logits.view(-1)
    targets_flat = targets.view(-1).float()

    p = torch.sigmoid(logits_flat)
    p_t = torch.where(targets_flat == 1, p, 1 - p)  # Eq. pt

    bce = per_example_weighted_bce(logits_flat, targets_flat, w_pos)
    focal = (1 - p_t).clamp(min=1e-8).pow(gamma) * bce
    return focal


def fedprox_penalty(local_params, global_params, mu: float = LOSS.fedprox_mu) -> torch.Tensor:
    """
    (mu / 2) * ||theta - theta^(t)||_2^2  (Eq. fedprox), summed over all
    parameter tensors. `local_params`/`global_params` are iterables of
    same-shaped tensors in matching order (e.g. model.parameters() before and
    after taking a copy of the broadcast global weights).
    """
    penalty = 0.0
    for lp, gp in zip(local_params, global_params):
        penalty = penalty + torch.sum((lp - gp.detach()) ** 2)
    return 0.5 * mu * penalty


def per_example_fedprox_objective(
    logits: torch.Tensor,
    targets: torch.Tensor,
    w_pos: float,
    local_params,
    global_params,
    gamma: float = LOSS.focal_gamma,
    mu: float = LOSS.fedprox_mu,
) -> torch.Tensor:
    """
    Full per-example local objective (Eq. fedprox): focal loss per example plus
    the (shared, non-per-example) FedProx proximal term added once per example
    so downstream per-example-gradient machinery (Opacus) sees a scalar loss
    per example, matching Algorithm 2's per-example loss `l_i`.
    """
    focal = focal_reweight(logits, targets, w_pos, gamma=gamma)
    prox = fedprox_penalty(local_params, global_params, mu=mu)
    return focal + prox  # broadcasts the (scalar) prox term onto every per-example focal loss
