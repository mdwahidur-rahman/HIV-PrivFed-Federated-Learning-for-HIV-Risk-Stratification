"""
Algorithm 2 (ClientUpdate): differentially private local training.

Per-example gradients are clipped to C=1.0 (Eq. clip) and averaged with
injected Gaussian noise (Eq. dpsgd) via Opacus's PrivacyEngine, which performs
true per-sample gradient clipping/noising rather than approximating it at the
batch level -- this is the mechanism the manuscript's patient-level privacy
claim rests on (and the mechanism review issue #5 says is NOT, by itself,
sufficient for an end-to-end guarantee once data-dependent preprocessing,
data-dependent class weights, and pooled threshold selection are also in the
pipeline -- see that module's docstrings for where those additional privacy
costs would need to be accounted for separately).
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Optional

import torch
from torch.utils.data import DataLoader, TensorDataset
from opacus import PrivacyEngine

from .resmlp_dp import ResMLP_DP
from .losses import focal_reweight, fedprox_penalty, class_weight_from_counts
from .config import DP, LOSS


@dataclass
class ClientUpdateResult:
    state_dict: dict
    n_examples: int
    final_train_loss: float


def client_update(
    global_state_dict: dict,
    X: torch.Tensor,
    y: torch.Tensor,
    noise_multiplier: float,
    clip_norm: float = DP.clip_norm,
    learning_rate: float = DP.learning_rate,
    local_epochs: int = DP.local_epochs,
    batch_size: int = DP.batch_size,
    fedprox_mu: float = LOSS.fedprox_mu,
    focal_gamma: float = LOSS.focal_gamma,
    device: str = "cpu",
) -> ClientUpdateResult:
    """
    One client's local DP-SGD training for `local_epochs` epochs, matching
    Algorithm 2. `X`, `y` must contain ONLY real (is_synthetic == 0) records
    for this client, per the manuscript's stated FedProx objective (Eq. fedprox).
    """
    input_dim = X.shape[-1]  # inferred from the actual data, not assumed to be the
                              # paper's d=51 -- keeps this correct for any feature set
    model = ResMLP_DP(input_dim=input_dim).to(device)
    model.load_state_dict(global_state_dict)
    global_params = [p.clone().detach() for p in model.parameters()]

    n_pos = int((y == 1).sum().item())
    n_neg = int((y == 0).sum().item())
    w_pos = class_weight_from_counts(n_pos, n_neg)

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    privacy_engine = PrivacyEngine()
    model, optimizer, loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=loader,
        noise_multiplier=noise_multiplier,
        max_grad_norm=clip_norm,
        poisson_sampling=True,
    )

    model.train()
    last_loss = float("nan")
    for _epoch in range(local_epochs):
        for xb, yb in loader:
            if xb.shape[0] == 0:
                continue  # Poisson sampling can draw an empty lot
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()

            logits = model(xb)
            per_example_loss = focal_reweight(logits, yb, w_pos, gamma=focal_gamma)
            # FedProx compares against the ORIGINAL global weights, not this
            # client's evolving local copy, and is added as a shared scalar
            # onto every per-example loss so Opacus still sees one scalar
            # per example for its per-sample-gradient hook.
            current_params = [p for p in model.parameters()]
            prox = fedprox_penalty(current_params, global_params, mu=fedprox_mu)
            loss = (per_example_loss + prox).mean()

            loss.backward()
            optimizer.step()
            last_loss = float(loss.detach().cpu())

    # Opacus wraps the module (GradSampleModule); unwrap before returning the
    # plain state_dict so aggregation code doesn't need to know about Opacus.
    clean_state = {k.replace("_module.", ""): v for k, v in model.state_dict().items()}
    return ClientUpdateResult(state_dict=clean_state, n_examples=len(dataset), final_train_loss=last_loss)


def non_private_client_update(
    global_state_dict: dict,
    X: torch.Tensor,
    y: torch.Tensor,
    learning_rate: float = DP.learning_rate,
    local_epochs: int = DP.local_epochs,
    batch_size: int = DP.batch_size,
    fedprox_mu: Optional[float] = None,   # None => plain FedAvg (no proximal term); a value => FedProx
    focal_gamma: float = LOSS.focal_gamma,
    device: str = "cpu",
) -> ClientUpdateResult:
    """
    Non-DP counterpart used for the FedAvg / FedProx / local-only baselines
    (Table 5), sharing the same loss machinery as the private path so the only
    difference between "FedProx" and "HIV-PrivFed" in this codebase is the
    presence of the Opacus privacy engine.
    """
    input_dim = X.shape[-1]
    model = ResMLP_DP(input_dim=input_dim).to(device)
    model.load_state_dict(global_state_dict)
    global_params = [p.clone().detach() for p in model.parameters()]

    n_pos = int((y == 1).sum().item())
    n_neg = int((y == 0).sum().item())
    w_pos = class_weight_from_counts(n_pos, n_neg)

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    model.train()
    last_loss = float("nan")
    for _epoch in range(local_epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            per_example_loss = focal_reweight(logits, yb, w_pos, gamma=focal_gamma)
            if fedprox_mu is not None:
                current_params = [p for p in model.parameters()]
                prox = fedprox_penalty(current_params, global_params, mu=fedprox_mu)
                loss = (per_example_loss + prox).mean()
            else:
                loss = per_example_loss.mean()
            loss.backward()
            optimizer.step()
            last_loss = float(loss.detach().cpu())

    return ClientUpdateResult(state_dict=copy.deepcopy(model.state_dict()), n_examples=len(dataset), final_train_loss=last_loss)
