"""
Algorithm 1 (server-side training loop): orchestrates K clients across T
communication rounds, aggregating via FedAvg weighted by local real-cohort
size (Eq. fedavg).

Supports three non-private modes (local-only, FedAvg, FedProx) and the
private HIV-PrivFed mode, so all four of Table 5's federated rows come from
the same orchestration code with different flags -- reducing the chance that
"FedAvg" and "FedProx" silently differ in more ways than the proximal term,
which review issue #9 flagged as a real risk when the private/non-private
comparison in Table 8 mixes FedAvg and FedProx.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch

from .resmlp_dp import ResMLP_DP
from .dp_training import client_update, non_private_client_update, ClientUpdateResult
from .config import DP


@dataclass
class ClientData:
    client_id: str
    X: torch.Tensor
    y: torch.Tensor


@dataclass
class RoundLog:
    round_index: int
    avg_train_loss: float
    n_clients_participated: int


@dataclass
class FederatedRunResult:
    final_state_dict: dict
    history: List[RoundLog] = field(default_factory=list)


def fedavg_aggregate(client_results: List[ClientUpdateResult]) -> dict:
    """theta^(t+1) = sum_k (n_k / N) * theta_k^(t+1)   (Eq. fedavg)."""
    total_n = sum(r.n_examples for r in client_results)
    if total_n == 0:
        raise ValueError("All participating clients had zero examples this round.")

    agg_state = None
    for r in client_results:
        weight = r.n_examples / total_n
        if agg_state is None:
            agg_state = {k: v.clone().float() * weight for k, v in r.state_dict.items()}
        else:
            for k, v in r.state_dict.items():
                agg_state[k] += v.float() * weight
    return agg_state


def run_local_only(
    clients: List[ClientData],
    n_rounds: int = DP.communication_rounds,
    device: str = "cpu",
) -> Dict[str, dict]:
    """
    Each client trains independently for the same total number of local
    epochs as the federated conditions, with NO aggregation step -- the
    baseline against which FedAvg/FedProx's improvement is measured.
    """
    results = {}
    input_dim = clients[0].X.shape[-1]
    model = ResMLP_DP(input_dim=input_dim).to(device)
    init_state = model.state_dict()
    for c in clients:
        state = init_state
        for _ in range(n_rounds):
            r = non_private_client_update(state, c.X, c.y, fedprox_mu=None, device=device)
            state = r.state_dict
        results[c.client_id] = state
    return results


def run_federated(
    clients: List[ClientData],
    mode: str,  # "fedavg" | "fedprox" | "hivprivfed"
    n_rounds: int = DP.communication_rounds,
    noise_multiplier: Optional[float] = None,
    fedprox_mu: Optional[float] = None,
    device: str = "cpu",
) -> FederatedRunResult:
    """
    Shared orchestration for FedAvg, FedProx, and HIV-PrivFed (DP-FedProx).
    mode="fedavg"     -> no proximal term, no privacy engine
    mode="fedprox"    -> proximal term, no privacy engine
    mode="hivprivfed" -> proximal term AND per-example DP-SGD via Opacus
                          (noise_multiplier must be provided)
    """
    if mode not in {"fedavg", "fedprox", "hivprivfed"}:
        raise ValueError(f"Unknown mode: {mode}")
    if mode == "hivprivfed" and noise_multiplier is None:
        raise ValueError("noise_multiplier is required for mode='hivprivfed'")

    input_dim = clients[0].X.shape[-1]
    global_model = ResMLP_DP(input_dim=input_dim).to(device)
    global_state = global_model.state_dict()
    history = []

    for t in range(n_rounds):
        client_results: List[ClientUpdateResult] = []
        for c in clients:
            if mode == "hivprivfed":
                r = client_update(
                    global_state, c.X, c.y,
                    noise_multiplier=noise_multiplier,
                    device=device,
                )
            else:
                mu = fedprox_mu if mode == "fedprox" else None
                r = non_private_client_update(global_state, c.X, c.y, fedprox_mu=mu, device=device)
            client_results.append(r)

        global_state = fedavg_aggregate(client_results)
        avg_loss = sum(r.final_train_loss for r in client_results) / len(client_results)
        history.append(RoundLog(round_index=t, avg_train_loss=avg_loss, n_clients_participated=len(client_results)))

    return FederatedRunResult(final_state_dict=global_state, history=history)
