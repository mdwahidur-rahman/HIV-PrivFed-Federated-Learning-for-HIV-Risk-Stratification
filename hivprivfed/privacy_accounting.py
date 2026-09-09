"""
Renyi differential privacy accounting, via Opacus's own accountant rather than
a hand-rolled RDP-to-DP conversion.

Review issue #6 specifically flags that a generic textbook RDP->DP formula
(Eq. rdp2dp in the manuscript) is a valid explanatory BOUND but is looser than
the tighter conversion Opacus's accountant actually implements internally --
and that the manuscript's reported epsilon values must come from whichever
formula actually generated them. This module therefore calls Opacus's
accountant directly (get_epsilon) as the single source of truth, rather than
reimplementing Eq. rdp2dp by hand, so there is no risk of the two silently
diverging.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from opacus.accountants import RDPAccountant


@dataclass
class PrivacyState:
    achieved_epsilon: float
    delta: float
    noise_multiplier: float
    sample_rate: float
    steps: int


def make_accountant() -> RDPAccountant:
    return RDPAccountant()


def step_accountant(accountant: RDPAccountant, noise_multiplier: float, sample_rate: float) -> None:
    """Record one optimizer step (one Poisson-sampled lot) against the accountant."""
    accountant.step(noise_multiplier=noise_multiplier, sample_rate=sample_rate)


def get_epsilon(accountant: RDPAccountant, delta: float) -> float:
    """
    Direct call into Opacus's own (tight) RDP->DP conversion -- this is what
    should be reported as the "achieved epsilon" in any results table, not a
    hand-computed value from the generic bound in Eq. rdp2dp.
    """
    return accountant.get_epsilon(delta=delta)


def compose_across_rounds(
    per_round_noise_multiplier: float,
    sample_rate: float,
    steps_per_round: int,
    n_rounds: int,
    delta: float,
) -> PrivacyState:
    """
    Compose privacy loss across every local optimizer step in every
    communication round for a single client -- matching the manuscript's
    statement that "privacy loss composes across every local step, not merely
    across communication rounds."
    """
    accountant = make_accountant()
    total_steps = steps_per_round * n_rounds
    for _ in range(total_steps):
        step_accountant(accountant, per_round_noise_multiplier, sample_rate)
    eps = get_epsilon(accountant, delta)
    return PrivacyState(
        achieved_epsilon=eps, delta=delta, noise_multiplier=per_round_noise_multiplier,
        sample_rate=sample_rate, steps=total_steps,
    )


def per_client_accounting_table(
    client_dataset_sizes: List[int],
    expected_batch_size: int,
    noise_multiplier: float,
    local_epochs: int,
    n_rounds: int,
    delta: float,
) -> List[dict]:
    """
    Builds the client-level accounting table requested in review issue #6:
    cohort size, expected batch size, sampling probability, steps, noise
    multiplier, delta, and achieved epsilon -- per client, since client sizes
    differ and a single unlabeled epsilon is not interpretable on its own.

    NOTE: sample_rate here is `expected_batch_size / n_k` (Opacus's expected-
    batch-size convention), not `realized_batch_size / n_k`. Review issue #6
    flags that these two denominators are not the same thing and that using
    the wrong one changes the accounted epsilon -- this function uses the
    expected-size convention to match Opacus's own internal assumption.
    """
    rows = []
    for n_k in client_dataset_sizes:
        sample_rate = min(1.0, expected_batch_size / max(n_k, 1))
        steps_per_epoch = max(1, round(n_k / expected_batch_size))
        steps_per_round = steps_per_epoch * local_epochs
        state = compose_across_rounds(
            per_round_noise_multiplier=noise_multiplier,
            sample_rate=sample_rate,
            steps_per_round=steps_per_round,
            n_rounds=n_rounds,
            delta=delta,
        )
        rows.append({
            "n_k": n_k,
            "expected_batch_size": expected_batch_size,
            "sample_rate": sample_rate,
            "steps_per_round": steps_per_round,
            "total_steps": state.steps,
            "noise_multiplier": noise_multiplier,
            "delta": delta,
            "achieved_epsilon": state.achieved_epsilon,
        })
    return rows
