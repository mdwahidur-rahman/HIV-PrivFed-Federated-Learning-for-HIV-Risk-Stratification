#!/usr/bin/env python3
"""
Reproduces the structure of Table 6/7: sweeps HIV-PrivFed over the four noise
multipliers, reporting BOTH predictive metrics and the accountant-derived
achieved epsilon for each -- using the actual sigma value as the primary label
(per review issue #6), not the nominal "eps~N" naming.

Usage:
    python scripts/run_privacy_sweep.py
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hivprivfed.synthetic_data import generate_synthetic_sialon, NUMERIC_COLS, COUNT_COLS, CATEGORICAL_COLS
from hivprivfed.preprocessing import participant_level_split, fit_preprocessing, apply_preprocessing
from hivprivfed.federated import run_federated
from hivprivfed.resmlp_dp import ResMLP_DP
from hivprivfed.conventional_models import evaluate_predictions
from hivprivfed.privacy_accounting import per_client_accounting_table
from hivprivfed.utils import set_all_seeds
from hivprivfed.config import DP
from run_federated import build_clients, evaluate_state_dict  # reuse helpers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=5, help="Communication rounds (manuscript uses T=30)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected-batch-size", type=int, default=DP.batch_size)
    parser.add_argument("--out", type=str, default="privacy_utility_results.csv")
    args = parser.parse_args()

    set_all_seeds(args.seed)
    print("[info] Using synthetic data. Results are illustrative only -- run against your "
          "real Sialon-II extract for numbers you intend to report.")

    df = generate_synthetic_sialon(random_state=args.seed)
    train_df, test_df = participant_level_split(df, "hiv_status", "city", random_state=args.seed)
    artifacts = fit_preprocessing(train_df, NUMERIC_COLS, COUNT_COLS, CATEGORICAL_COLS)
    train_proc = apply_preprocessing(train_df, artifacts)
    test_proc = apply_preprocessing(test_df, artifacts)

    clients = build_clients(train_proc, train_df, "city", "hiv_status")
    input_dim = train_proc.shape[1]
    X_test = test_proc.to_numpy()
    y_test = test_df["hiv_status"].to_numpy()

    client_sizes = [len(c.X) for c in clients]
    rows = []
    for sigma, nominal_eps in zip(DP.noise_multipliers, DP.nominal_epsilon_targets):
        result = run_federated(
            clients, mode="hivprivfed", n_rounds=args.rounds,
            noise_multiplier=sigma, fedprox_mu=0.01,
        )
        m = evaluate_state_dict(result.final_state_dict, input_dim, X_test, y_test)

        acc_table = per_client_accounting_table(
            client_dataset_sizes=client_sizes,
            expected_batch_size=args.expected_batch_size,
            noise_multiplier=sigma,
            local_epochs=DP.local_epochs,
            n_rounds=args.rounds,
            delta=DP.delta,
        )
        achieved_eps_per_client = [r["achieved_epsilon"] for r in acc_table]
        # The federation-level epsilon is the MAXIMUM across clients (the
        # weakest link) since each client is a distinct data subject pool --
        # reporting a single unlabeled scalar without this per-client context
        # is exactly what review issue #6 flags as uninterpretable.
        federation_epsilon = max(achieved_eps_per_client)

        row = {
            "sigma": sigma, "nominal_epsilon_target": nominal_eps,
            "federation_epsilon_delta_1e-5": federation_epsilon,
            "AUROC": m.auroc, "AUPRC": m.auprc, "MCC": m.mcc,
            "Sensitivity": m.sensitivity, "Specificity": m.specificity,
        }
        rows.append(row)
        print(f"sigma={sigma:.1f}  achieved_eps={federation_epsilon:.4f}  "
              f"AUROC={m.auroc:.4f}  AUPRC={m.auprc:.4f}  Sens={m.sensitivity:.4f}  Spec={m.specificity:.4f}")

    results = pd.DataFrame(rows)
    results.to_csv(args.out, index=False)
    print(f"\nSaved {len(results)} rows to {args.out}")


if __name__ == "__main__":
    main()
