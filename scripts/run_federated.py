#!/usr/bin/env python3
"""
Reproduces the structure of Table 5's non-private rows: local-only, FedAvg,
and FedProx, with each of the 13 cities as a natural federated client.

Usage:
    python scripts/run_federated.py                 # synthetic data, few rounds (fast demo)
    python scripts/run_federated.py --rounds 30      # matches the manuscript's T=30
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
from hivprivfed.federated import ClientData, run_federated, run_local_only
from hivprivfed.resmlp_dp import ResMLP_DP
from hivprivfed.conventional_models import evaluate_predictions
from hivprivfed.utils import set_all_seeds


def build_clients(train_proc: pd.DataFrame, train_df: pd.DataFrame, city_col: str, label_col: str):
    clients = []
    for city, idx in train_df.groupby(city_col).groups.items():
        X = torch.tensor(train_proc.loc[idx].to_numpy(), dtype=torch.float32)
        y = torch.tensor(train_df.loc[idx, label_col].to_numpy(), dtype=torch.float32)
        if len(X) == 0:
            continue
        clients.append(ClientData(client_id=str(city), X=X, y=y))
    return clients


def evaluate_state_dict(state_dict: dict, input_dim: int, X_test: np.ndarray, y_test: np.ndarray):
    model = ResMLP_DP(input_dim=input_dim)
    model.load_state_dict(state_dict)
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(torch.tensor(X_test, dtype=torch.float32))).numpy().ravel()
    return evaluate_predictions(y_test, probs)


def average_local_only_predictions(local_states: dict, input_dim: int, X_test: np.ndarray, y_test: np.ndarray):
    """
    For the local-only baseline there is no single global model -- report the
    mean of each client's own model evaluated on the shared test set, matching
    the manuscript's "Local-only(avg)" row label.
    """
    all_probs = []
    for state in local_states.values():
        model = ResMLP_DP(input_dim=input_dim)
        model.load_state_dict(state)
        model.eval()
        with torch.no_grad():
            probs = torch.sigmoid(model(torch.tensor(X_test, dtype=torch.float32))).numpy().ravel()
        all_probs.append(probs)
    mean_probs = np.mean(all_probs, axis=0)
    return evaluate_predictions(y_test, mean_probs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--label-col", type=str, default="hiv_status")
    parser.add_argument("--city-col", type=str, default="city")
    parser.add_argument("--rounds", type=int, default=5, help="Communication rounds (manuscript uses T=30)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_all_seeds(args.seed)

    if args.data is None:
        print("[info] --data not given: using synthetic data. Results are illustrative only.")
        df = generate_synthetic_sialon(random_state=args.seed)
        numeric_cols, count_cols, categorical_cols = NUMERIC_COLS, COUNT_COLS, CATEGORICAL_COLS
    else:
        raise NotImplementedError("Wire up your real schema's numeric/count/categorical columns first.")

    train_df, test_df = participant_level_split(df, args.label_col, args.city_col, random_state=args.seed)
    artifacts = fit_preprocessing(train_df, numeric_cols, count_cols, categorical_cols)
    train_proc = apply_preprocessing(train_df, artifacts)
    test_proc = apply_preprocessing(test_df, artifacts)

    clients = build_clients(train_proc, train_df, args.city_col, args.label_col)
    input_dim = train_proc.shape[1]
    X_test = test_proc.to_numpy()
    y_test = test_df[args.label_col].to_numpy()

    print(f"Built {len(clients)} clients from column '{args.city_col}'.")

    local_states = run_local_only(clients, n_rounds=args.rounds)
    m_local = average_local_only_predictions(local_states, input_dim, X_test, y_test)
    print(f"Local-only(avg)  AUROC={m_local.auroc:.4f} AUPRC={m_local.auprc:.4f} "
          f"Sens={m_local.sensitivity:.4f} Spec={m_local.specificity:.4f}")

    fedavg_result = run_federated(clients, mode="fedavg", n_rounds=args.rounds)
    m_fedavg = evaluate_state_dict(fedavg_result.final_state_dict, input_dim, X_test, y_test)
    print(f"FedAvg           AUROC={m_fedavg.auroc:.4f} AUPRC={m_fedavg.auprc:.4f} "
          f"Sens={m_fedavg.sensitivity:.4f} Spec={m_fedavg.specificity:.4f}")

    fedprox_result = run_federated(clients, mode="fedprox", n_rounds=args.rounds, fedprox_mu=0.01)
    m_fedprox = evaluate_state_dict(fedprox_result.final_state_dict, input_dim, X_test, y_test)
    print(f"FedProx          AUROC={m_fedprox.auroc:.4f} AUPRC={m_fedprox.auprc:.4f} "
          f"Sens={m_fedprox.sensitivity:.4f} Spec={m_fedprox.specificity:.4f}")


if __name__ == "__main__":
    main()
