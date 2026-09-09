#!/usr/bin/env python3
"""
Reproduces the structure of Table 2: all ten conventional classifiers,
trained under both "Augmented" and "Real-only" conditions, evaluated once on
the same disjoint held-out test set.

Usage:
    python scripts/run_conventional_ml.py                     # runs on synthetic data
    python scripts/run_conventional_ml.py --data path/to.csv  # runs on your real extract

Your real CSV must have at minimum a label column and a city column; pass
their names via --label-col / --city-col if they differ from the synthetic
schema's "hiv_status" / "city".
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hivprivfed.synthetic_data import generate_synthetic_sialon, NUMERIC_COLS, COUNT_COLS, CATEGORICAL_COLS
from hivprivfed.preprocessing import participant_level_split, fit_preprocessing, apply_preprocessing
from hivprivfed.augmentation import build_augmented_training_set
from hivprivfed.conventional_models import build_model_zoo, train_and_evaluate
from hivprivfed.utils import set_all_seeds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=None, help="Path to your real Sialon-II extract (CSV). Omit to use synthetic data.")
    parser.add_argument("--label-col", type=str, default="hiv_status")
    parser.add_argument("--city-col", type=str, default="city")
    parser.add_argument("--positive-label", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="conventional_ml_results.csv")
    args = parser.parse_args()

    set_all_seeds(args.seed)

    if args.data is None:
        print("[info] --data not given: using the synthetic Sialon-II-shaped dataset "
              "(see hivprivfed/synthetic_data.py). Results below are NOT comparable to "
              "the manuscript's numbers -- point --data at your real extract for that.")
        df = generate_synthetic_sialon(random_state=args.seed)
        numeric_cols, count_cols, categorical_cols = NUMERIC_COLS, COUNT_COLS, CATEGORICAL_COLS
    else:
        df = pd.read_csv(args.data)
        # NOTE: you must tell this script which of your columns are numeric,
        # count-valued, and categorical -- there is no way to infer that
        # reliably from a real dataset without domain knowledge. Edit the
        # three lists below (or wire up your own CLI flags) to match your
        # actual feature dictionary before trusting these numbers.
        raise NotImplementedError(
            "Set numeric_cols / count_cols / categorical_cols for your real data schema "
            "at the top of main() before running against --data. This is deliberately "
            "not guessed automatically."
        )

    train_df, test_df = participant_level_split(
        df, label_col=args.label_col, city_col=args.city_col, random_state=args.seed
    )

    artifacts = fit_preprocessing(train_df, numeric_cols, count_cols, categorical_cols)
    train_proc = apply_preprocessing(train_df, artifacts)
    test_proc = apply_preprocessing(test_df, artifacts)

    feature_cols = numeric_cols + categorical_cols
    aug_result = build_augmented_training_set(
        train_df, label_col=args.label_col, city_col=args.city_col,
        categorical_cols=categorical_cols, feature_cols=feature_cols,
        positive_label=args.positive_label, random_state=args.seed,
    )

    # "Real-only": per config.DATA.real_only_uses_full_original_partition, this
    # defaults to the FULL original real partition (not the undersampled
    # subset) -- see that flag's docstring in config.py for why, and confirm
    # against your own scripts before trusting this default for your data.
    from hivprivfed.config import DATA as DATA_CFG
    if DATA_CFG.real_only_uses_full_original_partition:
        real_only_train_df = train_df
    else:
        real_only_train_df = aug_result.undersampled_real

    real_only_train_proc = apply_preprocessing(real_only_train_df, artifacts)
    aug_train_proc = apply_preprocessing(aug_result.augmented, artifacts)

    y_test = test_df[args.label_col].to_numpy()
    X_test = test_proc.to_numpy()

    zoo = build_model_zoo(random_state=args.seed)
    rows = []
    for train_set_name, X_train, y_train in [
        ("Augmented", aug_train_proc.to_numpy(), aug_result.augmented[args.label_col].to_numpy()),
        ("Real-only", real_only_train_proc.to_numpy(), real_only_train_df[args.label_col].to_numpy()),
    ]:
        for model_name in zoo:
            metrics = train_and_evaluate(model_name, X_train, y_train, X_test, y_test, random_state=args.seed)
            row = {"Model": model_name, "TrainSet": train_set_name, **metrics.as_dict()}
            rows.append(row)
            print(f"{model_name:14s} {train_set_name:10s} AUROC={row['AUROC']:.4f} AUPRC={row['AUPRC']:.4f} "
                  f"Sens={row['Sensitivity']:.4f} Spec={row['Specificity']:.4f}")

    results = pd.DataFrame(rows)
    results.to_csv(args.out, index=False)
    print(f"\nSaved {len(results)} rows to {args.out}")


if __name__ == "__main__":
    main()
