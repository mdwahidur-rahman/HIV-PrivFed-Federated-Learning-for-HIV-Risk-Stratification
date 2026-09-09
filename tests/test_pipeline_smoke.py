"""
End-to-end smoke test. Run with: python -m pytest tests/ -v
(or just `python tests/test_pipeline_smoke.py` to run without pytest).

This exercises every stage of the pipeline on synthetic data: preprocessing,
augmentation, conventional ML, ResMLP-DP forward pass, DP-SGD client update,
FedAvg/FedProx/local-only/HIV-PrivFed orchestration, RDP accounting, secure
aggregation with AES-GCM, and bootstrap evaluation. It does NOT validate that
results match the manuscript -- it validates that the code runs, shapes and
types are consistent, and the privacy/security invariants hold (masks cancel,
achieved epsilon increases as noise decreases, etc.).
"""
import numpy as np
import pandas as pd
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hivprivfed.synthetic_data import generate_synthetic_sialon, NUMERIC_COLS, COUNT_COLS, CATEGORICAL_COLS
from hivprivfed.preprocessing import participant_level_split, fit_preprocessing, apply_preprocessing
from hivprivfed.augmentation import build_augmented_training_set
from hivprivfed.conventional_models import train_and_evaluate, build_model_zoo
from hivprivfed.resmlp_dp import ResMLP_DP, gn_groups_that_divide
from hivprivfed.dp_training import client_update, non_private_client_update
from hivprivfed.federated import ClientData, run_federated, run_local_only, fedavg_aggregate
from hivprivfed.privacy_accounting import compose_across_rounds, per_client_accounting_table
from hivprivfed.secure_aggregation import (
    compute_masked_update, server_side_sum, aggregation_correctness_check,
    generate_client_key, encrypt_update, decrypt_update,
)
from hivprivfed.evaluation import youden_threshold, bootstrap_ci, paired_bootstrap_test, metric_auroc, metric_auprc
from hivprivfed.utils import set_all_seeds


def test_data_and_preprocessing():
    df = generate_synthetic_sialon(random_state=0)
    assert len(df) > 0
    assert df["city"].nunique() == 13

    train_df, test_df = participant_level_split(df, label_col="hiv_status", city_col="city", random_state=0)
    assert set(train_df["participant_id"]).isdisjoint(set(test_df["participant_id"]))

    artifacts = fit_preprocessing(train_df, NUMERIC_COLS, COUNT_COLS, CATEGORICAL_COLS)
    train_proc = apply_preprocessing(train_df, artifacts)
    test_proc = apply_preprocessing(test_df, artifacts)

    assert train_proc.isna().sum().sum() == 0, "no NaNs should remain after KNN imputation"
    assert test_proc.isna().sum().sum() == 0
    assert list(train_proc.columns) == list(test_proc.columns)
    print(f"[OK] preprocessing: train shape {train_proc.shape}, test shape {test_proc.shape}")
    return df, train_df, test_df, artifacts, train_proc, test_proc


def test_augmentation(train_df):
    feature_cols = NUMERIC_COLS + CATEGORICAL_COLS
    result = build_augmented_training_set(
        train_df, label_col="hiv_status", city_col="city",
        categorical_cols=CATEGORICAL_COLS, feature_cols=feature_cols,
        positive_label=1, random_state=0,
    )
    assert len(result.original_real) == len(train_df)
    assert (result.augmented["hiv_status"] == 1).sum() >= (result.undersampled_real["hiv_status"] == 1).sum()
    assert result.n_synthetic >= 0
    print(f"[OK] augmentation: original_real={len(result.original_real)}, "
          f"undersampled_real={len(result.undersampled_real)}, augmented={len(result.augmented)}, "
          f"n_synthetic={result.n_synthetic}")
    return result


def test_conventional_models(train_proc, test_proc, train_df, test_df):
    X_train = train_proc.to_numpy()
    y_train = train_df["hiv_status"].to_numpy()
    X_test = test_proc.to_numpy()
    y_test = test_df["hiv_status"].to_numpy()

    zoo = build_model_zoo()
    assert len(zoo) == 10, f"expected 10 conventional models, got {len(zoo)}"

    # Just test a fast subset for the smoke test (LR + RF); the full zoo is
    # exercised in scripts/run_conventional_ml.py.
    for name in ["LR", "RF"]:
        metrics = train_and_evaluate(name, X_train, y_train, X_test, y_test, threshold=0.5)
        d = metrics.as_dict()
        assert 0.0 <= d["AUROC"] <= 1.0
        assert 0.0 <= d["AUPRC"] <= 1.0
        print(f"[OK] conventional model {name}: AUROC={d['AUROC']:.4f} AUPRC={d['AUPRC']:.4f}")


def test_resmlp_dp_forward():
    model = ResMLP_DP()
    assert model.param_count > 0
    x = torch.randn(16, model.linear1.in_features)
    out = model(x)
    assert out.shape == (16, 1)
    probs = model.predict_proba(x)
    assert torch.all((probs >= 0) & (probs <= 1))
    g = gn_groups_that_divide(64, preferred=8)
    assert 64 % g == 0
    print(f"[OK] ResMLP-DP forward pass: {model.param_count} parameters, output shape {tuple(out.shape)}")


def test_client_updates_and_federated_orchestration(train_proc, train_df):
    set_all_seeds(0)
    X = torch.tensor(train_proc.to_numpy(), dtype=torch.float32)
    y = torch.tensor(train_df["hiv_status"].to_numpy(), dtype=torch.float32)

    # Split into a couple of fake "clients" for the smoke test.
    n = len(X)
    clients = [
        ClientData(client_id="clientA", X=X[: n // 2], y=y[: n // 2]),
        ClientData(client_id="clientB", X=X[n // 2:], y=y[n // 2:]),
    ]

    input_dim = X.shape[-1]
    global_model = ResMLP_DP(input_dim=input_dim)
    init_state = global_model.state_dict()

    # Non-private client update (FedProx-style)
    r_nonpriv = non_private_client_update(init_state, clients[0].X, clients[0].y, fedprox_mu=0.01, local_epochs=1)
    assert r_nonpriv.n_examples == len(clients[0].X)

    # Private (DP-SGD) client update -- this is the part that needs Opacus working correctly.
    r_priv = client_update(init_state, clients[0].X, clients[0].y, noise_multiplier=1.0, local_epochs=1, batch_size=8)
    assert r_priv.n_examples == len(clients[0].X)
    assert set(r_priv.state_dict.keys()) == set(init_state.keys())
    print("[OK] client_update (DP-SGD via Opacus) ran without error")

    # FedAvg aggregation weighting sanity check
    agg = fedavg_aggregate([r_nonpriv, r_priv])
    assert set(agg.keys()) == set(init_state.keys())

    # Full orchestration, 2 rounds, tiny for speed
    fed_result = run_federated(clients, mode="fedprox", n_rounds=2, fedprox_mu=0.01)
    assert len(fed_result.history) == 2
    print(f"[OK] run_federated(fedprox): losses={[round(h.avg_train_loss, 4) for h in fed_result.history]}")

    priv_result = run_federated(clients, mode="hivprivfed", n_rounds=2, noise_multiplier=1.0, fedprox_mu=0.01)
    assert len(priv_result.history) == 2
    print(f"[OK] run_federated(hivprivfed): losses={[round(h.avg_train_loss, 4) for h in priv_result.history]}")

    local_result = run_local_only(clients, n_rounds=2)
    assert set(local_result.keys()) == {"clientA", "clientB"}
    print("[OK] run_local_only completed")


def test_privacy_accounting():
    state = compose_across_rounds(
        per_round_noise_multiplier=1.0, sample_rate=0.1, steps_per_round=5, n_rounds=3, delta=1e-5
    )
    assert state.achieved_epsilon > 0

    # Sanity: LOWER noise multiplier -> LOOSER (larger) achieved epsilon, matching
    # the manuscript's own observed sigma/epsilon direction.
    tight = compose_across_rounds(per_round_noise_multiplier=2.0, sample_rate=0.1, steps_per_round=5, n_rounds=3, delta=1e-5)
    loose = compose_across_rounds(per_round_noise_multiplier=0.5, sample_rate=0.1, steps_per_round=5, n_rounds=3, delta=1e-5)
    assert loose.achieved_epsilon > tight.achieved_epsilon, "less noise should mean a larger (looser) epsilon"
    print(f"[OK] privacy accounting: sigma=2.0 -> eps={tight.achieved_epsilon:.3f}, "
          f"sigma=0.5 -> eps={loose.achieved_epsilon:.3f}")

    table = per_client_accounting_table(
        client_dataset_sizes=[60, 92, 127], expected_batch_size=32,
        noise_multiplier=1.0, local_epochs=3, n_rounds=5, delta=1e-5,
    )
    assert len(table) == 3
    print(f"[OK] per_client_accounting_table produced {len(table)} rows")


def test_secure_aggregation_weighted_correctness():
    rng = np.random.RandomState(0)
    dim = 200
    raw_updates = {
        "A": rng.randn(dim).astype(np.float32),
        "B": rng.randn(dim).astype(np.float32),
        "C": rng.randn(dim).astype(np.float32),
    }
    n_k = {"A": 60, "B": 92, "C": 127}
    N = sum(n_k.values())
    weights = {k: v / N for k, v in n_k.items()}

    session_secret = os.urandom(32)
    masked = {
        cid: compute_masked_update(cid, list(raw_updates.keys()), raw_updates[cid], weights[cid], session_secret)
        for cid in raw_updates
    }
    aggregated = server_side_sum(list(masked.values()))
    err = aggregation_correctness_check(raw_updates, weights, aggregated)
    assert err < 1e-4, f"secure-aggregation weighted correctness failed, max abs error = {err}"
    print(f"[OK] secure aggregation (weighted): max abs error vs. true weighted mean = {err:.2e}")

    # AES-GCM round trip on one client's masked update
    key = generate_client_key()
    payload = encrypt_update(key, masked["A"])
    recovered = decrypt_update(key, payload, expected_len=dim)
    assert np.allclose(recovered, masked["A"], atol=1e-6)
    print("[OK] AES-GCM encrypt/decrypt round trip matches original masked update")


def test_evaluation_utilities():
    rng = np.random.RandomState(0)
    n = 500
    y_true = (rng.rand(n) < 0.15).astype(int)
    # Deliberately noisy, non-degenerate scores (moderate separability) so the
    # bootstrap CI below has real width to demonstrate, rather than collapsing
    # to a point because the synthetic scores are perfectly separable.
    y_prob_train = np.clip(0.15 + 0.25 * y_true + rng.normal(0, 0.20, n), 0, 1)
    tau = youden_threshold(y_true, y_prob_train)
    assert 0.0 <= tau <= 1.0

    ci_auroc = bootstrap_ci(y_true, y_prob_train, metric_auroc, n_resamples=200)
    assert ci_auroc.ci_low <= ci_auroc.point_estimate <= ci_auroc.ci_high
    print(f"[OK] bootstrap_ci(AUROC) = {ci_auroc.point_estimate:.3f} "
          f"({ci_auroc.ci_low:.3f}, {ci_auroc.ci_high:.3f})")

    y_prob_b = np.clip(y_prob_train + rng.normal(0, 0.10, n), 0, 1)
    paired = paired_bootstrap_test(y_true, y_prob_train, y_prob_b, metric_auroc, n_resamples=200)
    print(f"[OK] paired_bootstrap_test mean_diff={paired.mean_diff:.4f}, "
          f"CI=({paired.ci_low:.4f}, {paired.ci_high:.4f}), significant={paired.significant_at_alpha}")


if __name__ == "__main__":
    df, train_df, test_df, artifacts, train_proc, test_proc = test_data_and_preprocessing()
    test_augmentation(train_df)
    test_conventional_models(train_proc, test_proc, train_df, test_df)
    test_resmlp_dp_forward()
    test_client_updates_and_federated_orchestration(train_proc, train_df)
    test_privacy_accounting()
    test_secure_aggregation_weighted_correctness()
    test_evaluation_utilities()
    print("\nALL SMOKE TESTS PASSED")
