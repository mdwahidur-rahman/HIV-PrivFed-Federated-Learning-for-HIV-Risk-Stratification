# HIV-PrivFed — Reference Implementation

Code accompanying *"HIV-PrivFed: Privacy–Utility–Fairness-Aware Federated
Learning for HIV Risk Stratification and Testing Prioritization under
Non-IID Clinical Heterogeneity."*

## Read this before anything else

**This is a from-specification reference implementation, not a recovery of
the original experiment scripts.** It was written from the manuscript's own
equations, algorithm listings (Algorithms 1–2), and stated hyperparameters —
not from the authors' original code, which this repository's authors did not
have access to when writing it.

Concretely, that means:

- Every architectural and algorithmic choice traces to a specific equation or
  stated value in the manuscript (see [Manuscript → code map](#manuscript--code-map)
  below). Where the manuscript did **not** specify a value precisely (e.g. the
  GroupNorm group count, the DP-SGD batch size), that is flagged in
  [`hivprivfed/config.py`](hivprivfed/config.py) as an explicit, documented
  default — not silently guessed.
- **It will not reproduce the manuscript's exact reported numbers** unless run
  on the real, access-controlled Sialon-II data, using the exact same
  preprocessing decisions, hyperparameters, and random seeds the original
  experiments used.
- One specific open question is left as a visible `# TODO(authors)` comment in
  the code and in the manuscript text itself: whether the "Real-only"
  conventional-ML condition trained on the full 2,358-participant original
  real partition or the 1,427-participant undersampled subset. This is
  controlled by a single flag —
  `config.DATA.real_only_uses_full_original_partition` — rather than hidden
  inside logic. **Confirm this against your own training scripts before
  trusting any Real-only numbers this code produces.**
- This repository ships a synthetic, Sialon-II-*shaped* dataset generator
  (`hivprivfed/synthetic_data.py`) purely so the pipeline can be run and
  tested without a data-use agreement in place. Every number in this README's
  examples comes from that synthetic data and is illustrative only.

If you are a reviewer or reader checking this repository against the paper:
please treat close-but-not-exact methodological correspondence as the
expected state of a from-specification reimplementation, and treat any exact
numeric match to the manuscript's tables as coincidental unless the real data
and seeds were used.

## What's implemented

| Manuscript element | Code |
|---|---|
| Winsorization, log-transform, robust scaling, 7-NN imputation, one-hot encoding (Eqs. `winsorization`, `log_transform`, `robust_scaling`, `knn_imputation`, `knn_weight`) | [`hivprivfed/preprocessing.py`](hivprivfed/preprocessing.py) |
| Participant-level 50:50 holdout, jointly stratified by city + label | [`preprocessing.participant_level_split`](hivprivfed/preprocessing.py) |
| City-conditioned SMOTENC + proportional undersampling (Eqs. `smotenc`, `smote_lambda`) | [`hivprivfed/augmentation.py`](hivprivfed/augmentation.py) |
| The ten conventional classifiers (Table 2: LR, RBF-SVM, RF, ExtraTrees, HistGB, XGBoost, LightGBM, CatBoost, BRF, EasyEnsemble) | [`hivprivfed/conventional_models.py`](hivprivfed/conventional_models.py) |
| ResMLP-DP architecture, ~9,857 parameters (Eqs. `layer1`–`sigmoid`) | [`hivprivfed/resmlp_dp.py`](hivprivfed/resmlp_dp.py) |
| Focal loss + weighted BCE + FedProx proximal term (Eqs. `pt`, `wbce`, `focal`, `fedprox`) | [`hivprivfed/losses.py`](hivprivfed/losses.py) |
| Per-example DP-SGD via Opacus — Algorithm 2 (Eqs. `clip`, `dpsgd`) | [`hivprivfed/dp_training.py`](hivprivfed/dp_training.py) |
| FedAvg/FedProx/local-only/HIV-PrivFed orchestration — Algorithm 1 (Eq. `fedavg`) | [`hivprivfed/federated.py`](hivprivfed/federated.py) |
| Rényi-DP accounting via Opacus's own accountant (Eq. `rdp2dp`) | [`hivprivfed/privacy_accounting.py`](hivprivfed/privacy_accounting.py) |
| Bonawitz-style pairwise-masked secure aggregation (**with weighted-sum cancellation correctly derived** — see module docstring) + AES-GCM transport encryption | [`hivprivfed/secure_aggregation.py`](hivprivfed/secure_aggregation.py) |
| Youden's *J* threshold selection on training data only (Eq. `youden`) | [`evaluation.youden_threshold`](hivprivfed/evaluation.py) |
| Bootstrap CIs + paired bootstrap significance testing | [`hivprivfed/evaluation.py`](hivprivfed/evaluation.py) |

Three runnable scripts reproduce the *structure* (not the exact numbers) of
the manuscript's main comparisons:

| Script | Reproduces the structure of |
|---|---|
| [`scripts/run_conventional_ml.py`](scripts/run_conventional_ml.py) | Table 2 — 10 classifiers × {Augmented, Real-only} |
| [`scripts/run_federated.py`](scripts/run_federated.py) | Table 5's non-private rows — local-only, FedAvg, FedProx |
| [`scripts/run_privacy_sweep.py`](scripts/run_privacy_sweep.py) | Tables 6–7 — HIV-PrivFed across all four noise multipliers, with per-client privacy accounting |

**Not yet implemented** (flagged rather than silently omitted): figure
generation for Figures 1–8, leave-one-city-out validation, multi-seed
variance reporting, and the additional federated-LR / DP-federated-LR /
clipping-only-FedProx baselines recommended during review. See
[Open items](#open-items) below.

## Repository structure

```
hiv-privfed-repo/
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
├── hivprivfed/                  # the installable package
│   ├── config.py                # every hyperparameter, with ambiguities flagged
│   ├── preprocessing.py
│   ├── augmentation.py
│   ├── conventional_models.py
│   ├── resmlp_dp.py
│   ├── losses.py
│   ├── dp_training.py
│   ├── federated.py
│   ├── privacy_accounting.py
│   ├── secure_aggregation.py
│   ├── evaluation.py
│   ├── synthetic_data.py        # Sialon-II-*shaped* data for testing, NOT real data
│   └── utils.py
├── scripts/                     # runnable entry points
│   ├── run_conventional_ml.py
│   ├── run_federated.py
│   └── run_privacy_sweep.py
└── tests/
    └── test_pipeline_smoke.py   # end-to-end smoke test on synthetic data
```

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Tested on Python 3.12 with the package versions pinned in
`requirements.txt`. If `torch` fails to install for your platform/CUDA
version, install it separately first following
[pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/),
then re-run `pip install -r requirements.txt` for the rest.

## Quickstart (synthetic data — runs out of the box)

```bash
# Smoke-test every module end-to-end (~1-2 minutes on CPU)
python tests/test_pipeline_smoke.py

# Table 2-style conventional ML comparison (10 models x 2 training conditions)
python scripts/run_conventional_ml.py

# Table 5-style federated comparison (local-only / FedAvg / FedProx)
python scripts/run_federated.py --rounds 5

# Tables 6-7-style privacy sweep across all four noise multipliers
python scripts/run_privacy_sweep.py --rounds 5
```

Every script prints `[info] --data not given: using synthetic data` when run
without `--data` — that is your reminder that the printed numbers are not
paper results.

## Getting the real data

Sialon-II is a real, access-controlled bio-behavioural cohort. It is **not**
included here and is **not** something this repository can grant access to.
See the manuscript's Data Availability Statement and Sialon-II's own
publications for the access process. Once you have a real extract:

1. Ensure it has a label column (HIV serostatus) and a city column.
2. Edit the `numeric_cols` / `count_cols` / `categorical_cols` lists at the
   top of each script's `main()` to match your actual feature dictionary —
   these are deliberately **not** auto-inferred, since guessing a real
   clinical feature's type wrong (e.g. treating an ordinal scale as free-form
   numeric) can silently distort preprocessing.
3. Resolve the `# TODO(authors)` cohort question in
   [`hivprivfed/preprocessing.py`](hivprivfed/preprocessing.py) /
   `config.DATA.real_only_uses_full_original_partition` against your actual
   prior training scripts before trusting Real-only numbers.
4. Run with `--data path/to/your_extract.csv`.

## Testing

```bash
pip install pytest
python -m pytest tests/ -v
# or, without pytest:
python tests/test_pipeline_smoke.py
```

The smoke test validates that the pipeline *runs correctly and its
invariants hold* (e.g. masks cancel in secure aggregation to <1e-6 error,
achieved ε increases as the noise multiplier decreases, no NaNs survive
imputation) — it does **not** validate agreement with the manuscript's
reported numbers, which requires the real data.

## Open items

These are known gaps, most of them raised during manuscript review, that this
codebase does not resolve on its own:

- [ ] **Cohort confirmation** (see `# TODO(authors)` in `preprocessing.py`) —
      which cohort trained the Table 2 "Real-only" rows.
- [ ] **Figure generation** for Figures 1–8 is not yet implemented.
- [ ] **Multi-seed variance** — the manuscript's federated/private results are
      currently single-seed point estimates; `evaluation.multi_seed_summary`
      is provided but needs to be wired into a multi-seed training loop
      (≥5 seeds recommended).
- [ ] **Additional baselines** — federated logistic regression, DP-federated
      logistic regression, and a clipping-only (no-noise) FedProx ablation are
      not yet implemented.
- [ ] **End-to-end DP guarantee audit** — this codebase's DP-SGD covers the
      neural network training step only. It does *not* separately account for
      privacy leakage through data-dependent preprocessing statistics
      (winsorization thresholds, robust-scaling median/IQR), data-dependent
      focal-loss class weights, or pooled-training-data threshold selection.
      Treat the achieved ε values as covering model training only, not the
      full pipeline, until that audit is done.
- [ ] **Secure aggregation** here implements pairwise masking and AES-GCM as
      distinct, measurable mechanisms, but does not implement Bonawitz et
      al.'s full protocol (key establishment, dropout recovery, collusion
      thresholds). It is suitable for measuring the *arithmetic correctness*
      and *overhead* of masking/encryption, not for a production security
      claim.

## Citation

```bibtex
@article{rahman2026hivprivfed,
  title   = {HIV-PrivFed: Privacy--Utility--Fairness-Aware Federated Learning
             for HIV Risk Stratification and Testing Prioritization under
             Non-IID Clinical Heterogeneity},
  author  = {Rahman, Md Wahidur and Adarbah, Haitham Y. and Pasha, Atena and Noore, Afzel},
  year    = {2026},
  journal = {Will add},
  note    = {Will add}
}
```

## License

Code: MIT — see [LICENSE](LICENSE). This does **not** cover the Sialon-II
dataset, which is governed by its own access conditions.
