"""
Synthetic data generator for testing.

This does NOT reproduce Sialon-II. It generates a dataset with the same
SHAPE described in the manuscript -- 13 named cities, per-city HIV prevalence
varying across a similar range (roughly 4%-30%), a mix of numeric/count/
categorical feature types, and a comparable total participant count -- purely
so the rest of this codebase (preprocessing, augmentation, conventional ML,
federated training, DP-SGD, secure aggregation, bootstrap evaluation) can be
exercised end-to-end in CI or on a laptop with no data-use agreement in place.

Replace this module's output with your real, access-controlled Sialon-II
extract before generating any number you intend to publish.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import SIALON_CITIES

# Approximate per-city (n_participants, hiv_prevalence) used only to make the
# synthetic data's SHAPE resemble Table 1 -- these are illustrative round
# numbers, not the manuscript's real per-city counts.
_CITY_PROFILE = {
    "Barcelona": (250, 0.20), "Bratislava": (230, 0.07), "Brighton": (255, 0.22),
    "Brussels": (205, 0.16), "Bucharest": (130, 0.27), "Hamburg": (215, 0.18),
    "Lisbon": (245, 0.24), "Ljubljana": (135, 0.07), "Sofia": (140, 0.07),
    "Stockholm": (140, 0.08), "Verona": (200, 0.11), "Vilnius": (120, 0.04),
    "Warsaw": (230, 0.19),
}

NUMERIC_COLS = ["age", "n_sex_partners_12mo", "n_condomless_ai_12mo", "substance_use_score"]
COUNT_COLS = ["n_sex_partners_12mo", "n_condomless_ai_12mo"]
CATEGORICAL_COLS = ["education", "migration_status", "outness", "sexual_role"]


def generate_synthetic_sialon(random_state: int = 42, missing_rate: float = 0.05) -> pd.DataFrame:
    rng = np.random.RandomState(random_state)
    frames = []

    for city, (n, prev) in _CITY_PROFILE.items():
        n_pos = max(1, int(round(n * prev)))
        n_neg = n - n_pos
        labels = np.array([1] * n_pos + [0] * n_neg)
        rng.shuffle(labels)

        age = rng.normal(34, 9, size=n).clip(18, 75)
        # Positives skewed toward slightly higher partner/condomless counts,
        # purely to give downstream classifiers *something* nonrandom to find.
        base_partners = rng.poisson(3, size=n) + labels * rng.poisson(2, size=n)
        base_condomless = rng.poisson(1, size=n) + labels * rng.poisson(1, size=n)
        substance_score = rng.normal(2 + labels * 0.8, 1.5, size=n).clip(0, 10)

        education = rng.choice(["primary", "secondary", "tertiary"], size=n, p=[0.15, 0.45, 0.40])
        migration = rng.choice(["native", "eu_migrant", "non_eu_migrant"], size=n, p=[0.7, 0.2, 0.1])
        outness = rng.choice(["out", "partially_out", "not_out"], size=n, p=[0.5, 0.3, 0.2])
        sexual_role = rng.choice(["top", "bottom", "versatile", "prefer_not_say"], size=n, p=[0.25, 0.25, 0.4, 0.1])

        df = pd.DataFrame({
            "participant_id": [f"{city}_{i:04d}" for i in range(n)],
            "city": city,
            "hiv_status": labels,
            "age": age,
            "n_sex_partners_12mo": base_partners,
            "n_condomless_ai_12mo": base_condomless,
            "substance_use_score": substance_score,
            "education": education,
            "migration_status": migration,
            "outness": outness,
            "sexual_role": sexual_role,
        })
        frames.append(df)

    full = pd.concat(frames, axis=0).reset_index(drop=True)

    # Inject missing-at-random values so the KNN-imputation path is exercised.
    for col in NUMERIC_COLS:
        mask = rng.rand(len(full)) < missing_rate
        full.loc[mask, col] = np.nan
    for col in CATEGORICAL_COLS:
        mask = rng.rand(len(full)) < missing_rate
        full.loc[mask, col] = np.nan

    return full
