"""
Leakage-safe preprocessing pipeline.

Implements, in the order described in the manuscript:
  1. Participant-level 50:50 holdout, jointly stratified by city and HIV label
     (fit BEFORE any statistic estimation -- this ordering matters and is the
     opposite of what a schematic "acquisition -> preprocessing -> split"
     diagram might suggest; see the figure-caption fix discussed in review
     issue #11).
  2. Winsorization at the 1st/99th percentile (Eq. winsorization), fit on train only.
  3. log(1+x) transform for count-valued variables (Eq. log_transform).
  4. Median/IQR robust scaling (Eq. robust_scaling), fit on train only.
  5. 7-NN inverse-distance-weighted imputation (Eq. knn_imputation / knn_weight),
     with binary missingness indicators retained.
  6. One-hot encoding of categorical variables, fit on train only.

All fitted statistics (percentiles, median/IQR, imputer, encoder) come ONLY
from the training partition and are applied unchanged to the test partition.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder

from .config import DATA


@dataclass
class PreprocessingArtifacts:
    """Everything fit on the training partition, needed to transform new data identically."""
    numeric_cols: List[str]
    count_cols: List[str]
    categorical_cols: List[str]
    winsor_low: pd.Series = None
    winsor_high: pd.Series = None
    median_: pd.Series = None
    iqr_: pd.Series = None
    knn_imputer: Optional[KNNImputer] = None
    ohe: Optional[OneHotEncoder] = None
    missingness_cols: List[str] = field(default_factory=list)


def participant_level_split(
    df: pd.DataFrame,
    label_col: str,
    city_col: str,
    holdout_fraction: float = DATA.holdout_fraction,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Strict participant-level split, jointly stratified by (city, label) so both
    geographic distribution and class balance are preserved in each half.
    Returns (train_df, test_df) with train_df.index and test_df.index disjoint.
    """
    strata = df[city_col].astype(str) + "_" + df[label_col].astype(str)
    train_df, test_df = train_test_split(
        df,
        test_size=holdout_fraction,
        stratify=strata,
        random_state=random_state,
    )
    assert set(train_df.index).isdisjoint(set(test_df.index)), "train/test participant overlap detected"
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def _winsorize_fit(train: pd.DataFrame, cols: List[str], low_q: float, high_q: float):
    return train[cols].quantile(low_q), train[cols].quantile(high_q)


def _winsorize_apply(df: pd.DataFrame, cols: List[str], low: pd.Series, high: pd.Series) -> pd.DataFrame:
    out = df.copy()
    out[cols] = df[cols].clip(lower=low, upper=high, axis=1)
    return out


def _log_transform(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    out[cols] = np.log1p(out[cols].clip(lower=0))
    return out


def _robust_scale_fit(train: pd.DataFrame, cols: List[str]):
    median_ = train[cols].median()
    iqr_ = train[cols].quantile(0.75) - train[cols].quantile(0.25)
    return median_, iqr_


def _robust_scale_apply(df: pd.DataFrame, cols: List[str], median_: pd.Series, iqr_: pd.Series, eps: float) -> pd.DataFrame:
    out = df.copy()
    out[cols] = (df[cols] - median_) / (iqr_ + eps)
    return out


def add_missingness_indicators(df: pd.DataFrame, cols: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    out = df.copy()
    new_cols = []
    for c in cols:
        ind_col = f"{c}__was_missing"
        out[ind_col] = df[c].isna().astype(int)
        new_cols.append(ind_col)
    return out, new_cols


def fit_preprocessing(
    train_df: pd.DataFrame,
    numeric_cols: List[str],
    count_cols: List[str],
    categorical_cols: List[str],
) -> PreprocessingArtifacts:
    """Fit every preprocessing statistic on the training partition only."""
    artifacts = PreprocessingArtifacts(
        numeric_cols=numeric_cols, count_cols=count_cols, categorical_cols=categorical_cols
    )

    all_numeric = list(dict.fromkeys(numeric_cols + count_cols))  # winsorize both groups
    artifacts.winsor_low, artifacts.winsor_high = _winsorize_fit(
        train_df, all_numeric, DATA.winsor_low_q, DATA.winsor_high_q
    )

    w = _winsorize_apply(train_df, all_numeric, artifacts.winsor_low, artifacts.winsor_high)
    w = _log_transform(w, count_cols)

    artifacts.median_, artifacts.iqr_ = _robust_scale_fit(w, all_numeric)

    scaled = _robust_scale_apply(w, all_numeric, artifacts.median_, artifacts.iqr_, DATA.eps_div)
    artifacts.knn_imputer = KNNImputer(
        n_neighbors=DATA.knn_neighbors,
        weights="distance",  # inverse-distance weighting, matching Eq. knn_weight
    )
    artifacts.knn_imputer.fit(scaled[all_numeric])

    if categorical_cols:
        cat_filled = train_df[categorical_cols].astype("object").fillna("__MISSING__")
        artifacts.ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        artifacts.ohe.fit(cat_filled)

    return artifacts


def apply_preprocessing(df: pd.DataFrame, artifacts: PreprocessingArtifacts) -> pd.DataFrame:
    """Apply already-fitted preprocessing to any partition (train or test)."""
    all_numeric = list(dict.fromkeys(artifacts.numeric_cols + artifacts.count_cols))

    out = df.copy()
    out, missingness_cols = add_missingness_indicators(out, all_numeric)

    out = _winsorize_apply(out, all_numeric, artifacts.winsor_low, artifacts.winsor_high)
    out = _log_transform(out, artifacts.count_cols)
    out = _robust_scale_apply(out, all_numeric, artifacts.median_, artifacts.iqr_, DATA.eps_div)

    imputed = artifacts.knn_imputer.transform(out[all_numeric])
    out[all_numeric] = imputed

    frames = [out.drop(columns=all_numeric)]
    numeric_frame = out[all_numeric].reset_index(drop=True)
    frames = [numeric_frame]

    if artifacts.categorical_cols and artifacts.ohe is not None:
        cat_filled = df[artifacts.categorical_cols].astype("object").fillna("__MISSING__").reset_index(drop=True)
        cat_encoded = artifacts.ohe.transform(cat_filled)
        cat_df = pd.DataFrame(
            cat_encoded,
            columns=artifacts.ohe.get_feature_names_out(artifacts.categorical_cols),
        )
        frames.append(cat_df)

    miss_df = out[missingness_cols].reset_index(drop=True)
    frames.append(miss_df)

    result = pd.concat(frames, axis=1)
    return result
