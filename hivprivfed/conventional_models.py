"""
The ten conventional classifiers compared in Table 2 (LR, RBF-SVM, RF,
ExtraTrees, HistGB, XGBoost, LightGBM, CatBoost, BRF, EasyEnsemble), plus the
shared evaluation harness (accuracy, balanced accuracy, sensitivity,
specificity, MCC, AUROC, AUPRC, Brier score, ECE) used across both the
"Augmented" and "Real-only" training conditions.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict, Callable

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, matthews_corrcoef,
    roc_auc_score, average_precision_score, brier_score_loss,
    confusion_matrix,
)
from imblearn.ensemble import BalancedRandomForestClassifier, EasyEnsembleClassifier
import xgboost as xgb
import lightgbm as lgb
import catboost as cb


def build_model_zoo(random_state: int = 42) -> Dict[str, Callable[[], object]]:
    """Factory functions (not instances) so each run gets a fresh, unfitted model."""
    return {
        "LR": lambda: LogisticRegression(max_iter=2000, random_state=random_state),
        "RBF-SVM": lambda: SVC(kernel="rbf", probability=True, random_state=random_state),
        "RF": lambda: RandomForestClassifier(n_estimators=300, random_state=random_state, n_jobs=-1),
        "ExtraTrees": lambda: ExtraTreesClassifier(n_estimators=300, random_state=random_state, n_jobs=-1),
        "HistGB": lambda: HistGradientBoostingClassifier(random_state=random_state),
        "XGBoost": lambda: xgb.XGBClassifier(
            n_estimators=300, use_label_encoder=False, eval_metric="logloss",
            random_state=random_state, verbosity=0,
        ),
        "LightGBM": lambda: lgb.LGBMClassifier(n_estimators=300, random_state=random_state, verbose=-1),
        "CatBoost": lambda: cb.CatBoostClassifier(
            n_estimators=300, random_state=random_state, verbose=False
        ),
        "BRF": lambda: BalancedRandomForestClassifier(
            n_estimators=300, random_state=random_state, n_jobs=-1,
            sampling_strategy="all", replacement=True, bootstrap=False,
        ),
        "EasyEnsemble": lambda: EasyEnsembleClassifier(n_estimators=20, random_state=random_state, n_jobs=-1),
    }


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Standard equal-width-bin ECE."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi if i < n_bins - 1 else y_prob <= hi)
        if mask.sum() == 0:
            continue
        bin_acc = y_true[mask].mean()
        bin_conf = y_prob[mask].mean()
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return float(ece)


@dataclass
class ClassificationMetrics:
    accuracy: float
    balanced_accuracy: float
    sensitivity: float
    specificity: float
    mcc: float
    auroc: float
    auprc: float
    brier: float
    ece: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "Accuracy": self.accuracy, "Bal_Acc": self.balanced_accuracy,
            "Sensitivity": self.sensitivity, "Specificity": self.specificity,
            "MCC": self.mcc, "AUROC": self.auroc, "AUPRC": self.auprc,
            "Brier": self.brier, "ECE": self.ece,
        }


def evaluate_predictions(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> ClassificationMetrics:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")

    return ClassificationMetrics(
        accuracy=accuracy_score(y_true, y_pred),
        balanced_accuracy=balanced_accuracy_score(y_true, y_pred),
        sensitivity=sensitivity,
        specificity=specificity,
        mcc=matthews_corrcoef(y_true, y_pred) if len(np.unique(y_pred)) > 1 else 0.0,
        auroc=roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan"),
        auprc=average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan"),
        brier=brier_score_loss(y_true, y_prob),
        ece=expected_calibration_error(y_true, y_prob),
    )


def get_predict_proba(model, X: np.ndarray) -> np.ndarray:
    """Uniform positive-class probability extraction across all ten model types."""
    proba = model.predict_proba(X)
    return proba[:, 1]


def train_and_evaluate(
    model_name: str,
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    threshold: float = 0.5,
    random_state: int = 42,
) -> ClassificationMetrics:
    zoo = build_model_zoo(random_state=random_state)
    if model_name not in zoo:
        raise KeyError(f"Unknown model '{model_name}'. Available: {list(zoo)}")
    model = zoo[model_name]()
    model.fit(X_train, y_train)
    y_prob = get_predict_proba(model, X_test)
    return evaluate_predictions(y_test, y_prob, threshold=threshold)
