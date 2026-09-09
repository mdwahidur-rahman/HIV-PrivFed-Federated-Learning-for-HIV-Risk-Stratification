"""
Central configuration for HIV-PrivFed.

Every value here is taken directly from the manuscript
("HIV-PrivFed: Privacy-Utility-Fairness-Aware Federated Learning for HIV Risk
Stratification and Testing Prioritization under Non-IID Clinical Heterogeneity").
Where the manuscript's own review flagged a value as ambiguous or unconfirmed,
that is noted explicitly in a comment -- this file does not silently pick an
answer to an open question.
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class DataConfig:
    input_dim: int = 51                 # d = 51 (Section: Network Architecture)
    n_clients: int = 13                 # K = 13 Sialon-II cities
    holdout_fraction: float = 0.5       # participant-level 50:50 holdout
    knn_neighbors: int = 7              # 7-NN imputation (Eq. knn_imputation)
    winsor_low_q: float = 0.01          # Q_{0.01,j}
    winsor_high_q: float = 0.99         # Q_{0.99,j}
    eps_div: float = 1e-8               # epsilon added to denominators (scaling, KNN weights)

    # NOTE (see review issue #2 / the manuscript's own % TODO(authors) comment):
    # whether the "real-only" conventional-ML condition in Table 2 trains on the
    # FULL original real training partition (2,358 participants) or on the
    # UNDERSAMPLED real subset (1,427 participants) that feeds the augmented/
    # balanced set is not settled in the manuscript text alone. This flag makes
    # that choice explicit and auditable instead of silently hard-coded.
    real_only_uses_full_original_partition: bool = True


@dataclass
class AugmentationConfig:
    target_ratio: float = 1.0           # SMOTENC + undersampling -> 1:1 balance (Eq. balanced_training)


@dataclass
class ModelConfig:
    hidden_1: int = 64                  # W1 in R^{64 x d}
    hidden_2: int = 64                  # W2 in R^{64 x 64} (residual block)
    hidden_3: int = 32                  # W3 in R^{32 x 64}
    output_dim: int = 1                 # W4 in R^{1 x 32}
    group_norm_groups: int = 8          # GroupNorm groups (not specified numerically in the
                                         # manuscript; 8 is a standard default for width-64 layers
                                         # and must divide the channel count evenly -- confirm/tune
                                         # against your own validation set before treating as final)


@dataclass
class LossConfig:
    focal_gamma: float = 2.0            # gamma = 2 (Eq. focal)
    fedprox_mu: float = 0.01            # mu = 0.01 (Eq. fedprox)


@dataclass
class DPConfig:
    clip_norm: float = 1.0              # C = 1.0 (Eq. clip)
    learning_rate: float = 5e-3         # eta = 5e-3
    local_epochs: int = 3               # E = 3
    communication_rounds: int = 30      # T = 30
    delta: float = 1e-5                 # delta = 1e-5
    noise_multipliers: List[float] = field(default_factory=lambda: [2.0, 1.2, 0.8, 0.5])
    nominal_epsilon_targets: List[float] = field(default_factory=lambda: [1, 2, 4, 8])
    batch_size: int = 32                 # Poisson-sampled lot size (expected); not stated
                                          # explicitly in the manuscript for the DP runs --
                                          # confirm against your own scripts.


@dataclass
class EvalConfig:
    n_bootstrap: int = 1000             # 1,000-resample bootstrap (Section: Statistical Robustness)
    ci_level: float = 0.95
    random_seed: int = 42
    min_positives_for_client_report: int = 3   # cities with < 3 positive test cases excluded (Table 8 caption)


@dataclass
class SecureAggConfig:
    aes_key_bits: int = 256             # AES-GCM-256
    mask_seed_bits: int = 128


DATA = DataConfig()
AUG = AugmentationConfig()
MODEL = ModelConfig()
LOSS = LossConfig()
DP = DPConfig()
EVAL = EvalConfig()
SECAGG = SecureAggConfig()

# The 13 Sialon-II cities, spelled consistently (the manuscript's Table 1/8 misspell
# Ljubljana as "Lubljana" in several rows -- use the corrected spelling going forward).
SIALON_CITIES = [
    "Barcelona", "Bratislava", "Brighton", "Brussels", "Bucharest", "Hamburg",
    "Lisbon", "Ljubljana", "Sofia", "Stockholm", "Verona", "Vilnius", "Warsaw",
]
assert len(SIALON_CITIES) == DATA.n_clients
