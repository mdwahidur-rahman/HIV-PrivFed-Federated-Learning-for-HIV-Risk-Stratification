"""
HIV-PrivFed reference implementation.

This package is a from-specification reference implementation of the methods
described in the HIV-PrivFed manuscript (preprocessing, city-conditioned
SMOTENC augmentation, conventional ML benchmarking, ResMLP-DP, DP-SGD via
Opacus, FedAvg/FedProx orchestration, an RDP privacy accountant, Bonawitz-style
secure aggregation with AES-GCM, and bootstrap-based statistical evaluation).

IMPORTANT: this is NOT a recovery of the authors' original scripts. It was
written from the manuscript's equations, algorithm listings, and stated
hyperparameters. It is expected to be methodologically faithful, but it will
not reproduce the manuscript's exact reported numbers unless (a) run on the
real Sialon-II data under a genuine data-use agreement, and (b) the open
methodological questions flagged throughout this codebase (see config.py and
each module's docstring) are resolved against the authors' own scripts first.
"""

__version__ = "0.1.0"
