"""
Secure aggregation: Bonawitz-style pairwise additive masking for update-level
protection from an honest-but-curious server, plus AES-GCM for transport-level
confidentiality of each client's masked update. These are two independent
protections; AES-GCM alone (encrypting an update) does NOT provide the
secure-aggregation property (hiding each client's individual update from the
server), and pairwise masking alone does not provide transport confidentiality
against a network eavesdropper -- both are implemented and measured here.

WEIGHTED AGGREGATION (review issue #10):
Plain Bonawitz masking cancels only under an UNWEIGHTED sum: if client i adds
+mask(i,j) and client j adds -mask(i,j) to their raw updates, summing all
masked updates cancels every pairwise term exactly. FedAvg, however, needs a
WEIGHTED sum (n_k / N per client), and unequal weights break that
cancellation if applied naively (masking the raw update, then having the
server itself apply per-client weights on TOP of already-masked values, would
leave residual, non-cancelling mask terms).

The correct fix implemented here: each client pre-multiplies its OWN update by
its OWN weight (n_k / N) BEFORE masking. The server then performs a plain,
UNWEIGHTED sum of the masked, already-weighted updates. Because masking is
applied after weighting, cancellation still holds exactly, and the server's
unweighted sum yields the correctly weighted FedAvg result directly, with no
further division needed.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from .config import SECAGG


def _prg_from_seed(seed: bytes, length: int) -> np.ndarray:
    """
    Deterministic pseudo-random vector of `length` float32 values in
    [-0.5, 0.5), derived from a shared seed via repeated SHA-256 expansion.
    Any two parties holding the same seed derive an identical mask.
    """
    out = np.empty(length, dtype=np.float32)
    counter = 0
    filled = 0
    while filled < length:
        block = hashlib.sha256(seed + counter.to_bytes(8, "big")).digest()
        ints = np.frombuffer(block, dtype=np.uint32)
        vals = (ints.astype(np.float64) / np.iinfo(np.uint32).max) - 0.5
        take = min(len(vals), length - filled)
        out[filled:filled + take] = vals[:take].astype(np.float32)
        filled += take
        counter += 1
    return out


def pairwise_seed(client_i: str, client_j: str, session_secret: bytes) -> bytes:
    """
    A symmetric, order-independent seed for the (i, j) pair, derived from a
    shared session secret. In a real deployment this would come from a
    Diffie-Hellman key agreement per pair; here it is simulated with a shared
    session secret for measurement purposes (see README for what is and is
    not a full protocol implementation).
    """
    a, b = sorted([client_i, client_j])
    return hashlib.sha256(session_secret + a.encode() + b.encode()).digest()


def compute_masked_update(
    client_id: str,
    all_client_ids: List[str],
    raw_update: np.ndarray,
    weight: float,
    session_secret: bytes,
) -> np.ndarray:
    """
    weighted_update = weight * raw_update
    masked_update    = weighted_update + sum_{j > i} mask(i,j) - sum_{j < i} mask(i,j)

    The sign convention (add for the lexicographically-first client in each
    pair, subtract for the other) guarantees that summing every client's
    masked_update across the whole cohort cancels all pairwise terms exactly.
    """
    weighted = raw_update.astype(np.float32) * np.float32(weight)
    masked = weighted.copy()
    for other in all_client_ids:
        if other == client_id:
            continue
        seed = pairwise_seed(client_id, other, session_secret)
        mask = _prg_from_seed(seed, len(raw_update))
        if client_id < other:
            masked += mask
        else:
            masked -= mask
    return masked


def server_side_sum(masked_updates: List[np.ndarray]) -> np.ndarray:
    """Plain, UNWEIGHTED sum -- masks cancel here because weighting already
    happened client-side before masking (see module docstring)."""
    return np.sum(np.stack(masked_updates, axis=0), axis=0)


def aggregation_correctness_check(
    raw_updates: Dict[str, np.ndarray],
    weights: Dict[str, float],
    aggregated: np.ndarray,
) -> float:
    """Max absolute error between the securely-aggregated result and the
    plaintext weighted mean -- matches the manuscript's reported
    'Aggregation correctness (max abs error vs true mean)' metric."""
    true_mean = np.sum(
        [weights[cid] * raw_updates[cid].astype(np.float32) for cid in raw_updates], axis=0
    )
    return float(np.max(np.abs(aggregated - true_mean)))


# --------------------------------------------------------------------------
# AES-GCM transport encryption (independent of the masking above)
# --------------------------------------------------------------------------

@dataclass
class EncryptedPayload:
    nonce: bytes
    ciphertext: bytes  # includes the GCM authentication tag, per AESGCM's API


def generate_client_key() -> bytes:
    return AESGCM.generate_key(bit_length=SECAGG.aes_key_bits)


def encrypt_update(key: bytes, update: np.ndarray, associated_data: bytes = b"") -> EncryptedPayload:
    aesgcm = AESGCM(key)
    nonce = os.urandom(12)  # 96-bit nonce, standard for AES-GCM
    plaintext = update.astype(np.float32).tobytes()
    ciphertext = aesgcm.encrypt(nonce, plaintext, associated_data)
    return EncryptedPayload(nonce=nonce, ciphertext=ciphertext)


def decrypt_update(key: bytes, payload: EncryptedPayload, expected_len: int, associated_data: bytes = b"") -> np.ndarray:
    aesgcm = AESGCM(key)
    plaintext = aesgcm.decrypt(payload.nonce, payload.ciphertext, associated_data)
    return np.frombuffer(plaintext, dtype=np.float32).reshape(expected_len)
