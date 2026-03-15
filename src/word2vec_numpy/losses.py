"""Numerically stable loss primitives for skip-gram with negative sampling."""

from __future__ import annotations

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Element-wise sigmoid with numerical stability.

    Uses the identity ``sigmoid(x) = 1 / (1 + exp(-x))`` rearranged to
    avoid overflow for large negative *x*:

    - When x ≥ 0:  σ(x) = 1 / (1 + exp(−x))
    - When x < 0:  σ(x) = exp(x) / (1 + exp(x))

    This avoids computing ``exp(large_positive)`` which would overflow.
    """
    pos_mask = x >= 0
    neg_mask = ~pos_mask

    result = np.empty_like(x, dtype=np.float64)

    # For x >= 0
    exp_neg = np.exp(-x[pos_mask])
    result[pos_mask] = 1.0 / (1.0 + exp_neg)

    # For x < 0
    exp_pos = np.exp(x[neg_mask])
    result[neg_mask] = exp_pos / (1.0 + exp_pos)

    return result


def sgns_loss(
    dot_positive: np.ndarray,
    dot_negatives: np.ndarray,
) -> float:
    """Skip-gram negative sampling (SGNS) loss for one training pair.

    The loss is the negative-log-likelihood of the binary classification
    objective::

        L = −log σ(v'_c · v_w) − Σ_{k} log σ(−v'_k · v_w)

    where ``v_w`` is the center (input) embedding, ``v'_c`` is the true
    context (output) embedding, and ``v'_k`` are the negative-sample
    output embeddings.

    We clip internal log arguments to ``[1e-10, 1]`` to avoid ``log(0)``.

    Args:
        dot_positive: Scalar dot product for the positive pair.
        dot_negatives: Array of dot products for negative samples, shape
            ``(K,)``.

    Returns:
        Scalar loss value.
    """
    eps = 1e-10

    # Positive term: −log σ(dot_positive)
    sig_pos = sigmoid(dot_positive)
    loss_pos = -np.log(np.clip(sig_pos, eps, 1.0))

    # Negative terms: −Σ log σ(−dot_negative_k)
    sig_neg = sigmoid(-dot_negatives)
    loss_neg = -np.sum(np.log(np.clip(sig_neg, eps, 1.0)))

    return float(loss_pos + loss_neg)
