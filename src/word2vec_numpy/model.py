# src/word2vec_numpy/model.py
"""Skip-gram model: embeddings, forward pass, gradient computation, and SGD update.

Mathematical notation (see docs/mathematical_core.md for full derivation)
=========================================================================

- V: vocabulary size
- D: embedding dimension
- W_in (V × D): input (center-word) embedding matrix
- W_out (V × D): output (context-word) embedding matrix
- v_w = W_in[w]: input embedding for center word w
- u_c = W_out[c]: output embedding for context word c
- u_k_i = W_out[k_i]: output embedding for negative sample k_i

Objective (negative sampling):
For a positive pair (w, c) (i.e. c is a real context word of w) and
K negative samples {k_1, ..., k_K}::

    L(w,c) = -log σ(u_c · v_w) - Σ_i log σ(-u_k_i · v_w)

Gradients:
Let σ_c = σ(u_c · v_w) and σ_k_i = σ(u_k_i · v_w).

∂L/∂v_w  = -(1 - σ_c) u_c  + Σ_i σ_k_i u_k_i
∂L/∂u_c  = -(1 - σ_c) v_w
∂L/∂u_k_i  =  σ_k_i v_w (for each negative sample k_i)

SGD update:
θ ← θ - η · ∂L/∂θ
"""

from __future__ import annotations

import numpy as np

from word2vec_numpy.losses import sgns_loss, sigmoid


class SkipGramModel:
    """Pure-NumPy skip-gram model with negative sampling.

    Attributes:
        W_in:  Input embedding matrix, shape (V, D).
        W_out: Output embedding matrix, shape (V, D).
        vocab_size: Size of the vocabulary V.
        embedding_dim: Dimensionality of embeddings D.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        rng: np.random.Generator,
    ) -> None:
        """Initialise embedding matrices with Xavier uniform initialisation.

        Xavier init keeps initial forward-pass magnitudes in a stable range,
        which helps gradient-based learning converge smoothly.

        Args:
            vocab_size: Number of words in the vocabulary V.
            embedding_dim: Embedding dimensionality D.
            rng: NumPy random generator for reproducibility.
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        # Xavier uniform: U(−√(6/(fan_in+fan_out)), +√(6/(fan_in+fan_out)))
        limit = np.sqrt(6.0 / (vocab_size + embedding_dim))
        self.W_in: np.ndarray = rng.uniform(-limit, limit, (vocab_size, embedding_dim))
        self.W_out: np.ndarray = rng.uniform(-limit, limit, (vocab_size, embedding_dim))

    # ------------------------------------------------------------------
    # Forward pass & loss
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        center_id: int,
        context_id: int,
        neg_ids: np.ndarray,
    ) -> float:
        """Compute the SGNS loss for one (center, context) pair.
        (Used for validation; training should call train_step() instead to avoid
        redundant embedding look-ups.)

        Args:
            center_id: Index of the center word w.
            context_id: Index of the true context word c.
            neg_ids: Integer array of negative-sample indices, shape (K,).

        Returns:
            Scalar loss value.
        """
        # Embedding look-up
        v_w = self.W_in[center_id]  # (D,)
        u_c = self.W_out[context_id]  # (D,)
        u_neg = self.W_out[neg_ids]  # (K, D)

        # Forward pass
        dot_pos = np.dot(u_c, v_w)  # scalar
        dot_neg = u_neg @ v_w  # (K,)

        return sgns_loss(dot_pos, dot_neg)

    # ------------------------------------------------------------------
    # Training step: forward + backward + SGD in one fused operation
    # ------------------------------------------------------------------

    def train_step(
        self,
        center_id: int,
        context_id: int,
        neg_ids: np.ndarray,
        lr: float,
    ) -> float:
        """Perform one SGD update for a single (center, context) pair.

        This method computes the forward pass, loss, exact gradients, and
        applies the SGD parameter update.

        Args:
            center_id: Index of center word w.
            context_id: Index of context word c.
            neg_ids: Negative-sample indices, shape (K,).
            lr: Current learning rate η.

        Returns:
            The loss before the parameter update (for logging).
        """
        # Embedding look-up
        v_w = self.W_in[center_id].copy()  # (D,)
        u_c = self.W_out[context_id].copy()  # (D,)
        u_neg = self.W_out[neg_ids].copy()  # (K, D)

        # Forward pass
        dot_pos = np.dot(u_c, v_w)  # scalar
        dot_neg = u_neg @ v_w  # (K,)

        loss = sgns_loss(dot_pos, dot_neg)

        # Gradient computation
        # Positive pair gradient:
        #   σ_c = σ(u_c · v_w) (sigma_pos)
        #   ∂L/∂u_c = −(1 − σ_c) v_w
        #   contribution to ∂L/∂v_w: −(1 − σ_c) u_c
        sigma_pos = sigmoid(dot_pos)  # scalar
        grad_u_c = -(1.0 - sigma_pos) * v_w  # (D,)
        grad_v_w = -(1.0 - sigma_pos) * u_c  # (D,)

        # Negative samples gradient:
        #   σ_k_i = σ(u_k_i · v_w) (sigma_neg)
        #   ∂L/∂u_k = σ_k v_w
        #   contribution to ∂L/∂v_w: Σ_i σ_k_i u_k_i
        sigma_neg = sigmoid(dot_neg)  # (K,)
        grad_u_neg = sigma_neg[:, np.newaxis] * v_w[np.newaxis, :]  # (K, D)
        grad_v_w += (sigma_neg[:, np.newaxis] * u_neg).sum(axis=0)  # (D,)

        # SGD update: θ ← θ − η · ∂L/∂θ
        self.W_in[center_id] -= lr * grad_v_w
        self.W_out[context_id] -= lr * grad_u_c
        self.W_out[neg_ids] -= lr * grad_u_neg

        return loss

    # ------------------------------------------------------------------
    # Embedding access
    # ------------------------------------------------------------------

    def get_embedding(self, word_id: int) -> np.ndarray:
        """Return the input embedding for a word (used for downstream tasks)."""
        return self.W_in[word_id].copy()

    def get_all_embeddings(self) -> np.ndarray:
        """Return a copy of the full input embedding matrix (V, D)."""
        return self.W_in.copy()
