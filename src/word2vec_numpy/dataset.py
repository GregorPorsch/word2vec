# src/word2vec_numpy/dataset.py
"""Context-window pair generation for skip-gram training."""

from __future__ import annotations

import numpy as np

from word2vec_numpy.vocabulary import Vocabulary


def generate_training_pairs(
    sentences: list[list[str]],
    vocab: Vocabulary,
    window_size: int,
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    """Generate (center, context) training pairs with dynamic windows.

    For each center word, the actual window is drawn uniformly from
    [1, window_size], following the original word2vec implementation.
    This effectively down-weights more distant context words.

    Subsampling is applied: each token is stochastically discarded
    with probability vocab.discard_probs[idx] before pair generation.

    Args:
        sentences: Tokenised corpus (list of token lists).
        vocab: Built vocabulary with discard probabilities set.
        window_size: Maximum one-sided window size.
        rng: NumPy random generator for reproducibility.

    Returns:
        A list of (center_word_id, context_word_id) tuples.
    """
    pairs: list[tuple[int, int]] = []

    for sentence in sentences:
        # Map tokens to indices, dropping OOV words.
        word_ids = vocab.encode_sentence(sentence)

        # Subsampling: randomly discard frequent words
        word_ids = [idx for idx in word_ids if rng.random() >= vocab.discard_probs[idx]]

        if len(word_ids) < 2:
            continue

        # Generate pairs with a dynamic (reduced) window
        for i, center_id in enumerate(word_ids):
            # Sample a reduced window size in [1, window_size].
            actual_window = rng.integers(1, window_size + 1)

            start = max(0, i - actual_window)
            end = min(len(word_ids), i + actual_window + 1)

            for j in range(start, end):
                if j == i:
                    continue
                pairs.append((center_id, word_ids[j]))

    return pairs
