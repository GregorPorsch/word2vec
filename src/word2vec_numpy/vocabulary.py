# src/word2vec_numpy/vocabulary.py
"""Vocabulary construction and frequency-based subsampling."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable


class Vocabulary:
    """Word-to-index mapping with frequency statistics.

    The vocabulary is constructed from a tokenised corpus and optionally
    pruned by a minimum-count threshold. It also precomputes the discard
    probabilities used for Mikolov-style frequent-word subsampling.

    Attributes:
        word2idx: Mapping from word string to integer index.
        idx2word: List mapping integer index back to word string.
        counts: Raw token counts for every word in the vocabulary.
        total_count: Sum of all word counts.
        frequencies: Normalised unigram frequencies (count / total_count).
        discard_probs: Per-word probability of *discarding* a token during
            training (see :pymethod:`subsample_discard_prob`).
    """

    def __init__(
        self,
        word2idx: dict[str, int],
        idx2word: list[str],
        counts: list[int],
    ) -> None:
        self.word2idx = word2idx
        self.idx2word = idx2word
        self.counts = counts
        self.total_count: int = sum(counts)
        self.frequencies: list[float] = [c / self.total_count for c in counts]
        # Discard probabilities are set later via `compute_discard_probs`.
        self.discard_probs: list[float] = [0.0] * len(idx2word)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def build_from_corpus(
        cls,
        sentences: Iterable[list[str]],
        min_count: int = 1,
    ) -> Vocabulary:
        """Build vocabulary from tokenised sentences.

        Args:
            sentences: Iterable of token lists.
            min_count: Words occurring fewer than this many times are dropped.

        Returns:
            A :class:`Vocabulary` instance.
        """
        raw_counts: Counter[str] = Counter()
        for sentence in sentences:
            raw_counts.update(sentence)

        # Filter by min_count and sort for deterministic ordering.
        filtered = sorted(
            ((w, c) for w, c in raw_counts.items() if c >= min_count),
            key=lambda t: (-t[1], t[0]),  # most frequent first, then alpha
        )

        idx2word = [w for w, _ in filtered]
        word2idx = {w: i for i, w in enumerate(idx2word)}
        counts = [c for _, c in filtered]

        return cls(word2idx, idx2word, counts)

    # ------------------------------------------------------------------
    # Subsampling
    # ------------------------------------------------------------------

    def compute_discard_probs(self, threshold: float) -> None:
        """Precompute per-word discard probabilities.

        The probability of keeping a word w_i with frequency f(w_i) is:

            P_keep(w_i) = sqrt(t / f(w_i))  +  t / f(w_i)

        where t is the threshold. Equivalently, the probability of
        discarding the word is 1 - P_keep.

        High-frequency words (articles, prepositions) are discarded more
        often, speeding up training and improving the quality of the
        learned representations for less-common words.

        Args:
            threshold: Subsampling threshold.  Reasonable values are
                around 1e-3 to 1e-5. Set to 0 to disable.
        """
        if threshold <= 0.0:
            self.discard_probs = [0.0] * len(self.idx2word)
            return

        for i, freq in enumerate(self.frequencies):
            if freq == 0.0:
                self.discard_probs[i] = 0.0
            else:
                ratio = threshold / freq
                keep_prob = math.sqrt(ratio) + ratio
                # Clamp to [0, 1] — very rare words should always be kept.
                self.discard_probs[i] = max(0.0, 1.0 - keep_prob)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        """Number of words in the vocabulary."""
        return len(self.idx2word)

    def encode_sentence(self, tokens: list[str]) -> list[int]:
        """Convert a list of tokens to a list of vocabulary indices.

        Tokens not in the vocabulary are silently dropped.
        """
        return [self.word2idx[t] for t in tokens if t in self.word2idx]

    def __len__(self) -> int:
        return self.size

    def __contains__(self, word: str) -> bool:
        return word in self.word2idx

    def __repr__(self) -> str:
        return f"Vocabulary(size={self.size}, total_count={self.total_count})"
