"""Negative sampling with the unigram^(3/4) noise distribution."""

from __future__ import annotations

import numpy as np

from word2vec_numpy.vocabulary import Vocabulary


class NegativeSampler:
    """Draw negative word indices according to the smoothed unigram distribution.

    The noise distribution is::

        P_noise(w_i) ∝ count(w_i)^(3/4)

    This is the distribution proposed in Mikolov et al. (2013).  The 3/4
    exponent up-weights rare words relative to the raw unigram distribution,
    ensuring they appear as negative samples often enough for the model to
    learn meaningful distinctions.

    Attributes:
        noise_dist: Normalised noise distribution array of shape ``(V,)``.
    """

    def __init__(self, vocab: Vocabulary, power: float = 0.75) -> None:
        """Precompute the noise distribution.

        Args:
            vocab: Vocabulary with word counts.
            power: Exponent applied to raw counts (default 0.75).
        """
        counts = np.array(vocab.counts, dtype=np.float64)
        weighted = counts ** power
        self.noise_dist: np.ndarray = weighted / weighted.sum()
        self._vocab_size = vocab.size

    def sample(
        self,
        num_samples: int,
        exclude: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Draw negative sample indices.

        Args:
            num_samples: How many negative samples to draw.
            exclude: Index to exclude from the samples (typically the
                true context word).  If a drawn sample equals *exclude*,
                it is redrawn.
            rng: NumPy random generator.  Falls back to the default RNG
                if not provided.

        Returns:
            Integer array of shape ``(num_samples,)`` with sampled word
            indices.
        """
        rng = rng or np.random.default_rng()

        samples = rng.choice(
            self._vocab_size,
            size=num_samples,
            replace=True,
            p=self.noise_dist,
        )

        # Redraw any samples that collide with the excluded index.
        if exclude is not None:
            for i in range(num_samples):
                while samples[i] == exclude:
                    samples[i] = rng.choice(self._vocab_size, p=self.noise_dist)

        return samples
