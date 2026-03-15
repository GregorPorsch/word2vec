"""Tests for the negative sampler."""

import numpy as np

from word2vec_numpy.sampling import NegativeSampler
from word2vec_numpy.vocabulary import Vocabulary


def _make_vocab(words_and_counts: list[tuple[str, int]]) -> Vocabulary:
    """Helper: create a Vocabulary from explicit word-count pairs."""
    idx2word = [w for w, _ in words_and_counts]
    word2idx = {w: i for i, w in enumerate(idx2word)}
    counts = [c for _, c in words_and_counts]
    return Vocabulary(word2idx, idx2word, counts)


class TestNegativeSampler:
    def test_returns_correct_count(self):
        vocab = _make_vocab([("a", 10), ("b", 5), ("c", 1)])
        sampler = NegativeSampler(vocab)
        rng = np.random.default_rng(42)

        samples = sampler.sample(7, rng=rng)
        assert samples.shape == (7,)

    def test_indices_are_valid(self):
        vocab = _make_vocab([("a", 10), ("b", 5), ("c", 1)])
        sampler = NegativeSampler(vocab)
        rng = np.random.default_rng(0)

        for _ in range(20):
            samples = sampler.sample(5, rng=rng)
            assert np.all(samples >= 0)
            assert np.all(samples < vocab.size)

    def test_exclude_works(self):
        vocab = _make_vocab([("a", 10), ("b", 5), ("c", 1)])
        sampler = NegativeSampler(vocab)
        rng = np.random.default_rng(123)

        for _ in range(50):
            samples = sampler.sample(5, exclude=1, rng=rng)
            assert 1 not in samples

    def test_noise_distribution_sums_to_one(self):
        vocab = _make_vocab([("a", 100), ("b", 50), ("c", 10)])
        sampler = NegativeSampler(vocab)
        assert abs(sampler.noise_dist.sum() - 1.0) < 1e-10

    def test_noise_distribution_shape(self):
        vocab = _make_vocab([("a", 100), ("b", 50), ("c", 10)])
        sampler = NegativeSampler(vocab)
        assert sampler.noise_dist.shape == (3,)
