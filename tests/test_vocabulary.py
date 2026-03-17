# tests/test_vocabulary.py
"""Tests for vocabulary construction and subsampling."""

from word2vec_numpy.vocabulary import Vocabulary


class TestVocabularyConstruction:
    def test_basic_build(self):
        sentences = [["the", "cat", "sat"], ["the", "cat", "slept"]]
        vocab = Vocabulary.build_from_corpus(sentences, min_count=1)

        assert "the" in vocab
        assert "cat" in vocab
        assert "sat" in vocab
        assert vocab.size == 4  # the, cat, sat, slept

    def test_min_count_filtering(self):
        sentences = [["the", "cat", "sat"], ["the", "cat", "slept"]]
        vocab = Vocabulary.build_from_corpus(sentences, min_count=2)

        assert "the" in vocab
        assert "cat" in vocab
        # "sat" and "slept" appear only once - should be filtered.
        assert "sat" not in vocab
        assert "slept" not in vocab
        assert vocab.size == 2

    def test_counts_are_correct(self):
        sentences = [["a", "b", "a"], ["a", "c"]]
        vocab = Vocabulary.build_from_corpus(sentences, min_count=1)

        idx_a = vocab.word2idx["a"]
        assert vocab.counts[idx_a] == 3
        assert vocab.counts[vocab.word2idx["b"]] == 1

    def test_deterministic_ordering(self):
        """Building the same corpus twice should yield identical mappings."""
        sentences = [["x", "y", "z", "x", "y"]]
        v1 = Vocabulary.build_from_corpus(sentences, min_count=1)
        v2 = Vocabulary.build_from_corpus(sentences, min_count=1)
        assert v1.idx2word == v2.idx2word

    def test_encode_sentence(self):
        sentences = [["a", "b", "c"]]
        vocab = Vocabulary.build_from_corpus(sentences, min_count=1)

        encoded = vocab.encode_sentence(["a", "c", "oov"])
        # "oov" is out of vocabulary - should be dropped.
        assert len(encoded) == 2
        assert encoded[0] == vocab.word2idx["a"]
        assert encoded[1] == vocab.word2idx["c"]


class TestSubsampling:
    def test_discard_probs_are_bounded(self):
        sentences = [["a"] * 100 + ["b"] * 10 + ["c"]]
        vocab = Vocabulary.build_from_corpus(sentences, min_count=1)
        vocab.compute_discard_probs(threshold=1e-2)

        for prob in vocab.discard_probs:
            assert 0.0 <= prob <= 1.0

    def test_frequent_words_discarded_more(self):
        sentences = [["a"] * 1000 + ["b"] * 10 + ["c"]]
        vocab = Vocabulary.build_from_corpus(sentences, min_count=1)
        vocab.compute_discard_probs(threshold=1e-3)

        # "a" is very frequent - should have higher discard prob than "c".
        prob_a = vocab.discard_probs[vocab.word2idx["a"]]
        prob_c = vocab.discard_probs[vocab.word2idx["c"]]
        assert prob_a >= prob_c

    def test_zero_threshold_disables(self):
        sentences = [["a"] * 100 + ["b"]]
        vocab = Vocabulary.build_from_corpus(sentences, min_count=1)
        vocab.compute_discard_probs(threshold=0.0)

        assert all(p == 0.0 for p in vocab.discard_probs)
