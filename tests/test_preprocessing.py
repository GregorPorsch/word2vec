# tests/test_preprocessing.py
"""Tests for text preprocessing utilities."""

from word2vec_numpy.preprocessing import clean_text, load_corpus, tokenize


class TestCleanText:
    def test_lowercases(self):
        assert clean_text("Hello World") == "hello world"

    def test_strips_punctuation(self):
        assert clean_text("it's a test!") == "it s a test"

    def test_collapses_whitespace(self):
        assert clean_text("  too   many   spaces  ") == "too many spaces"

    def test_removes_digits(self):
        assert clean_text("word2vec 3000") == "word vec"

    def test_empty_string(self):
        assert clean_text("") == ""

    def test_only_punctuation(self):
        assert clean_text("!!!...???") == ""


class TestTokenize:
    def test_simple_split(self):
        assert tokenize("the cat sat") == ["the", "cat", "sat"]

    def test_empty_string(self):
        assert tokenize("") == []

    def test_single_word(self):
        assert tokenize("hello") == ["hello"]


class TestLoadCorpus:
    def test_loads_bundled_corpus(self):
        """The bundled Shakespeare file should load without errors."""
        sentences = load_corpus()
        assert len(sentences) > 0
        # Every sentence should be a list of strings.
        for sent in sentences:
            assert isinstance(sent, list)
            for token in sent:
                assert isinstance(token, str)
                assert token == token.lower()

    def test_missing_file_raises(self):
        import pytest

        with pytest.raises(FileNotFoundError):
            load_corpus("/nonexistent/file.txt")
