"""Evaluation utilities: nearest-neighbour queries and analogy tests."""

from __future__ import annotations

import numpy as np

from word2vec_numpy.model import SkipGramModel
from word2vec_numpy.vocabulary import Vocabulary


def _normalised_embeddings(model: SkipGramModel) -> np.ndarray:
    """Return L2-normalised embedding matrix, shape ``(V, D)``."""
    emb = model.get_all_embeddings()
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    # Avoid division by zero for any zero-norm rows.
    norms = np.where(norms == 0, 1.0, norms)
    return emb / norms


def most_similar(
    word: str,
    model: SkipGramModel,
    vocab: Vocabulary,
    top_k: int = 10,
) -> list[tuple[str, float]]:
    """Find the *top_k* nearest neighbours of *word* by cosine similarity.

    Args:
        word: Query word.
        model: Trained skip-gram model.
        vocab: Vocabulary.
        top_k: Number of neighbours to return.

    Returns:
        List of ``(neighbour_word, cosine_similarity)`` tuples sorted
        by descending similarity.  The query word itself is excluded.
    """
    if word not in vocab:
        return []

    normed = _normalised_embeddings(model)
    word_id = vocab.word2idx[word]
    query = normed[word_id]                          # (D,)
    similarities = normed @ query                    # (V,)

    # Exclude the query word itself.
    similarities[word_id] = -np.inf

    top_ids = np.argsort(similarities)[::-1][:top_k]
    return [(vocab.idx2word[i], float(similarities[i])) for i in top_ids]


def analogy(
    a: str,
    b: str,
    c: str,
    model: SkipGramModel,
    vocab: Vocabulary,
    top_k: int = 5,
) -> list[tuple[str, float]]:
    """Solve the analogy *a* is to *b* as *c* is to *?*.

    Uses the classic vector-arithmetic approach::

        d ≈ b − a + c

    Args:
        a, b, c: Words defining the analogy.
        model: Trained model.
        vocab: Vocabulary.
        top_k: Number of candidates to return.

    Returns:
        List of ``(word, cosine_similarity)`` tuples.
    """
    for w in (a, b, c):
        if w not in vocab:
            return []

    normed = _normalised_embeddings(model)
    vec = (
        normed[vocab.word2idx[b]]
        - normed[vocab.word2idx[a]]
        + normed[vocab.word2idx[c]]
    )
    # Normalise the query vector.
    vec_norm = np.linalg.norm(vec)
    if vec_norm > 0:
        vec /= vec_norm

    similarities = normed @ vec
    # Exclude input words.
    for w in (a, b, c):
        similarities[vocab.word2idx[w]] = -np.inf

    top_ids = np.argsort(similarities)[::-1][:top_k]
    return [(vocab.idx2word[i], float(similarities[i])) for i in top_ids]


def print_evaluation_report(
    model: SkipGramModel,
    vocab: Vocabulary,
    query_words: list[str] | None = None,
    top_k: int = 8,
) -> None:
    """Print a qualitative evaluation report to stdout.

    Args:
        model: Trained model.
        vocab: Vocabulary.
        query_words: Words to query.  If ``None``, picks the 8 most
            frequent words automatically.
        top_k: Neighbours per query.
    """
    if query_words is None:
        # Pick some of the most frequent words (skip the very top which
        # are likely stop-words — take indices 2..10 if available).
        start = min(2, vocab.size)
        end = min(start + 8, vocab.size)
        query_words = [vocab.idx2word[i] for i in range(start, end)]

    print("\n" + "=" * 60)
    print("NEAREST-NEIGHBOUR EVALUATION")
    print("=" * 60)

    for word in query_words:
        if word not in vocab:
            print(f"\n  '{word}' not in vocabulary — skipping.")
            continue
        neighbours = most_similar(word, model, vocab, top_k=top_k)
        print(f"\n  {word}:")
        for nb_word, sim in neighbours:
            print(f"    {nb_word:20s}  {sim:+.4f}")

    print("\n" + "=" * 60)
