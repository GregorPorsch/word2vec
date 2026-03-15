"""Utility helpers: seeding, similarity, I/O."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from word2vec_numpy.model import SkipGramModel
from word2vec_numpy.vocabulary import Vocabulary


def set_seed(seed: int) -> np.random.Generator:
    """Create a seeded NumPy random generator and set the legacy seed.

    We also set ``np.random.seed`` for any library code that might use
    the legacy API, but all internal code should use the returned
    ``Generator`` directly.

    Args:
        seed: Integer seed.

    Returns:
        A ``numpy.random.Generator`` instance.
    """
    np.random.seed(seed)
    return np.random.default_rng(seed)


def cosine_similarity(u: np.ndarray, v: np.ndarray) -> float:
    """Compute cosine similarity between two vectors.

    Returns 0.0 if either vector has zero norm (avoids division by zero).
    """
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    if norm_u == 0.0 or norm_v == 0.0:
        return 0.0
    return float(np.dot(u, v) / (norm_u * norm_v))


def save_embeddings(
    model: SkipGramModel,
    vocab: Vocabulary,
    path: str | Path,
) -> None:
    """Save embeddings in a simple text format (word2vec `.txt` style).

    Format: first line is ``V D``, then one line per word:
    ``word val_1 val_2 ... val_D``.

    Args:
        model: Trained model.
        vocab: Vocabulary used during training.
        path: Output file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    embeddings = model.get_all_embeddings()  # (V, D)

    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{vocab.size} {model.embedding_dim}\n")
        for idx in range(vocab.size):
            word = vocab.idx2word[idx]
            vec_str = " ".join(f"{v:.6f}" for v in embeddings[idx])
            f.write(f"{word} {vec_str}\n")


def load_embeddings(path: str | Path) -> tuple[dict[str, np.ndarray], int]:
    """Load embeddings from a text file.

    Args:
        path: Path to the embeddings file.

    Returns:
        A tuple of ``(word_to_vector_dict, embedding_dim)``.
    """
    word_vectors: dict[str, np.ndarray] = {}
    embedding_dim = 0

    with open(path, encoding="utf-8") as f:
        header = f.readline().strip().split()
        _vocab_size, embedding_dim = int(header[0]), int(header[1])

        for line in f:
            parts = line.strip().split()
            word = parts[0]
            vec = np.array([float(x) for x in parts[1:]], dtype=np.float64)
            word_vectors[word] = vec

    return word_vectors, embedding_dim
