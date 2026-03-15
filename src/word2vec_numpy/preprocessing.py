"""Text preprocessing: cleaning, tokenisation, and corpus loading."""

from __future__ import annotations

import re
from pathlib import Path

# Bundled corpus lives at data/tiny_shakespeare.txt relative to project root.
_DEFAULT_CORPUS = Path(__file__).resolve().parents[2] / "data" / "tiny_shakespeare.txt"


def clean_text(text: str) -> str:
    """Lowercase, strip non-alphabetic characters, and normalise whitespace.

    We deliberately keep the preprocessing simple — for a small research
    corpus, aggressive cleaning is counterproductive. Keeping only ASCII
    letters and spaces yields a clean token stream without exotic edge cases.
    """
    text = text.lower()
    # Keep only letters and whitespace; collapse runs of whitespace.
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> list[str]:
    """Split cleaned text into tokens (whitespace-delimited)."""
    return text.split()


def load_corpus(path: str | Path | None = None) -> list[list[str]]:
    """Load a text file and return a list of tokenised sentences.

    Each non-empty line becomes one sentence (list of tokens).
    If *path* is ``None``, the bundled Shakespeare excerpt is used
    so that the project works out of the box.

    Args:
        path: Path to a plain-text file. One sentence per line is ideal,
              but the function also handles paragraph-style text.

    Returns:
        A list of sentences, each a list of lowercase tokens.
    """
    corpus_path = Path(path) if path is not None else _DEFAULT_CORPUS

    if not corpus_path.exists():
        raise FileNotFoundError(
            f"Corpus file not found: {corpus_path}. "
            "Pass a valid path or use the bundled tiny_shakespeare.txt."
        )

    raw = corpus_path.read_text(encoding="utf-8")
    sentences: list[list[str]] = []

    for line in raw.splitlines():
        tokens = tokenize(clean_text(line))
        if tokens:
            sentences.append(tokens)

    return sentences
