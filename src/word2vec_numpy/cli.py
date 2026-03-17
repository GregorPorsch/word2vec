# src/word2vec_numpy/cli.py
"""Command-line interface for word2vec training and evaluation."""

from __future__ import annotations

import argparse
import sys

from word2vec_numpy.config import TrainConfig
from word2vec_numpy.evaluation import print_evaluation_report
from word2vec_numpy.model import SkipGramModel
from word2vec_numpy.preprocessing import load_corpus
from word2vec_numpy.sampling import NegativeSampler
from word2vec_numpy.trainer import Trainer
from word2vec_numpy.utils import save_embeddings, set_seed
from word2vec_numpy.vocabulary import Vocabulary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="word2vec",
        description="Train skip-gram word2vec with negative sampling (pure NumPy).",
    )

    parser.add_argument(
        "--corpus",
        type=str,
        default=None,
        help="Path to a plain-text corpus file (one sentence per line). "
        "Defaults to the bundled tiny Shakespeare excerpt.",
    )
    parser.add_argument("--dim", type=int, default=50, help="Embedding dimension (default: 50).")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs (default: 5).")
    parser.add_argument("--window", type=int, default=5, help="Context window size (default: 5).")
    parser.add_argument(
        "--negatives",
        type=int,
        default=5,
        help="Number of negative samples per pair (default: 5).",
    )
    parser.add_argument("--lr", type=float, default=0.025, help="Initial learning rate.")
    parser.add_argument(
        "--min-count",
        type=int,
        default=2,
        help="Minimum word count for vocabulary inclusion (default: 2).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/embeddings.txt",
        help="Output path for saved embeddings (default: outputs/embeddings.txt).",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Entry point: full train -> evaluate -> save pipeline."""
    args = parse_args(argv)

    # Configuration
    config = TrainConfig(
        embedding_dim=args.dim,
        window_size=args.window,
        num_negatives=args.negatives,
        learning_rate=args.lr,
        epochs=args.epochs,
        min_count=args.min_count,
        seed=args.seed,
    )

    rng = set_seed(config.seed)

    # Preprocessing
    print("Loading corpus ...")
    sentences = load_corpus(args.corpus)
    print(f"  {len(sentences)} sentences loaded.\n")

    # Vocabulary
    print("Building vocabulary ...")
    vocab = Vocabulary.build_from_corpus(sentences, min_count=config.min_count)
    vocab.compute_discard_probs(config.subsample_threshold)
    print(f"  {vocab}\n")

    if vocab.size < 2:
        print(
            "ERROR: vocabulary too small to train. Use a larger corpus or lower --min-count.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Model and sampler
    model = SkipGramModel(vocab.size, config.embedding_dim, rng)
    sampler = NegativeSampler(vocab)

    # Training
    print("Starting training ...\n")
    trainer = Trainer(model, config, vocab, sampler, rng)
    trainer.train(sentences)

    # Evaluation
    print_evaluation_report(model, vocab)

    # Save embeddings
    save_embeddings(model, vocab, args.output)
    print(f"\nEmbeddings saved to {args.output}")


if __name__ == "__main__":
    main()
