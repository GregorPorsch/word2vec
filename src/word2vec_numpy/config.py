# src/word2vec_numpy/config.py
"""Hyperparameter configuration for word2vec training."""

from dataclasses import dataclass


@dataclass
class TrainConfig:
    """All hyperparameters for skip-gram with negative sampling.

    Attributes:
        embedding_dim: Dimensionality of word embeddings.
        window_size: Maximum distance between center and context word.
            The actual window is sampled uniformly from [1, window_size]
            for each center word, following the original word2vec paper.
        num_negatives: Number of negative samples per positive pair.
        learning_rate: Initial learning rate for SGD.
        min_learning_rate: Floor for learning rate during linear decay.
        epochs: Number of full passes over the training corpus.
        min_count: Minimum word frequency to be included in vocabulary.
        subsample_threshold: Threshold for frequent-word subsampling.
            Set to 0.0 to disable.
        seed: Random seed for reproducibility.
        batch_log_interval: Print loss every N training pairs.
    """

    embedding_dim: int = 50
    window_size: int = 5
    num_negatives: int = 5
    learning_rate: float = 0.025
    min_learning_rate: float = 1e-4
    epochs: int = 5
    min_count: int = 2
    subsample_threshold: float = 1e-3
    seed: int = 42
    batch_log_interval: int = 10_000
