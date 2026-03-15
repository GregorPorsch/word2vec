"""Tests for the training loop: loss decreases and end-to-end smoke test."""

import numpy as np

from word2vec_numpy.config import TrainConfig
from word2vec_numpy.model import SkipGramModel
from word2vec_numpy.sampling import NegativeSampler
from word2vec_numpy.trainer import Trainer
from word2vec_numpy.vocabulary import Vocabulary


def _toy_setup():
    """Create a tiny corpus and all components needed for training."""
    sentences = [
        ["the", "cat", "sat", "on", "the", "mat"],
        ["the", "dog", "sat", "on", "the", "rug"],
        ["the", "cat", "chased", "the", "dog"],
    ]

    config = TrainConfig(
        embedding_dim=10,
        window_size=2,
        num_negatives=3,
        learning_rate=0.05,
        epochs=1,
        min_count=1,
        subsample_threshold=0.0,  # disable subsampling for determinism
        seed=42,
        batch_log_interval=100_000,  # suppress logging
    )

    rng = np.random.default_rng(config.seed)
    vocab = Vocabulary.build_from_corpus(sentences, min_count=config.min_count)
    vocab.compute_discard_probs(config.subsample_threshold)
    model = SkipGramModel(vocab.size, config.embedding_dim, rng)
    sampler = NegativeSampler(vocab)

    return sentences, config, rng, vocab, model, sampler


class TestLossDecreases:
    def test_loss_decreases_on_repeated_pair(self):
        """Repeatedly training on the same pair should drive loss down."""
        rng = np.random.default_rng(0)
        model = SkipGramModel(vocab_size=5, embedding_dim=8, rng=rng)
        neg_ids = np.array([2, 3, 4])

        losses = []
        for _ in range(100):
            loss = model.train_step(center_id=0, context_id=1, neg_ids=neg_ids, lr=0.1)
            losses.append(loss)

        # Loss at end should be lower than at start.
        assert losses[-1] < losses[0], (
            f"Loss did not decrease: start={losses[0]:.4f}, end={losses[-1]:.4f}"
        )

    def test_epoch_loss_decreases(self):
        """Mean loss should decrease over multiple epochs on a toy corpus."""
        sentences, config, rng, vocab, model, sampler = _toy_setup()

        # Run 5 epochs.
        config_multi = TrainConfig(
            embedding_dim=config.embedding_dim,
            window_size=config.window_size,
            num_negatives=config.num_negatives,
            learning_rate=0.05,
            epochs=5,
            min_count=config.min_count,
            subsample_threshold=0.0,
            seed=config.seed,
            batch_log_interval=100_000,
        )

        trainer = Trainer(model, config_multi, vocab, sampler, rng)
        history = trainer.train(sentences)

        epoch_losses = history["epoch_losses"]
        assert len(epoch_losses) == 5
        # Epoch 5 loss should be lower than epoch 1.
        assert epoch_losses[-1] < epoch_losses[0], (
            f"Epoch losses did not decrease: {epoch_losses}"
        )


class TestEndToEnd:
    def test_smoke_test(self):
        """Full pipeline should run without errors."""
        sentences, config, rng, vocab, model, sampler = _toy_setup()
        trainer = Trainer(model, config, vocab, sampler, rng)
        history = trainer.train(sentences)

        assert "epoch_losses" in history
        assert len(history["epoch_losses"]) == 1
        assert history["epoch_losses"][0] > 0.0
