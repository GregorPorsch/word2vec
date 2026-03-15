"""Training loop for skip-gram with negative sampling."""

from __future__ import annotations

import sys
import time

import numpy as np

from word2vec_numpy.config import TrainConfig
from word2vec_numpy.dataset import generate_training_pairs
from word2vec_numpy.model import SkipGramModel
from word2vec_numpy.sampling import NegativeSampler
from word2vec_numpy.vocabulary import Vocabulary


class Trainer:
    """Orchestrates the word2vec training loop.

    Responsibilities:
    - Pair generation (per epoch, to get fresh subsampling / dynamic windows)
    - Learning-rate linear decay
    - Loss accumulation and periodic logging
    - Progress reporting

    Attributes:
        model: The :class:`SkipGramModel` being trained.
        config: Training hyperparameters.
        vocab: Vocabulary instance.
        sampler: Negative sampler.
        history: Dict with ``"epoch_losses"`` list (mean loss per epoch).
    """

    def __init__(
        self,
        model: SkipGramModel,
        config: TrainConfig,
        vocab: Vocabulary,
        sampler: NegativeSampler,
        rng: np.random.Generator,
    ) -> None:
        self.model = model
        self.config = config
        self.vocab = vocab
        self.sampler = sampler
        self.rng = rng
        self.history: dict[str, list[float]] = {"epoch_losses": []}

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self, sentences: list[list[str]]) -> dict[str, list[float]]:
        """Run the full training loop.

        For each epoch:
        1. Re-generate training pairs (fresh subsampling & window draws).
        2. Shuffle pairs.
        3. Iterate over pairs with SGD updates.
        4. Linearly decay learning rate from ``config.learning_rate``
           towards ``config.min_learning_rate``.

        Args:
            sentences: Tokenised corpus (list of token lists).

        Returns:
            Training history dict with per-epoch mean losses.
        """
        cfg = self.config

        # Total number of training steps (estimated) for LR scheduling.
        # We estimate by generating pairs once, then assume similar count
        # each epoch.
        sample_pairs = generate_training_pairs(
            sentences, self.vocab, cfg.window_size, self.rng,
        )
        estimated_total_steps = len(sample_pairs) * cfg.epochs
        global_step = 0

        print(f"Vocabulary size : {self.vocab.size}")
        print(f"Embedding dim   : {cfg.embedding_dim}")
        print(f"Window size     : {cfg.window_size}")
        print(f"Negative samples: {cfg.num_negatives}")
        print(f"Learning rate   : {cfg.learning_rate}")
        print(f"Epochs          : {cfg.epochs}")
        print(f"Est. pairs/epoch: ~{len(sample_pairs)}")
        print("-" * 50)

        for epoch in range(1, cfg.epochs + 1):
            epoch_start = time.time()

            # Re-generate pairs each epoch for variety (subsampling is
            # stochastic, and window sizes are re-drawn).
            pairs = generate_training_pairs(
                sentences, self.vocab, cfg.window_size, self.rng,
            )
            # Shuffle training pairs.
            indices = self.rng.permutation(len(pairs))

            epoch_loss = 0.0
            running_loss = 0.0
            log_count = 0

            for step_in_epoch, idx in enumerate(indices, start=1):
                center_id, context_id = pairs[idx]

                # Linear learning-rate decay.
                progress = global_step / max(estimated_total_steps, 1)
                lr = cfg.learning_rate - (cfg.learning_rate - cfg.min_learning_rate) * progress
                lr = max(lr, cfg.min_learning_rate)

                # Draw negative samples (excluding the true context word).
                neg_ids = self.sampler.sample(
                    cfg.num_negatives, exclude=context_id, rng=self.rng,
                )

                # Forward + backward + update.
                loss = self.model.train_step(center_id, context_id, neg_ids, lr)
                epoch_loss += loss
                running_loss += loss
                log_count += 1
                global_step += 1

                # Periodic logging.
                if step_in_epoch % cfg.batch_log_interval == 0:
                    avg = running_loss / log_count
                    pct = 100 * step_in_epoch / len(pairs)
                    sys.stdout.write(
                        f"\r  Epoch {epoch}/{cfg.epochs} "
                        f"[{pct:5.1f}%]  loss={avg:.4f}  lr={lr:.6f}"
                    )
                    sys.stdout.flush()
                    running_loss = 0.0
                    log_count = 0

            mean_loss = epoch_loss / len(pairs) if pairs else 0.0
            elapsed = time.time() - epoch_start
            self.history["epoch_losses"].append(mean_loss)

            print(
                f"\r  Epoch {epoch}/{cfg.epochs} "
                f"— mean loss: {mean_loss:.4f}  "
                f"({len(pairs)} pairs, {elapsed:.1f}s)"
            )

        return self.history
