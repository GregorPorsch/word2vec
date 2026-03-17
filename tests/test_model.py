# tests/test_model.py
"""Tests for the SkipGramModel: shapes, loss, and finite-difference gradient check."""

import numpy as np
import pytest

from word2vec_numpy.losses import sigmoid
from word2vec_numpy.model import SkipGramModel


class TestSigmoid:
    def test_zero(self):
        assert abs(sigmoid(np.array([0.0]))[0] - 0.5) < 1e-10

    def test_large_positive(self):
        result = sigmoid(np.array([100.0]))[0]
        assert abs(result - 1.0) < 1e-10

    def test_large_negative(self):
        result = sigmoid(np.array([-100.0]))[0]
        assert abs(result) < 1e-10

    def test_symmetry(self):
        """σ(x) + σ(-x) = 1."""
        x = np.array([0.5, -1.3, 2.7])
        assert np.allclose(sigmoid(x) + sigmoid(-x), 1.0)


class TestShapes:
    def test_embedding_shapes(self):
        rng = np.random.default_rng(0)
        model = SkipGramModel(vocab_size=100, embedding_dim=30, rng=rng)
        assert model.W_in.shape == (100, 30)
        assert model.W_out.shape == (100, 30)

    def test_compute_loss_returns_scalar(self):
        rng = np.random.default_rng(0)
        model = SkipGramModel(vocab_size=10, embedding_dim=5, rng=rng)
        neg_ids = np.array([2, 3, 4])
        loss = model.compute_loss(center_id=0, context_id=1, neg_ids=neg_ids)
        assert isinstance(loss, float)

    def test_train_step_returns_scalar(self):
        rng = np.random.default_rng(0)
        model = SkipGramModel(vocab_size=10, embedding_dim=5, rng=rng)
        neg_ids = np.array([2, 3])
        loss = model.train_step(center_id=0, context_id=1, neg_ids=neg_ids, lr=0.01)
        assert isinstance(loss, float)


class TestFiniteDifferenceGradientCheck:
    """Verify analytic gradients against finite-difference approximation.

    This is the most important correctness test in the project.
    We use a tiny model (V=4, D=3) and perturb each parameter by ε,
    computing (L(θ+ε) - L(θ−ε)) / (2ε) and comparing against the
    analytic gradient.  A relative error < 1e-5 for each parameter
    is expected.
    """

    @pytest.fixture()
    def setup(self):
        rng = np.random.default_rng(7)
        V, D = 4, 3
        model = SkipGramModel(V, D, rng)
        center_id = 0
        context_id = 1
        neg_ids = np.array([2, 3])
        return model, center_id, context_id, neg_ids

    @staticmethod
    def _loss_at(model, center_id, context_id, neg_ids):
        """Compute loss without updating parameters."""
        return model.compute_loss(center_id, context_id, neg_ids)

    def test_grad_W_in(self, setup):
        model, center_id, context_id, neg_ids = setup
        eps = 1e-5

        # Compute analytic gradient by doing a train step at lr=0
        # We extract the gradient indirectly: run train_step with lr=1
        # and the gradient is  -(W_after - W_before).
        W_in_before = model.W_in[center_id].copy()
        model.train_step(center_id, context_id, neg_ids, lr=1.0)
        analytic_grad = -(model.W_in[center_id] - W_in_before)
        # Restore
        model.W_in[center_id] = W_in_before
        # Also need to restore W_out that was modified
        # Re-init model fresh for numerical check
        rng = np.random.default_rng(7)
        model2 = SkipGramModel(4, 3, rng)

        numerical_grad = np.zeros_like(model2.W_in[center_id])
        for d in range(model2.embedding_dim):
            model2.W_in[center_id, d] += eps
            loss_plus = self._loss_at(model2, center_id, context_id, neg_ids)
            model2.W_in[center_id, d] -= 2 * eps
            loss_minus = self._loss_at(model2, center_id, context_id, neg_ids)
            model2.W_in[center_id, d] += eps  # restore
            numerical_grad[d] = (loss_plus - loss_minus) / (2 * eps)

        # Recompute analytic gradient fresh
        rng3 = np.random.default_rng(7)
        model3 = SkipGramModel(4, 3, rng3)
        W_in_before = model3.W_in[center_id].copy()
        model3.train_step(center_id, context_id, neg_ids, lr=1.0)
        analytic_grad = -(model3.W_in[center_id] - W_in_before)

        rel_error = np.abs(analytic_grad - numerical_grad) / (
            np.maximum(np.abs(analytic_grad), np.abs(numerical_grad)) + 1e-12
        )
        assert np.all(rel_error < 1e-4), (
            f"W_in grad check failed: max rel error = {rel_error.max()}"
        )

    def test_grad_W_out_context(self, setup):
        _, center_id, context_id, neg_ids = setup
        eps = 1e-5

        rng = np.random.default_rng(7)
        model_num = SkipGramModel(4, 3, rng)

        numerical_grad = np.zeros_like(model_num.W_out[context_id])
        for d in range(model_num.embedding_dim):
            model_num.W_out[context_id, d] += eps
            loss_plus = self._loss_at(model_num, center_id, context_id, neg_ids)
            model_num.W_out[context_id, d] -= 2 * eps
            loss_minus = self._loss_at(model_num, center_id, context_id, neg_ids)
            model_num.W_out[context_id, d] += eps
            numerical_grad[d] = (loss_plus - loss_minus) / (2 * eps)

        rng2 = np.random.default_rng(7)
        model_an = SkipGramModel(4, 3, rng2)
        W_out_before = model_an.W_out[context_id].copy()
        model_an.train_step(center_id, context_id, neg_ids, lr=1.0)
        analytic_grad = -(model_an.W_out[context_id] - W_out_before)

        rel_error = np.abs(analytic_grad - numerical_grad) / (
            np.maximum(np.abs(analytic_grad), np.abs(numerical_grad)) + 1e-12
        )
        assert np.all(rel_error < 1e-4), (
            f"W_out context grad check failed: max rel error = {rel_error.max()}"
        )

    def test_grad_W_out_negative(self, setup):
        _, center_id, context_id, neg_ids = setup
        eps = 1e-5

        for neg_idx_pos, neg_id in enumerate(neg_ids):
            rng = np.random.default_rng(7)
            model_num = SkipGramModel(4, 3, rng)

            numerical_grad = np.zeros(model_num.embedding_dim)
            for d in range(model_num.embedding_dim):
                model_num.W_out[neg_id, d] += eps
                loss_plus = self._loss_at(model_num, center_id, context_id, neg_ids)
                model_num.W_out[neg_id, d] -= 2 * eps
                loss_minus = self._loss_at(model_num, center_id, context_id, neg_ids)
                model_num.W_out[neg_id, d] += eps
                numerical_grad[d] = (loss_plus - loss_minus) / (2 * eps)

            rng2 = np.random.default_rng(7)
            model_an = SkipGramModel(4, 3, rng2)
            W_out_before = model_an.W_out[neg_id].copy()
            model_an.train_step(center_id, context_id, neg_ids, lr=1.0)
            analytic_grad = -(model_an.W_out[neg_id] - W_out_before)

            rel_error = np.abs(analytic_grad - numerical_grad) / (
                np.maximum(np.abs(analytic_grad), np.abs(numerical_grad)) + 1e-12
            )
            assert np.all(rel_error < 1e-4), (
                f"W_out neg[{neg_idx_pos}] grad check failed: max rel error = {rel_error.max()}"
            )
