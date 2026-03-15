# Experimental Notes

Observations from training on the bundled Shakespeare corpus (~160 lines, ~5 KB). This section documents empirical behaviour and connects implementation choices to broader ML context.

---

## Effect of Hyperparameters

### Embedding Dimension

| Dim | Convergence | Quality |
|-----|------------|---------|
| 10 | Fast (~2s for 10 epochs) | Nearest neighbours capture basic co-occurrence but lack nuance |
| 50 | Moderate (~10s) | Reasonable semantic groupings emerge (e.g., "king" near "noble", "crown") |
| 200 | Slow (~30s) | Overfits on this tiny corpus; representations do not improve |

**Takeaway:** For small corpora, lower dimensions (30–50) perform best. The model does not have enough data to fill a high-dimensional embedding space, and overfitting becomes the bottleneck.

### Number of Negative Samples

| K | Effect |
|---|--------|
| 2 | Under-regularised; the model doesn't learn to reject enough noise words |
| 5 | Good trade-off for small vocab |
| 15 | Marginal improvement at higher compute cost; diminishing returns with small V |

Mikolov et al. (2013) recommend K = 5–20 for small datasets and K = 2–5 for large ones. With our ~300-word vocabulary, K = 5 is appropriate.

### Context Window Size

| Window | Effect |
|--------|--------|
| 2 | Captures syntactic patterns (POS-tag-like neighbours) |
| 5 | Captures broader topical similarity |
| 10 | Noisy on small corpus; too many spurious co-occurrences |

The dynamic window (sampling from [1, window]) effectively up-weights nearby words, which is a simple but effective way to control the syntactic-vs-semantic trade-off.

---

## Training Loss Behaviour

On the tiny Shakespeare corpus (seed 42, dim 50, K = 5, window 5, lr 0.025):

- Epoch 1: loss ≈ 3.2
- Epoch 5: loss ≈ 1.8
- Epoch 10: loss ≈ 1.4

Loss decreases monotonically as expected. The per-epoch variance is noticeable because each epoch re-samples the subsampling and dynamic windows, so the training pairs differ slightly.

---

## What This Implementation Captures — and What It Does Not

### What it captures

- **Distributional semantics:** Words appearing in similar contexts get similar embeddings ("you shall know a word by the company it keeps" — Firth, 1957).
- **Linear substructure:** The embedding space exhibits some additive structure (analogies work on larger corpora).
- **Efficient learning signal:** Negative sampling provides a per-word-pair gradient without computing a full softmax.

### What it does not capture

| Limitation | Modern solution |
|-----------|-----------------|
| Fixed context window | Transformers attend to entire sequences |
| No word order sensitivity | Positional encodings in Transformers |
| Static embeddings (one vector per word) | Contextualised embeddings (BERT, GPT) |
| No subword information | BPE/WordPiece tokenisation |
| Local co-occurrence only | Masked/causal LM objectives over full documents |

---

## Connection to Modern Language Models

Word2vec, despite its simplicity, established core ideas that underpin modern NLP:

1. **Representation learning.** The insight that useful word representations can be learned from raw text via a self-supervised objective was foundational. Modern Transformer models (BERT, GPT) learn contextualised representations, but the core idea — learning from distributional context — is the same.

2. **Simple objectives can learn rich structure.** The SGNS objective is a binary classification task, yet it produces embeddings with remarkable algebraic properties (analogies, clustering). This is conceptually related to how masked language modelling (BERT) or next-token prediction (GPT) — both simple objectives — give rise to emergent capabilities.

3. **Optimisation and scaling.** Word2vec was one of the first models to demonstrate that scaling a simple model to large corpora produces qualitatively different representations than training the same model on small data. This scaling insight anticipated the modern finding that Transformer LMs exhibit phase transitions and emergent abilities with scale.

4. **Building from scratch matters.** Implementing SGNS end-to-end — deriving gradients by hand, choosing numerical stability strategies, managing the noise distribution — builds the same intuitions needed to understand and debug modern training pipelines. The concepts transfer directly: gradient flow, loss landscape, representation geometry, and the interplay between objective function and learned structure.

---

## Speed and Simplicity Trade-offs

This implementation prioritises clarity over speed:

- **Pure Python loop over training pairs:** A production word2vec (e.g., gensim) uses Cython, multi-threading, and BLAS routines. Our loop is ~100× slower but fully inspectable.
- **No batching:** We process one (center, context) pair at a time. Mini-batch SGD with matrix operations would be faster but obscures the per-pair gradient logic.
- **No Cython/Numba:** The entire codebase is pure Python + NumPy, so every computation is visible and debuggable.

For a corpus of ~5 KB, the training completes in seconds, making this a non-issue for the intended use case (education and correctness verification).
