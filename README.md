# word2vec

A from-scratch implementation of **skip-gram word2vec with negative sampling** in pure NumPy.
This implementation strips the algorithm down to its mathematical core: two embedding matrices, a binary classification loss, and stochastic gradient descent.

---


## Features

- **Skip-gram** architecture with **negative sampling** (SGNS)
- Pure NumPy - no PyTorch, TensorFlow, or JAX
- Explicit, hand-derived gradient computation
- Numerically stable sigmoid and loss
- Xavier embedding initialisation
- Dynamic context windows and Mikolov-style frequent-word subsampling
- Linear learning-rate decay
- Finite-difference gradient verification
- Nearest-neighbour and analogy evaluation
- Configurable hyperparameters with deterministic seeding
- CLI entry point for easy experimentation
- Bundled Shakespeare corpus for out-of-the-box training

---

## Repository Structure

```
word2vec/
├── pyproject.toml                  # Project config, dependencies, CLI entry point
├── Makefile                        # Common task automation (test, lint, train, …)
├── .devcontainer/
│   └── devcontainer.json           # VS Code / GitHub Codespaces dev container
├── data/
│   └── tiny_shakespeare.txt        # Bundled public-domain corpus (~5 KB)
├── docs/
│   ├── mathematical_core.md        # Full SGNS derivation with code cross-references
├── src/word2vec_numpy/
│   ├── __init__.py
│   ├── config.py                   # TrainConfig dataclass
│   ├── preprocessing.py            # Text cleaning, tokenisation, corpus loading
│   ├── vocabulary.py               # Vocab construction, subsampling probabilities
│   ├── dataset.py                  # Context-window pair generation
│   ├── sampling.py                 # Negative sampler (unigram^¾ distribution)
│   ├── losses.py                   # Numerically stable sigmoid and SGNS loss
│   ├── model.py                    # SkipGramModel: forward, gradients, SGD
│   ├── trainer.py                  # Training loop with LR decay and logging
│   ├── evaluation.py               # Nearest-neighbour and analogy queries
│   ├── utils.py                    # Seeding, cosine similarity, embedding I/O
│   └── cli.py                      # Command-line entry point
└── tests/
    ├── test_preprocessing.py       # Text cleaning and corpus loading
    ├── test_vocabulary.py          # Vocab construction, min_count, subsampling
    ├── test_sampling.py            # Negative sampler validity
    ├── test_model.py               # Shapes, sigmoid, gradient check
    └── test_training.py            # Loss decrease, end-to-end smoke test
```

---

## Setup

### Dev Container (VS Code / GitHub Codespaces)

The repository includes a [dev container](.devcontainer/devcontainer.json) configuration. Open the project in VS Code and select **"Reopen in Container"** (or launch a Codespace on GitHub) - dependencies are installed automatically via `uv sync` on container creation.

The container includes:
- Python 3.12
- `uv` (pre-installed via the Astral feature)
- VS Code extensions: Ruff, Python

### Local Setup

Requires Python ≥ 3.11 and [uv](https://docs.astral.sh/uv/).

```bash
# Clone and enter the repository
git clone <repo-url> && cd word2vec

# Install dependencies
make install

# Verify
make check
```

---

## Makefile Targets

All common tasks are automated via `make`:

| Target | Description |
|--------|-------------|
| `make help` | Show all available targets |
| `make install` | Install all dependencies (`uv sync`) |
| `make test` | Run the full test suite |
| `make lint` | Run ruff linter |
| `make format` | Auto-format code with ruff |
| `make check` | Run lint + tests (CI-style) |
| `make train` | Train on bundled corpus with default hyperparameters |
| `make clean` | Remove build artifacts, caches, and outputs |

---

## Usage

### Train on the bundled corpus

```bash
uv run word2vec
```

This trains with default hyperparameters (dim=50, window=5, K=5, 5 epochs) on the bundled Shakespeare excerpt, prints per-epoch loss, runs nearest-neighbour evaluation, and saves embeddings to `outputs/embeddings.txt`.

### Customise hyperparameters

```bash
uv run word2vec --dim 100 --epochs 10 --window 3 --negatives 10 --lr 0.05 --seed 123
```

### Train on your own corpus

```bash
uv run word2vec --corpus path/to/your/text.txt --dim 100 --epochs 20
```

The corpus should be a plain-text file (one sentence per line is ideal, but paragraph-style text also works).

### Full CLI options

```bash
uv run word2vec --help
```

---

## Expected Output

A typical training run on the bundled corpus looks like:

```
Loading corpus ...
  160 sentences loaded.

Building vocabulary ...
  Vocabulary(size=300, total_count=1100)

Starting training ...

  Epoch 1/5 — mean loss: 3.1842  (4200 pairs, 1.2s)
  Epoch 2/5 — mean loss: 2.5103  (4150 pairs, 1.1s)
  Epoch 3/5 — mean loss: 2.1297  (4180 pairs, 1.1s)
  Epoch 4/5 — mean loss: 1.8654  (4120 pairs, 1.1s)
  Epoch 5/5 — mean loss: 1.6983  (4160 pairs, 1.1s)

============================================================
NEAREST-NEIGHBOUR EVALUATION
============================================================
  king:
    crown               +0.4521
    noble               +0.4103
    ...

Embeddings saved to outputs/embeddings.txt
```

*(Exact numbers will vary with the random seed and corpus.)*

---

## Mathematics

The complete mathematical derivation — from the skip-gram objective through negative sampling loss to the explicit gradient formulas — is in **[`docs/mathematical_core.md`](docs/mathematical_core.md)**.

**Key equation.** For center word $w$, positive context word $c$, and negative samples $\{k_1, \ldots, k_K\}$:

$$\mathcal{L} = -\log \sigma(\mathbf{u}_c^\top \mathbf{v}_w) - \sum_{i=1}^{K} \log \sigma(-\mathbf{u}_{k_i}^\top \mathbf{v}_w)$$

The gradients are derived in full and cross-referenced to the code in `model.py`.

---

## Tests

```bash
make test
```

The test suite (37 tests) covers:

| Test file | What it verifies |
|-----------|-----------------|
| `test_preprocessing.py` | Text cleaning, tokenisation, corpus loading |
| `test_vocabulary.py` | Vocab counts, `min_count` filtering, subsampling bounds |
| `test_sampling.py` | Negative sampler validity, exclusion, distribution properties |
| `test_model.py` | Embedding shapes, sigmoid correctness, **finite-difference gradient check** |
| `test_training.py` | Loss decreases on toy corpus, end-to-end smoke test |

The **gradient check** is the most important correctness test — it verifies that the analytic gradients in `model.train_step()` match the numerical finite-difference approximation to within a relative error of $10^{-4}$.

---

## Reproducibility

- All random operations use a configurable seed (`--seed`, default 42)
- The bundled corpus is included in the repository
- Training hyperparameters are printed at the start of each run
- The project is fully deterministic given the same seed and corpus
