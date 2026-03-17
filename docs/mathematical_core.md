# Mathematical Core: Skip-Gram with Negative Sampling

This document provides a complete mathematical derivation of the skip-gram model with negative sampling (SGNS) as implemented in this repository. Every equation is connected to its implementation counterpart in `src/word2vec_numpy/`.

---

## 1. Notation

| Symbol | Meaning | Dimension | Code reference |
|--------|---------|-----------|----------------|
| $V$ | Vocabulary size | scalar | `vocab.size` |
| $D$ | Embedding dimension | scalar | `config.embedding_dim` |
| $\mathbf{W}_{\text{in}}$ | Input (center-word) embedding matrix | $V \times D$ | `model.W_in` |
| $\mathbf{W}_{\text{out}}$ | Output (context-word) embedding matrix | $V \times D$ | `model.W_out` |
| $\mathbf{v}_w$ | Input embedding of center word $w$ | $D$ | `model.W_in[center_id]` |
| $\mathbf{u}_c$ | Output embedding of context word $c$ | $D$ | `model.W_out[context_id]` |
| $\mathbf{u}_k$ | Output embedding of negative sample $k$ | $D$ | `model.W_out[neg_ids[k]]` |
| $K$ | Number of negative samples | scalar | `config.num_negatives` |
| $T$ | Total number of words in the corpus | scalar | `sum(len(s) for s in sentences)` |
| $\theta$ | All trainable parameters, i.e. $\theta = (\mathbf{W}_{\mathrm{in}}, \mathbf{W}_{\mathrm{out}})$ | — | `model.W_in`, `model.W_out` |
| $\eta$ | Learning rate | scalar | `lr` in `train_step` |
| $\sigma(\cdot)$ | Sigmoid function | — | `losses.sigmoid()` |

---

## 2. The Skip-Gram Objective

### 2.1. Core Idea

Given a corpus of words $w_1, w_2, \ldots, w_T$, skip-gram maximises the probability of observing actual context words given a center word. The model parameters 
$$
\theta = (\mathbf{W}_{in}, \mathbf{W}_{out})
$$ 
comprise both embedding matrices - these are the only trainable parameters. For each center word $w_t$ and context word $w_c$ within a window of size $m$:

$$
\max_{\theta} \sum_{t=1}^{T} \sum_{\substack{c \in \text{window}(t) \\ c \neq t}} \log P(w_c \mid w_t; \theta)
$$

In the basic (softmax) formulation:

$$
P(w_c \mid w_t) = \frac{\exp(\mathbf{u}_c^\top \mathbf{v}_{w_t})}{\sum_{j=1}^{V} \exp(\mathbf{u}_j^\top \mathbf{v}_{w_t})}
$$

This requires a sum over the entire vocabulary in the denominator - computationally prohibitive for large $V$.

### 2.2. Why Negative Sampling

Negative sampling replaces the expensive softmax with a binary classification objective. Instead of computing the partition function over $V$ words, we:

1. Treat the true (center, context) pair as a **positive** example.
2. Draw $K$ **negative** examples from a noise distribution.
3. Train a binary classifier to distinguish positive from negative pairs.

This reduces each update from $O(V)$ to $O(K)$, where typically $K \in [5, 20]$.

---

## 3. Negative Sampling Loss

### 3.1. Binary Classification Formulation

Instead of predicting context words via the full softmax (section 2.1), negative sampling recasts the problem as binary classification: given a word pair, is it a genuine (center, context) pair drawn from the corpus, or a "noise" pair?

We model the probability that a pair $(w, j)$ is a real context pair as:

$$
P(D = 1 \mid w, j) = \sigma(\mathbf{u}_j^\top \mathbf{v}_w)
$$

where $\sigma(x) = \frac{1}{1 + e^{-x}}$ is the sigmoid function. Conversely, the probability that the pair is noise is $P(D = 0 \mid w, j) = 1 - \sigma(\mathbf{u}_j^\top \mathbf{v}_w) = \sigma(-\mathbf{u}_j^\top \mathbf{v}_w)$.

For a positive pair $(w, c)$ with label $D = 1$ and $K$ negative samples $\{k_1, \ldots, k_K\}$ each with label $D = 0$, the joint log-likelihood is:

$$
\ell = \log P(D=1 \mid w, c) + \sum_{i=1}^{K} \log P(D=0 \mid w, k_i)
$$

$$
= \log \sigma(\mathbf{u}_c^\top \mathbf{v}_w) + \sum_{i=1}^{K} \log \sigma(-\mathbf{u}_{k_i}^\top \mathbf{v}_w)
$$

The loss is the **negative log-likelihood** (since we minimise):

$$
\mathcal{L}(w, c) = -\ell = -\log \sigma(\mathbf{u}_c^\top \mathbf{v}_w) - \sum_{i=1}^{K} \log \sigma(-\mathbf{u}_{k_i}^\top \mathbf{v}_w)
$$

**Interpretation:**
- The first term encourages $\mathbf{u}_c^\top \mathbf{v}_w$ to be **large** (positive pair should have high dot product).
- The second term encourages $\mathbf{u}_{k_i}^\top \mathbf{v}_w$ to be **small** (negative pairs should have low dot product).

Minimising this loss maximises the model's ability to distinguish real context words from noise samples, which forces the embeddings to encode distributional similarity.

**Code:** `losses.sgns_loss(dot_positive, dot_negatives)` in `losses.py`.

### 3.2. Noise Distribution

Negative samples are drawn from the smoothed unigram distribution:

$$
P_n(w_i) \propto \text{count}(w_i)^{3/4}
$$

The $3/4$ exponent up-weights rare words relative to the raw unigram distribution. This is important because rare words would otherwise almost never appear as negative samples, preventing the model from learning to distinguish them.

**Code:** `NegativeSampler.__init__()` in `sampling.py`.

### 3.3. Sigmoid: Numerical Stability

The sigmoid function is implemented in a numerically stable way:

$$
\sigma(x) = \begin{cases}
\frac{1}{1 + e^{-x}} & \text{if } x \geq 0 \\
\frac{e^x}{1 + e^x} & \text{if } x < 0
\end{cases}
$$

This avoids computing $e^{|x|}$ for large $|x|$, which would overflow. Instead only exponentials of small magnitude (e.g. $e^{-1000}$) are evaluated. The two forms of the sigmoid function are algebraically identical but numerically distinct (i.e., they behave differently under finite-precision floating-point arithmetic).

**Code:** `losses.sigmoid()` in `losses.py`.

---

## 4. Gradient Derivation

We now derive the gradients of $\mathcal{L}$ with respect to each parameter. These are the exact gradients implemented in `model.train_step()`.

### 4.1. Useful Sigmoid Identities

Before proceeding, recall:

$$
\frac{d}{dx} \sigma(x) = \sigma(x)(1 - \sigma(x))
$$

$$
\frac{d}{dx} \log \sigma(x) = 1 - \sigma(x)
$$

$$
\frac{d}{dx} \log \sigma(-x) = -\sigma(x)
$$

### 4.2. Gradient w.r.t. the Center Embedding $\mathbf{v}_w$

$$
\mathcal{L} = -\log \sigma(\mathbf{u}_c^\top \mathbf{v}_w) - \sum_{i=1}^{K} \log \sigma(-\mathbf{u}_{k_i}^\top \mathbf{v}_w)
$$

**Positive term:**

$$
\frac{\partial}{\partial \mathbf{v}_w} \left[-\log \sigma(\mathbf{u}_c^\top \mathbf{v}_w)\right] = -(1 - \sigma(\mathbf{u}_c^\top \mathbf{v}_w)) \cdot \mathbf{u}_c
$$

*Derivation:* Let $z = \mathbf{u}_c^\top \mathbf{v}_w$. By the chain rule, $\frac{\partial}{\partial \mathbf{v}_w}[-\log \sigma(z)] = \frac{d}{dz}[-\log \sigma(z)] \cdot \frac{\partial z}{\partial \mathbf{v}_w}$. We have
$
\frac{\partial z}{\partial \mathbf{v}_w} = \mathbf{u}_c
$
and
$
\frac{d}{dz}\left[-\log \sigma(z)\right] = -(1 - \sigma(z)).
$

**Negative terms:**

$$
\frac{\partial}{\partial \mathbf{v}_w} \left[-\log \sigma(-\mathbf{u}_{k_i}^\top \mathbf{v}_w)\right] = \sigma(\mathbf{u}_{k_i}^\top \mathbf{v}_w) \cdot \mathbf{u}_{k_i}
$$

*Derivation:* Let $z_i = \mathbf{u}_{k_i}^\top \mathbf{v}_w$. By the chain rule, $\frac{\partial}{\partial \mathbf{v}_w}[-\log \sigma(-z_i)] = \frac{d}{dz_i}[-\log \sigma(-z_i)] \cdot \frac{\partial z_i}{\partial \mathbf{v}_w}$. We have
$
\frac{\partial z_i}{\partial \mathbf{v}_w} = \mathbf{u}_{k_i}
$
and
$
\frac{d}{dz_i}\left[-\log \sigma(-z_i)\right] = \sigma(z_i).
$

**Combined:**

$$
\boxed{\frac{\partial \mathcal{L}}{\partial \mathbf{v}_w} = -(1 - \sigma_c) \, \mathbf{u}_c + \sum_{i=1}^{K} \sigma_{k_i} \, \mathbf{u}_{k_i}}
$$

where
$$
\sigma_c = \sigma(\mathbf{u}_c^\top \mathbf{v}_w)
$$
and
$$
\sigma_{k_i} = \sigma(\mathbf{u}_{k_i}^\top \mathbf{v}_w).
$$

**Code:**
```python
grad_v_w = -(1.0 - sigma_pos) * u_c                        # (D,)
grad_v_w += (sigma_neg[:, np.newaxis] * u_neg).sum(axis=0) # (D,)
```

### 4.3. Gradient w.r.t. the Context Embedding $\mathbf{u}_c$

Only the positive term involves $\mathbf{u}_c$:

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{u}_c} = \frac{\partial}{\partial \mathbf{u}_c} \left[-\log \sigma(\mathbf{u}_c^\top \mathbf{v}_w)\right]
$$

Let $z = \mathbf{u}_c^\top \mathbf{v}_w$. By the chain rule, $\frac{\partial}{\partial \mathbf{u}_c}[-\log \sigma(z)] = \frac{d}{dz}[-\log \sigma(z)] \cdot \frac{\partial z}{\partial \mathbf{u}_c}$. We have $\frac{\partial z}{\partial \mathbf{u}_c} = \mathbf{v}_w$ and $\frac{d}{dz}[-\log \sigma(z)] = -(1 - \sigma(z))$ (section 4.1).

$$
\boxed{\frac{\partial \mathcal{L}}{\partial \mathbf{u}_c} = -(1 - \sigma_c) \, \mathbf{v}_w}
$$

**Code:**
```python
grad_u_c = -(1.0 - sigma_pos) * v_w   # (D,)
```

### 4.4. Gradient w.r.t. a Negative Sample Embedding $\mathbf{u}_{k_i}$

Only the $i$-th negative term involves $\mathbf{u}_{k_i}$:

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{u}_{k_i}} = \frac{\partial}{\partial \mathbf{u}_{k_i}} \left[-\log \sigma(-\mathbf{u}_{k_i}^\top \mathbf{v}_w)\right]
$$

Let $z_i = \mathbf{u}_{k_i}^\top \mathbf{v}_w$.

By the chain rule,
$$
\frac{\partial}{\partial \mathbf{u}_{k_i}} \left[-\log \sigma(-z_i)\right]
= \frac{d}{dz_i} \left[-\log \sigma(-z_i)\right] \cdot \frac{\partial z_i}{\partial \mathbf{u}_{k_i}}.
$$

We have
$$
\frac{\partial z_i}{\partial \mathbf{u}_{k_i}} = \mathbf{v}_w
$$
and
$$
\frac{d}{dz_i} \left[-\log \sigma(-z_i)\right] = \sigma(z_i).
$$

(Section~4.1)

$$
\boxed{\frac{\partial \mathcal{L}}{\partial \mathbf{u}_{k_i}} = \sigma_{k_i} \, \mathbf{v}_w}
$$

**Code:**
```python
grad_u_neg = sigma_neg[:, np.newaxis] * v_w[np.newaxis, :]  # (K, D)
```

---

## 5. Parameter Update (SGD)

Each parameter is updated using stochastic gradient descent:

$$
\theta \leftarrow \theta - \eta \, \frac{\partial \mathcal{L}}{\partial \theta}
$$

Concretely for each training pair $(w, c)$ with negatives $\{k_1, \ldots, k_K\}$:

| Parameter | Update rule | Dimension |
|-----------|------------|-----------|
| $\mathbf{v}_w$ | $\mathbf{v}_w \leftarrow \mathbf{v}_w - \eta \, \frac{\partial \mathcal{L}}{\partial \mathbf{v}_w}$ | $(D,)$ |
| $\mathbf{u}_c$ | $\mathbf{u}_c \leftarrow \mathbf{u}_c - \eta \, \frac{\partial \mathcal{L}}{\partial \mathbf{u}_c}$ | $(D,)$ |
| $\mathbf{u}_{k_i}$ | $\mathbf{u}_{k_i} \leftarrow \mathbf{u}_{k_i} - \eta \, \frac{\partial \mathcal{L}}{\partial \mathbf{u}_{k_i}}$ | $(D,)$ for each $i$ |

**Code:**
```python
self.W_in[center_id]   -= lr * grad_v_w
self.W_out[context_id] -= lr * grad_u_c
self.W_out[neg_ids]    -= lr * grad_u_neg
```

### 5.1. Learning Rate Schedule

The learning rate decays linearly from $\eta_0$ to $\eta_{\min}$ over the course of training:

$$
\eta(t) = \eta_0 - (\eta_0 - \eta_{\min}) \cdot \frac{t}{T_{\text{total}}}
$$

where $t$ is the current global step and $T_{\text{total}}$ is the estimated total number of steps.
This decay allows the model to take larger parameter updates early in training for faster progress, while using smaller updates later to stabilise learning and avoid oscillating around the optimum.


**Code:** `trainer.py`, inside the training loop.

---

## 6. Dimension Summary

End-to-end shape tracking for one training step:

| Quantity | Shape | Notes |
|----------|-------|-------|
| `v_w = W_in[center_id]` | $(D,)$ | Center word input embedding |
| `u_c = W_out[context_id]` | $(D,)$ | Context word output embedding |
| `u_neg = W_out[neg_ids]` | $(K, D)$ | Negative sample output embeddings |
| `dot_pos = u_c · v_w` | scalar | Positive pair dot product |
| `dot_neg = u_neg @ v_w` | $(K,)$ | Negative pair dot products |
| `sigma_pos = σ(dot_pos)` | scalar | Sigmoid of positive dot product |
| `sigma_neg = σ(dot_neg)` | $(K,)$ | Sigmoid of negative dot products |
| `grad_v_w` | $(D,)$ | Gradient w.r.t. center embedding |
| `grad_u_c` | $(D,)$ | Gradient w.r.t. context embedding |
| `grad_u_neg` | $(K, D)$ | Gradients w.r.t. negative embeddings |

---

## 7. Subsampling of Frequent Words

Before generating training pairs, each token is stochastically discarded with probability:

$$
P_{\text{discard}}(w_i) = 1 - \left(\sqrt{\frac{t}{f(w_i)}} + \frac{t}{f(w_i)}\right)
$$

where $f(w_i) = \frac{\text{count}(w_i)}{\sum_j \text{count}(w_j)}$ is the word's relative frequency and $t$ is a threshold (typically $10^{-3}$ to $10^{-5}$).

High-frequency words (articles, prepositions) are discarded more often, which:
1. Speeds up training by reducing redundant updates.
2. Improves representation quality for less-common words by balancing the effective training distribution.

**Code:** `vocabulary.py`, `Vocabulary.compute_discard_probs()`.

---

## 8. Dynamic Context Window

For each center word, the actual window size is drawn uniformly:

$$
m_{\text{actual}} \sim \text{Uniform}\{1, 2, \ldots, m\}
$$

where $m$ is the maximum window size. This effectively gives higher weight to nearby context words (they are included in every window draw) and lower weight to distant ones (only included when the sampled window is large enough).

**Code:** `dataset.py`, `generate_training_pairs()`.

---

## 9. Initialisation

Both embedding matrices are initialised with Xavier uniform:

$$
W_{ij} \sim \text{Uniform}\left(-\sqrt{\frac{6}{V + D}}, \; +\sqrt{\frac{6}{V + D}}\right)
$$

This keeps the variance of the initial forward-pass activations in a reasonable range, preventing gradients from vanishing or exploding in the first few updates.

**Code:** `model.py`, `SkipGramModel.__init__()`.

---

## 10. Gradient Verification

The implementation includes a finite-difference gradient check (see `tests/test_model.py`). For each parameter $\theta_j$, the numerical gradient is:

$$
\frac{\partial \mathcal{L}}{\partial \theta_j} \approx \frac{\mathcal{L}(\theta_j + \varepsilon) - \mathcal{L}(\theta_j - \varepsilon)}{2\varepsilon}
$$

with $\varepsilon = 10^{-5}$. The relative error between analytic and numerical gradients is:

$$
\mathrm{rel\_error}_j = \frac{|g_{\mathrm{analytic},j} - g_{\mathrm{numerical},j}|}{\max(|g_{\mathrm{analytic},j}|, \; |g_{\mathrm{numerical},j}|) + 10^{-12}}
$$

We assert $\mathrm{rel\_error}_j < 10^{-4}$ for all $j$, which is checked for:

- $\frac{\partial \mathcal{L}}{\partial \mathbf{v}_w}$ (center embedding)
- $\frac{\partial \mathcal{L}}{\partial \mathbf{u}_c}$ (context embedding)
- $\frac{\partial \mathcal{L}}{\partial \mathbf{u}_{k_i}}$ (each negative sample embedding)
