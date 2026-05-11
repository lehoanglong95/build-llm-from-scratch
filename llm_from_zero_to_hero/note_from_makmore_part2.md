# Notes from Makemore Part 2 (MLP)

Reference: Bengio et al. (2003) — "A Neural Probabilistic Language Model"

---

## Loss Function

Use `F.cross_entropy` instead of manually computing softmax + negative log-likelihood, for two reasons:

1. **Fused kernel** — PyTorch executes the combined operation in a single optimized CUDA kernel, which is faster.
2. **Numerical stability** — Manual softmax is vulnerable to overflow. When logits are large (e.g., all values ~100), `exp(100)` becomes `inf`, causing `nan` in the loss. `cross_entropy` internally shifts logits by subtracting the max before exponentiating, which prevents this.

```python
# Avoid this
probs = F.softmax(logits, dim=1)
loss = -probs[range(n), y].log().mean()

# Prefer this
loss = F.cross_entropy(logits, y)
```

---

## Learning Rate

### Finding a Good Learning Rate

- Run a sweep over many candidate learning rates (e.g., log-spaced from `1e-3` to `1`) at the start of training.
- Plot loss vs. learning rate and pick the value just before loss starts rising.

```python
lrs = torch.linspace(-3, 0, 1000)  # exponents
lrs = 10 ** lrs                     # actual lr values
```

### Learning Rate Decay

- Start with a higher learning rate when the model is far from the optimum — it can take large steps safely.
- Reduce the learning rate (e.g., by 10×) once loss plateaus to fine-tune around the optimum.
- This is a simple form of **learning rate scheduling**.

---

## Dataset Splits

Always split data into three sets:

| Split      | Typical size | Purpose                                      |
| ---------- | ------------ | -------------------------------------------- |
| Train      | 80%          | Model learns from this                       |
| Validation | 10%          | Tune hyperparameters (lr, hidden size, etc.) |
| Test       | 10%          | Final unbiased evaluation — use sparingly    |

**Why not just train and test?**
If you repeatedly evaluate on the test set and adjust based on it, you implicitly overfit to the test set. The validation set absorbs this tuning, keeping the test set as a true held-out measure.

---

## Tuning the Model

### Diagnosing Underfitting vs. Overfitting

| Symptom                        | Diagnosis    | Fix                                                      |
| ------------------------------ | ------------ | -------------------------------------------------------- |
| Train loss high, val loss high | Underfitting | Increase model capacity (more neurons, larger embedding) |
| Train loss low, val loss high  | Overfitting  | Regularization, more data, or smaller model              |
| Train ≈ val loss (both good)   | Good fit     | Possibly squeeze more capacity                           |

### Ways to Increase Capacity

- Increase embedding dimension (e.g., 2 → 10)
- Increase hidden layer size (e.g., 100 → 300)
- Add more hidden layers

### Visualizing Embeddings

- After training, plot the 2D character embeddings to inspect what the model learned.
- Well-trained embeddings cluster similar characters together (e.g., vowels near each other).
- If the embedding dimension is >2, apply PCA or t-SNE to project down for visualization.

---

## Architecture Summary (MLP Character Model)

```
Input: context window of k characters
  → Embedding lookup: each char → vector of size d
  → Concatenate embeddings: k*d input to MLP
  → Hidden layer: tanh(W1 @ x + b1)
  → Output layer: W2 @ h + b2  (logits over vocab)
  → Loss: cross_entropy(logits, next_char)
```
