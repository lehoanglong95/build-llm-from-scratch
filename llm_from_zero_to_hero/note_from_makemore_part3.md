# Notes from Makemore Part 3 (Activations, Gradients & BatchNorm)

Reference: Karpathy — "Building makemore Part 3: Activations & Gradients, BatchNorm"

Focus: the internal dynamics of neural networks during training — specifically the behavior of activations and gradients in an MLP.

---

## 1. Initialization and the "Hockey Stick" Loss

### Predicting the Initial Loss

At initialization, a well-configured network should have a **predictable** loss. For a 27-character classification task, a uniform output distribution should give:

```
loss ≈ -ln(1/27) ≈ 3.29
```

### The "Confidently Wrong" Problem

If the network is improperly initialized, initial losses can be much higher (e.g., ~27) because logits take extreme values, making the model **confidently wrong** about its predictions.

### Eliminating the Hockey Stick

- Scale **down** the output-layer weights `W2` (e.g., multiply by a small constant like `0.01`).
- **Zero** the output-layer biases `b2`.

This lowers the initial loss to the expected value and eliminates the "hockey stick" loss curve, where the first few thousand iterations are wasted just shrinking weights back to reasonable magnitudes.

---

## 2. Activation Saturation and Dead Neurons

### The Tanh Gradient

The `tanh` function squashes values into `[-1, 1]`. Its local gradient is:

```
d/dx tanh(x) = 1 - tanh(x)^2
```

If the input is very large (positive or negative), the output saturates near `±1` and the gradient becomes ~0.

### Gradient Vanishing

When activations sit in the **flat regions** of the nonlinearity, the gradient is killed and backpropagation stops flowing through those neurons.

### Dead Neurons

A neuron is **dead** if its weights and biases are initialized such that it never activates in its non-flat region for _any_ example in the dataset — it can never learn.

This problem also occurs with:

- **Dead ReLUs** — neuron always outputs 0
- **Saturated sigmoid** — output stuck near 0 or 1

---

## 3. Principled Initialization (Kaiming / He Init)

### Fan-in and Variance

To keep activations well-behaved through a deep network, weights should be scaled to preserve the standard deviation of inputs. A common rule:

```
W ~ N(0, 1 / sqrt(fan_in))
```

where `fan_in` is the number of input connections to the neuron.

### Gain for Nonlinearities

Nonlinearities like `tanh` and `ReLU` are **contractive** — they squash the distribution. A **gain** multiplier compensates:

| Nonlinearity | Gain    |
| ------------ | ------- |
| tanh         | 5/3     |
| ReLU         | sqrt(2) |
| linear       | 1       |

### Kaiming Initialization

```python
# PyTorch equivalent
torch.nn.init.kaiming_normal_(W, mode='fan_in', nonlinearity='tanh')

# Or manually
W = torch.randn(fan_in, fan_out) * (gain / fan_in ** 0.5)
```

This ensures activations **and** gradients remain roughly unit Gaussian across many layers — essential for training deep networks.

---

## 4. Batch Normalization (BatchNorm)

### The Core Concept

Instead of relying on perfect initialization, BatchNorm **explicitly normalizes** the pre-activations to be unit Gaussian over the current batch:

```
x_hat = (x - mean_batch) / sqrt(var_batch + eps)
```

### Scale and Shift

To allow the network to recover non-Gaussian distributions when useful, BatchNorm adds two **learnable** parameters:

```
y = gamma * x_hat + beta
```

- `gamma` — gain
- `beta` — bias

### Regularization Side Effect

Because BatchNorm couples examples within a batch through the shared mean/variance, it introduces a small amount of **noise** (entropy) into each example's representation. This acts as a regularizer and helps prevent overfitting.

### Useless Biases

When a linear layer is immediately followed by BatchNorm, its bias `b1` becomes **useless**:

```
BN(W @ x + b) = BN(W @ x)   # b is subtracted by the mean step
```

So in practice, omit the bias from any linear layer feeding directly into BatchNorm.

### Train vs. Inference

At training time, BatchNorm uses **batch statistics**. At inference, it uses **running estimates** of mean/variance accumulated during training (so single-example inference is deterministic).

```python
# During training
bnmean = x.mean(0, keepdim=True)
bnvar  = x.var(0, keepdim=True)

# Running estimates (updated each step)
running_mean = 0.999 * running_mean + 0.001 * bnmean
running_var  = 0.999 * running_var  + 0.001 * bnvar
```

---

## Takeaways

- **Diagnose by looking at activations and gradients**, not just the loss curve.
- Initialize so that the first forward/backward pass already has unit-Gaussian-like statistics.
- BatchNorm makes initialization less critical, but it isn't free — it couples examples in a batch and adds bookkeeping (running stats, train/eval mode).
- The "kill the hockey stick" trick (small `W2`, zero `b2`) is a cheap and very effective sanity-check fix.
