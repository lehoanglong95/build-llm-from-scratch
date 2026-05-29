# Backpropagation Example: Step-by-Step Math

This document walks through a concrete backpropagation example with actual numbers, showing how gradients are calculated and weights are updated in a simple neural network.

## Network Architecture

- **Input:** 1×3 vector (single sample)
- **Layer 1:** 3×4 weights + 4 biases, tanh activation
- **Layer 2:** 4×1 weights + 1 bias, no activation
- **Output:** Single number
- **Loss:** Mean Squared Error (MSE)

---

## Network Setup

**Input:** x = [1, 2, 3] (1×3, using single sample for clarity)

**Layer 1:** 3×4 weights + 4 biases

```
W1 = [[0.1, 0.2, 0.3, 0.4],
      [0.2, 0.1, 0.4, 0.3],
      [0.3, 0.4, 0.1, 0.2]]

b1 = [0.1, 0.2, 0.1, 0.2]
```

**Layer 2:** 4×1 weights + 1 bias

```
W2 = [[0.5],
      [0.6],
      [0.7],
      [0.8]]

b2 = 0.1
```

**Target:** y_true = 3.0

**Learning rate:** α = 0.1

---

## Forward Pass

### Layer 1 (Linear Transformation)

```
h1 = x @ W1 + b1

h1[0] = 1(0.1) + 2(0.2) + 3(0.3) + 0.1 = 0.1 + 0.4 + 0.9 + 0.1 = 1.5
h1[1] = 1(0.2) + 2(0.1) + 3(0.4) + 0.2 = 0.2 + 0.2 + 1.2 + 0.2 = 1.8
h1[2] = 1(0.3) + 2(0.4) + 3(0.1) + 0.1 = 0.3 + 0.8 + 0.3 + 0.1 = 1.5
h1[3] = 1(0.4) + 2(0.3) + 3(0.2) + 0.2 = 0.4 + 0.6 + 0.6 + 0.2 = 1.8

h1 = [1.5, 1.8, 1.5, 1.8]
```

### Activation (tanh)

```
a1 = tanh(h1)

a1[0] = tanh(1.5) ≈ 0.905
a1[1] = tanh(1.8) ≈ 0.947
a1[2] = tanh(1.5) ≈ 0.905
a1[3] = tanh(1.8) ≈ 0.947

a1 = [0.905, 0.947, 0.905, 0.947]
```

### Layer 2 (Linear Transformation)

```
h2 = a1 @ W2 + b2

h2 = 0.905(0.5) + 0.947(0.6) + 0.905(0.7) + 0.947(0.8) + 0.1
   = 0.453 + 0.568 + 0.634 + 0.758 + 0.1
   = 2.513

y_pred = 2.513
```

### Loss (MSE)

```
L = (y_pred - y_true)²
  = (2.513 - 3.0)²
  = (-0.487)²
  = 0.237
```

---

## Backward Pass

Now we compute gradients flowing backward through the network using the chain rule.

### Loss Gradient

```
∂L/∂y_pred = 2(y_pred - y_true)
           = 2(2.513 - 3.0)
           = 2(-0.487)
           = -0.974
```

### Layer 2 Gradients

**Bias gradient:**

```
∂L/∂b2 = ∂L/∂y_pred = -0.974
```

**Weight gradients:**

```
∂L/∂W2 = ∂L/∂y_pred · a1ᵀ
       = -0.974 · [[0.905],
                   [0.947],
                   [0.905],
                   [0.947]]

∂L/∂W2[0] = -0.974 × 0.905 = -0.881
∂L/∂W2[1] = -0.974 × 0.947 = -0.922
∂L/∂W2[2] = -0.974 × 0.905 = -0.881
∂L/∂W2[3] = -0.974 × 0.947 = -0.922

∂L/∂W2 = [[-0.881],
          [-0.922],
          [-0.881],
          [-0.922]]
```

**Gradient flowing to activation:**

```
∂L/∂a1 = ∂L/∂y_pred · W2ᵀ
       = -0.974 · [0.5, 0.6, 0.7, 0.8]

∂L/∂a1[0] = -0.974 × 0.5 = -0.487
∂L/∂a1[1] = -0.974 × 0.6 = -0.584
∂L/∂a1[2] = -0.974 × 0.7 = -0.682
∂L/∂a1[3] = -0.974 × 0.8 = -0.779

∂L/∂a1 = [-0.487, -0.584, -0.682, -0.779]
```

### Tanh Gradient

The derivative of tanh is:

```
∂tanh(x)/∂x = 1 - tanh²(x)
```

Computing for each element:

```
For h1[0] = 1.5: 1 - 0.905² = 1 - 0.819 = 0.181
For h1[1] = 1.8: 1 - 0.947² = 1 - 0.897 = 0.103
For h1[2] = 1.5: 1 - 0.905² = 1 - 0.819 = 0.181
For h1[3] = 1.8: 1 - 0.947² = 1 - 0.897 = 0.103

tanh'(h1) = [0.181, 0.103, 0.181, 0.103]
```

**Gradient flowing to Layer 1 pre-activation:**

```
∂L/∂h1 = ∂L/∂a1 ⊙ (1 - a1²)   [⊙ means element-wise multiplication]
       = [-0.487, -0.584, -0.682, -0.779] ⊙ [0.181, 0.103, 0.181, 0.103]

∂L/∂h1[0] = -0.487 × 0.181 = -0.088
∂L/∂h1[1] = -0.584 × 0.103 = -0.060
∂L/∂h1[2] = -0.682 × 0.181 = -0.123
∂L/∂h1[3] = -0.779 × 0.103 = -0.080

∂L/∂h1 = [-0.088, -0.060, -0.123, -0.080]
```

### Layer 1 Gradients

**Bias gradient:**

```
∂L/∂b1 = ∂L/∂h1 = [-0.088, -0.060, -0.123, -0.080]
```

**Weight gradients:**

```
∂L/∂W1 = xᵀ · ∂L/∂h1
       = [[1],    · [-0.088, -0.060, -0.123, -0.080]
          [2],
          [3]]
```

Computing each element (each row of x multiplied by each element of gradient):

```
Row 0 (x[0] = 1):
∂L/∂W1[0,0] = 1 × -0.088 = -0.088
∂L/∂W1[0,1] = 1 × -0.060 = -0.060
∂L/∂W1[0,2] = 1 × -0.123 = -0.123
∂L/∂W1[0,3] = 1 × -0.080 = -0.080

Row 1 (x[1] = 2):
∂L/∂W1[1,0] = 2 × -0.088 = -0.176
∂L/∂W1[1,1] = 2 × -0.060 = -0.120
∂L/∂W1[1,2] = 2 × -0.123 = -0.246
∂L/∂W1[1,3] = 2 × -0.080 = -0.160

Row 2 (x[2] = 3):
∂L/∂W1[2,0] = 3 × -0.088 = -0.264
∂L/∂W1[2,1] = 3 × -0.060 = -0.180
∂L/∂W1[2,2] = 3 × -0.123 = -0.369
∂L/∂W1[2,3] = 3 × -0.080 = -0.240

∂L/∂W1 = [[-0.088, -0.060, -0.123, -0.080],
          [-0.176, -0.120, -0.246, -0.160],
          [-0.264, -0.180, -0.369, -0.240]]
```

---

## Weight Update

Using gradient descent: **new_weight = old_weight - learning_rate × gradient**

### Layer 2 Updates

**Weights:**

```
W2[0]_new = 0.5 - 0.1 × (-0.881) = 0.5 + 0.0881 = 0.588
W2[1]_new = 0.6 - 0.1 × (-0.922) = 0.6 + 0.0922 = 0.692
W2[2]_new = 0.7 - 0.1 × (-0.881) = 0.7 + 0.0881 = 0.788
W2[3]_new = 0.8 - 0.1 × (-0.922) = 0.8 + 0.0922 = 0.892

W2_new = [[0.588],
          [0.692],
          [0.788],
          [0.892]]
```

**Bias:**

```
b2_new = 0.1 - 0.1 × (-0.974) = 0.1 + 0.0974 = 0.197
```

### Layer 1 Updates

**Weights (Row 0):**

```
W1[0,0]_new = 0.1 - 0.1 × (-0.088) = 0.1 + 0.0088 = 0.109
W1[0,1]_new = 0.2 - 0.1 × (-0.060) = 0.2 + 0.0060 = 0.206
W1[0,2]_new = 0.3 - 0.1 × (-0.123) = 0.3 + 0.0123 = 0.312
W1[0,3]_new = 0.4 - 0.1 × (-0.080) = 0.4 + 0.0080 = 0.408
```

**Weights (Row 1):**

```
W1[1,0]_new = 0.2 - 0.1 × (-0.176) = 0.2 + 0.0176 = 0.218
W1[1,1]_new = 0.1 - 0.1 × (-0.120) = 0.1 + 0.0120 = 0.112
W1[1,2]_new = 0.4 - 0.1 × (-0.246) = 0.4 + 0.0246 = 0.425
W1[1,3]_new = 0.3 - 0.1 × (-0.160) = 0.3 + 0.0160 = 0.316
```

**Weights (Row 2):**

```
W1[2,0]_new = 0.3 - 0.1 × (-0.264) = 0.3 + 0.0264 = 0.326
W1[2,1]_new = 0.4 - 0.1 × (-0.180) = 0.4 + 0.0180 = 0.418
W1[2,2]_new = 0.1 - 0.1 × (-0.369) = 0.1 + 0.0369 = 0.137
W1[2,3]_new = 0.2 - 0.1 × (-0.240) = 0.2 + 0.0240 = 0.224
```

**Complete updated W1:**

```
W1_new = [[0.109, 0.206, 0.312, 0.408],
          [0.218, 0.112, 0.425, 0.316],
          [0.326, 0.418, 0.137, 0.224]]
```

**Biases:**

```
b1[0]_new = 0.1 - 0.1 × (-0.088) = 0.1 + 0.0088 = 0.109
b1[1]_new = 0.2 - 0.1 × (-0.060) = 0.2 + 0.0060 = 0.206
b1[2]_new = 0.1 - 0.1 × (-0.123) = 0.1 + 0.0123 = 0.112
b1[3]_new = 0.2 - 0.1 × (-0.080) = 0.2 + 0.0080 = 0.208

b1_new = [0.109, 0.206, 0.112, 0.208]
```

---

## Summary

### Key Insights

1. **Gradient Flow:** The gradient flows backward through the network:

   ```
   ∂L/∂y_pred → ∂L/∂W2 → ∂L/∂a1 → ∂L/∂h1 → ∂L/∂W1
   ```

2. **Chain Rule:** Each gradient is computed by multiplying the upstream gradient with the local derivative.

3. **Weight Updates:** All weights moved in the direction that reduces loss. Since gradients were negative and we subtract them, weights increased slightly.

4. **Activation Derivative:** The tanh derivative (0.103-0.181) acts as a "gate", reducing gradient magnitude as it flows backward. This is why gradients become smaller in earlier layers (vanishing gradient problem).

5. **Input Influence:** Larger input values (x[2] = 3) contribute more to weight gradients than smaller values (x[0] = 1).

### Gradient Flow Visualization

```
Loss: 0.237
  ↓ (∂L/∂y_pred = -0.974)
Layer 2: W2, b2
  ↓ (∂L/∂a1 = [-0.487, -0.584, -0.682, -0.779])
Tanh: 1 - tanh²(h1)
  ↓ (∂L/∂h1 = [-0.088, -0.060, -0.123, -0.080])
Layer 1: W1, b1
  ↓
Input: x
```

### Next Step

After this weight update, if we run another forward pass with the same input, the loss should be slightly lower than 0.237, showing that the network has learned slightly better weights.
