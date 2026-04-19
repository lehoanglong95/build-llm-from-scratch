import torch
inputs = torch.tensor([
    [0.43, 0.15, 0.89],
    [0.55, 0.87, 0.66],
    [0.57, 0.85, 0.64],
    [0.22, 0.58, 0.33],
    [0.77, 0.25, 0.10],
    [0.05, 0.80, 0.55],
])

attn_scores = torch.empty(inputs.shape[0], inputs.shape[0])
for i, x_i in enumerate(inputs):
    for j, x_j in enumerate(inputs):
        attn_scores[i, j] = torch.dot(x_i, x_j)
print(attn_scores)

attn_weights = torch.nn.Softmax(dim=1)(attn_scores)
print(attn_weights)

context_vectors = torch.matmul(attn_weights, inputs)
print(context_vectors)

a = inputs[:, 0]
b = attn_weights[0, :]
print(torch.matmul(b, a))