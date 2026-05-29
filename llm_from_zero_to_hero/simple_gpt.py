"""
Flow:
- load dataset from tinyshakespeare
- create stoi (string to integer), itos(integer to string), encode, decode function
- create train, val dataset
- create bigram model: forward, genereate function
"""

import torch
import torch.nn as nn
import torch.functional as F

BATCH_SIZE = 4
CONTEXT_LENGTH = 8
EVAL_ITERS = 300

with open("./tiny_shakespeare.txt") as f:
    data = f.read()

uni_chars = set(sorted(list(data)))
stoi = {ch: i for i, ch in enumerate(uni_chars)}
itos = {i: ch for i, ch in enumerate(uni_chars)}
encode = lambda s: [stoi[ch] for ch in s]
decode = lambda i: "".join([itos[ii] for ii in i])
threshold = 0.9 * len(data)
train_data = data[:threshold]
val_data = data[threshold:]

def get_data(split: str):
    dataset = train_data if split == "train" else val_data
    random_idxs = torch.randint(0, len(dataset) - CONTEXT_LENGTH, (BATCH_SIZE, ))
    x_stack = []
    y_stack = []
    for idx in random_idxs:
        x_stack.append(torch.tensor(dataset[idx: idx + CONTEXT_LENGTH], dtype=torch.long))
        y_stack.append(torch.tensor(dataset[idx + 1: idx + 1 + CONTEXT_LENGTH], dtype=torch.long))

    return torch.stack(x_stack), torch.stack(y_stack)

    
@torch.no_grad
def calculate_loss(model):
    out = {}
    for split in ["train", "val"]:
        model.eval()
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            X, Y = get_data(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    return out

class BigramModel(nn.Module):

    def __init__(self, vocab_size):
        pass
    
    def forward(self, X, Y):
        pass

    def generate(self, X):
        pass