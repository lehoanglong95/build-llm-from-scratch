"""
Flow:
- load dataset from tinyshakespeare
- create stoi (string to integer), itos(integer to string), encode, decode function
- create train, val dataset
- create bigram model: forward, genereate function
"""

import torch
import torch.nn as nn
from torch.nn import functional as F

BATCH_SIZE = 4
CONTEXT_LENGTH = 8
EVAL_ITERS = 300
EPOCH = 10_000

with open("./tiny_shakespeare.txt") as f:
    data = f.read()

uni_chars = set(sorted(list(data)))
vocab_size = len(uni_chars)
learning_rate = 1e-3
device = "cuda" if torch.cuda.is_available() else "cpu"
stoi = {ch: i for i, ch in enumerate(uni_chars)}
itos = {i: ch for i, ch in enumerate(uni_chars)}
encode = lambda s: [stoi[ch] for ch in s]
decode = lambda i: "".join([itos[ii] for ii in i])
threshold = int(0.9 * len(data))
train_data = data[:threshold]
val_data = data[threshold:]

def get_data(split: str):
    dataset = train_data if split == "train" else val_data
    random_idxs = torch.randint(0, len(dataset) - CONTEXT_LENGTH, (BATCH_SIZE, ))
    x_stack = []
    y_stack = []
    for idx in random_idxs:
        x_stack.append(torch.tensor(encode(dataset[idx: idx + CONTEXT_LENGTH]), dtype=torch.long))
        y_stack.append(torch.tensor(encode(dataset[idx + 1: idx + 1 + CONTEXT_LENGTH]), dtype=torch.long))

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
        super().__init__()
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, vocab_size)
    
    def forward(self, X, Y=None):
        B, C = X.shape # B: batch size, C: context length, E: embeddings
        logits = self.embed(X) # B x C x E
        if Y is None:
            loss = None
        else:
            logits = logits.view(B * C, -1)
            Y = Y.view(B * C)
            loss = F.cross_entropy(logits, Y)
        return logits, loss

    def generate(self, X, max_new_token):
        for _ in range(max_new_token):
            logits, _ = self(X) # logits: (B x C, E)
            logits = logits[:, -1, :] # B x E
            probs = F.softmax(logits, -1)
            next_idx = torch.multinomial(probs, num_samples=1)
            X = torch.cat((X, next_idx), dim=-1)
        return X

# train
model = BigramModel(vocab_size=vocab_size)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(EPOCH):
    if epoch % EVAL_ITERS == 0:
        loss = calculate_loss(model)
        print(f"epoch: {epoch} get train loss: {loss["train"]} and val loss: {loss["val"]}")
    X_train, Y_train = get_data("train")    
    _, loss = model(X_train, Y_train)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, 500)[0].tolist()))
