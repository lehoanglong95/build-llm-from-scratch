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
    for idx in random_idxs:
        

class BigramModel()