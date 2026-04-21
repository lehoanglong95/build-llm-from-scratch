from dataclasses import dataclass
from typing import Any
import torch
import torch.nn as nn

@dataclass
class GPTConfig:
    vocab_size: int = 50257
    context_length: int = 1024
    emb_dim: int = 768
    n_heads: int = 12
    n_layers: int = 12
    dropout: float = 0.1
    qkv_bias: bool = False

class GPTModel(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.tok_emb = nn.Embedding(config.vocab_size, config.emb_dim)
        self.pos_emb = nn.Embedding(config.context_length, config.emb_dim)
        self.drop_emb = nn.Dropout(config.dropout)
        self.transformer_block = nn.Sequential(
            *[DummyTransformerBlock(config)
              for _ in range(config.n_layers)]
        )
        self.final_norm = LayerNorm(config.emb_dim)
        self.output_head = nn.Linear(config.emb_dim, config.vocab_size, bias=False)

    def forward(self, x):
        batch_size, seq_len = x.shape
        tok_embeds = self.tok_emb(x)
        pos_embeds = self.pos_emb(
            torch.arange(seq_len, device=x.device)
        )
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.transformer_block(x)
        x = self.final_norm(x)
        logits = self.output_head(x)
        return logits


class DummyTransformerBlock(nn.Module):

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()

    def forward(self, x):
        return x

class LayerNorm(nn.Module):
    """
    This class help normalise the matrix to reduce vanishing or exploding gradients
    """
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim)) # trainable params
        self.shift = nn.Parameter(torch.zeros(emb_dim)) # trainable params
        
    def forward(self, x: torch.Tensor):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift