
import torch
import torch.nn as nn

from chapter3.multihead_attention import MultiHeadAttention
from chapter4.config import GPTConfig
from chapter4.gpt_model import LayerNorm, FeedForward

class TransformerBlock(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.attention = MultiHeadAttention(
            config.emb_dim,
            config.emb_dim,
            config.n_heads,
            config.context_length,
            config.dropout,
            config.qkv_bias
        )
        self.norm1 = LayerNorm(config.emb_dim)
        self.norm2 = LayerNorm(config.emb_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.ff = FeedForward(config)
    
    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.attention(x)
        x = self.dropout(x)
        x += shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.attention(x)
        x = self.dropout(x)

        return x + shortcut

if __name__ == "__main__":
    x = torch.rand(2, 4, 768)
    block = TransformerBlock(GPTConfig())
    output = block(x)
    print(output.shape)
