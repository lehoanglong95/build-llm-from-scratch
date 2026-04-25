
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

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
 #1
        pos_embeds = self.pos_emb(
            torch.arange(seq_len, device=in_idx.device)
        )
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

if __name__ == "__main__":
    x = torch.rand(2, 4, 768)
    block = TransformerBlock(GPTConfig())
    output = block(x)
    print(output.shape)
