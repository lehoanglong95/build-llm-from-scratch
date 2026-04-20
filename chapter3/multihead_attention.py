import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, num_heads, context_length, dropout, bias=False):
        super().__init__()
        assert d_out % num_heads == 0
        self.d_in = d_in
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dimension = d_out // num_heads
        self.context_length = context_length
        self.W_query = nn.Linear(d_in, d_out, bias)
        self.W_key = nn.Linear(d_in, d_out, bias)
        self.W_value = nn.Linear(d_in, d_out, bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))
        self.output_projection = nn.Linear(d_out, d_out, bias)

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        keys = keys.view(b, num_tokens, self.num_heads, self.head_dimension)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dimension)
        values = values.view(b, num_tokens, self.num_heads, self.head_dimension)

        keys = keys.transpose(1, 2) # (b, num_heads, num_tokens, head_dimension)
        queries = queries.transpose(1, 2) # (b, num_heads, num_tokens, head_dimension)
        values = values.transpose(1, 2) # (b, num_heads, num_tokens, head_dimension)

        atten_scores = torch.matmul(queries, keys.transpose(-2, -1)) # (b, num_heads, num_tokens, num_tokens)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        atten_scores.masked_fill_(mask_bool, -torch.inf)
        atten_weights = torch.softmax(atten_scores / self.head_dimension ** 0.5, dim=-1)
        atten_weights = self.dropout(atten_weights)

        context_vectors = torch.matmul(atten_weights, values) # (b, num_heads, num_tokens, head_dimension)
        context_vectors = context_vectors.transpose(1, 2).contiguous().view(b, num_tokens, self.d_out) # (b, num_tokens, d_out)
        output = self.output_projection(context_vectors) # (b, num_tokens, d_out)
        return output