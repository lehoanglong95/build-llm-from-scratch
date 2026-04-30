import tiktoken
import torch
import torch.nn as nn

from chapter4.config import GPTConfig
from chapter4.transformer import GPTModel

tokenizer = tiktoken.get_encoding("gpt2")
start_context = "Hello, I am"
encoded = tokenizer.encode(start_context)
encoded_tensor = torch.tensor(encoded).unsqueeze(0)

config = GPTConfig()
model = GPTModel(config)

def generate_text(inp, model, max_new_token, context_length):
    for _ in range(max_new_token):
        inp = inp[:, -context_length:]
        with torch.no_grad():
            output = model(inp)
        
        last_token = output[:, -1, :]
        idx_next = torch.argmax(last_token, dim=-1, keepdim=True)
        inp = torch.cat((inp, idx_next), dim=1)
    return inp

output = generate_text(encoded_tensor, model, 6, config.context_length)
print(output.shape)