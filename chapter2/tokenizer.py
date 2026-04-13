import re

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
all_words = sorted(set(preprocessed))

vocab = {word: idx for idx, word in enumerate(all_words)}

class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.vocab = vocab
        self.int_to_word = {idx: word for word, idx in vocab.items()}
    
    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        return [self.vocab[word] for word in preprocessed]

    def decode(self, ids):
        text = " ".join([self.int_to_word[i] for i in ids]) 
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text

all_words_v2 = all_words + ["<|endoftext|>", "<|unk|>"]
vocab_v2 = {word: idx for idx, word in enumerate(all_words_v2)}

class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.vocab = vocab
        self.int_to_word = {idx: word for word, idx in vocab.items()}
    
    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        return [self.vocab.get(word, self.vocab["<|unk|>"]) for word in preprocessed]

    def decode(self, ids):
        text = " ".join([self.int_to_word[i] for i in ids]) 
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text

text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."
text = " <|endoftext|> ".join((text1, text2))

tokenizer1 = SimpleTokenizerV1(vocab)
tokenizer2 = SimpleTokenizerV2(vocab_v2)

# print(tokenizer1.encode(text)) # Error: KeyError: 'Hello'
print(tokenizer2.encode(text))

from torch.utils.data import Dataset, DataLoader
import tiktoken

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i: i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
        
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
        

def create_dataloader_v1(txt, batch_size=4, max_length=256, 
                         stride=128, shuffle=True, drop_last=True,
                         num_workers=0):

    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return dataloader

# Token Embeddings
import torch
inputs_ids = torch.tensor([2, 3, 5, 1])

vocab_size = 6
output_dim = 3

torch.manual_seed(123)
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
print(embedding_layer.weight)
print(embedding_layer(torch.tensor([3])))
print(embedding_layer(inputs_ids))

# Encoding word positions
vocab_size = 50257
output_dim = 256
batch_size = 8
max_length = 4 # context length

token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
dataloader = create_dataloader_v1(raw_text, batch_size=batch_size, max_length=max_length, stride=max_length, shuffle=False)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("Token IDs:\n", inputs)
print("\nInputs shape:\n", inputs.shape)

token_embeddings = token_embedding_layer(inputs)
print("\nToken embeddings shape:\n", token_embeddings.shape)

pos_embedding_layer = torch.nn.Embedding(max_length, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(max_length))
print("\nPosition embeddings shape:\n", pos_embeddings.shape)

input_embeddings = token_embeddings + pos_embeddings
print("\nInput embeddings shape:\n", input_embeddings.shape)