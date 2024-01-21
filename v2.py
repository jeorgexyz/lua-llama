# Decoder only Transformer

import torch
import torch.nn as nn
from torch.nn import functional as F


# hyperparameters
batch_size = 32
block_size = 8
max_iters = 5000
eval_interval = 500
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 32

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)

# create a mapping from characters to integers

stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of size `bptt` from data
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()

def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)

        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * C**-0.5

        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))

        wei = F.softmax(wei, dim=-1)

        # perform the weighted aggregation of the values
        v = self.value(x)
        out = wei @ v
        return out
    
    class MultiHeadAttention(nn.Module):
        """ multiple heads of self-attention in parallel """

        def __init__(self, num_heads, head_size):
            super().__init__()
            self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
            self.proj = nn.Linear(n_embd, n_embd)
            self.dropout = nn.Dropout(0.1)

        def forward(self, x):
            out = torch.cat([h(x) for h in self.heads], dim=-1)
            out = self.dropout(self.proj(out))
            return out
        
        class FeedForward(nn.Module):
            """ a simple linear layer + tanh + dropout """

            def __init__(self, n_embd):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(n_embd, 4 * n_embd),
                    nn.ReLU(),
                    nn.Linear(4 * n_embd, n_embd),
                    nn.Dropout(0.1),
                )

            def forward(self, x):
                return self.net(x)
    

    # simple bigram model
    class BigramLanguageModel(nn.Module):
        def __init__(self):
            super().__init__()
            # each token directly reads off the logits for the next token from a lookup table
            self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
            self.position_embedding_table = nn.Embedding(block_size, n_embd)
            #self.head = Head(n_embd)
            self.sa_heads = MultiHeadAttention(4, n_embd//4) # 4 heads of 8 dim self-attention
            
            self.ln_f = nn.LayerNorm(n_embd) # final layer norm
            self.ffwd = FeedForward(n_embd) # feed-forward network
            self.lm_head = nn.Linear(n_embd, vocab_size)

        def forward(self, idx, targets=None):
            B, T = idx.shape

            # idx and targets are both (B,T) tensor of integers
            tok_emb = self.token_embedding_table(idx) # (B,T,C)
            pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
            x = tok_emb + pos_emb # (B,T,C)
            x = self.sa_heads(x)
            x = self.ffwd(x)
            logits = self.lm_head(x)
            