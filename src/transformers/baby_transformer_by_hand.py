import torch
from torch.nn import functional as F
from transformers import AutoTokenizer
import math
import numpy as np

# Simple implementation of attention replicating the hand/latex drawn example
# from Zed: https://x.com/zmkzmkz/status/1798421634883490274

sequence = ["Glasses are just really versatile"]
sequence_length = 6
batch_size = len(sequence)

# Tokenize string
tokenizer = AutoTokenizer.from_pretrained("gpt2")
x = [tokenizer.encode(i) for i in sequence]
x = torch.tensor(x)

# Text embedding
embedding_index_size = 100000
embedding_dimension = 4
embeddings = torch.nn.Embedding(
    num_embeddings=embedding_index_size, embedding_dim=embedding_dimension
)
x = embeddings(x)


# Positional embeddings
def sin_cos_positional(length, embedding_dim, n=10000):
    assert length % 2 == 0
    assert embedding_dim % 2 == 0
    pe = np.zeros(length * embedding_dim).reshape(length, embedding_dim)
    for k in range(length):
        for i in range(embedding_dim // 2):
            theta = k / n ** (2 * i / n)
            pe[k, 2 * i] = math.sin(theta)
            pe[k, 2 * i + 1] = math.cos(theta)
    return pe


pos_x = sin_cos_positional(sequence_length, embedding_dimension)
x = x + torch.tensor(pos_x)

# Layer Norm
ln = torch.nn.LayerNorm(embedding_dimension).to("cpu", dtype=x.dtype)
ln_x = ln(x)

# Multi-head attention
num_heads = 2


def attend_single_head(ln_x):
    head_dimension = embedding_dimension // num_heads
    Q = torch.nn.Linear(embedding_dimension, head_dimension).to("cpu", dtype=x.dtype)
    q = Q(ln_x)

    K = torch.nn.Linear(embedding_dimension, head_dimension).to("cpu", dtype=x.dtype)
    k = K(ln_x)
    k = torch.transpose(k, 1, 2)

    qk = torch.einsum("bsh, bht -> bst", q, k)
    qk = qk.masked_fill(
        torch.tril(torch.ones(sequence_length, sequence_length)).view(
            1, sequence_length, sequence_length
        )
        == 0,
        float("-inf"),
    )
    qk = F.softmax(qk / math.sqrt(embedding_dimension), dim=2)

    V = torch.nn.Linear(embedding_dimension, head_dimension).to("cpu", dtype=x.dtype)
    v = V(ln_x)

    return torch.einsum("bst, bth -> bsh", qk, v)


head_1_output = attend_single_head(ln_x)
head_2_output = attend_single_head(ln_x)

# Concatenate head outputs & do linear projection
ln_x = torch.concatenate((head_1_output, head_2_output), dim=-1)
attention_projection = torch.nn.Linear(embedding_dimension, embedding_dimension).to(
    "cpu", dtype=ln_x.dtype
)
ln_x = attention_projection(ln_x)

# add residual connection
x = torch.add(x, ln_x)

# Layer Norm
ln = torch.nn.LayerNorm(embedding_dimension).to("cpu", dtype=x.dtype)
ln_x = ln(x)

# MLP layer
projected_dimension = (
    embedding_dimension * 2
)  # Can be anything since we will down project later anyway
up_projection = torch.nn.Linear(embedding_dimension, projected_dimension).to(
    "cpu", dtype=x.dtype
)
gelu = torch.nn.GELU().to("cpu", dtype=x.dtype)
down_projection = torch.nn.Linear(projected_dimension, embedding_dimension).to(
    "cpu", dtype=x.dtype
)
ln_x = up_projection(ln_x)
ln_x = gelu(ln_x)
ln_x = down_projection(ln_x)

# Add residual connection
x = torch.add(x, ln_x)

# Output embeddings
output_embeddings = torch.nn.Linear(embedding_dimension, tokenizer.vocab_size).to(
    "cpu", dtype=x.dtype
)
x = output_embeddings(x)
x = F.softmax(x, dim=-1)

x = torch.argmax(x, dim=-1)

print(tokenizer.decode(x[0]))
