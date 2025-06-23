import mlx.core as mx
import mlx.nn as nn
import math

class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128, out_dim=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.projection = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def __call__(self, x):  # x: [batch, seq_len]
        x = self.embedding(x)                # [batch, seq_len, embed_dim]
        x = mx.mean(x, axis=1)              # mean-pooling over tokens
        x = self.projection(x)              # project to output dim
        norm = mx.sqrt(mx.sum(x ** 2, axis=1, keepdims=True) + 1e-8)
        return x / norm


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = mx.zeros((max_len, d_model))
        position = mx.arange(0, max_len).reshape(-1, 1)
        div_term = mx.exp(mx.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = mx.sin(position * div_term)
        pe[:, 1::2] = mx.cos(position * div_term)
        self.pe = pe

    def __call__(self, x):
        return x + self.pe[:x.shape[1]]


class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiHeadAttention(d_model, num_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )

    def __call__(self, x, mask=None):
        # Self-attention with residual
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x), mask=mask)
        # Feed-forward with residual
        x = x + self.ffn(self.norm2(x))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model=768, num_heads=12, num_layers=12, d_ff=3072, max_len=512):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model, max_len)
        self.layers = [TransformerEncoderBlock(d_model, num_heads, d_ff) for _ in range(num_layers)]
        self.norm = nn.LayerNorm(d_model)

    def __call__(self, input_ids, mask=None):
        x = self.token_emb(input_ids)
        x = self.pos_emb(x)
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

