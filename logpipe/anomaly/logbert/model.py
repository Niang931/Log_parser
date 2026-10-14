"""LogBERT encoder and masked log-key head.

Adapted from ``bert_pytorch/model`` in https://github.com/HelenGuohx/logbert
(MIT, Copyright (c) 2023 Helennnnn), itself derived from
https://github.com/codertimo/BERT-pytorch (Apache-2.0).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass
class LogBERTConfig:
    vocab_size: int
    hidden: int = 64
    layers: int = 2
    attn_heads: int = 4
    max_len: int = 128
    dropout: float = 0.1

    def to_dict(self) -> dict:
        return asdict(self)


class PositionalEmbedding(nn.Module):
    pe: torch.Tensor

    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        # Adapted from the original attention paper
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # Register the tensor inside module buffer
        # and add another layer for broadcasting
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # return the pe broadcasted to be the exact length of input sequence
        return self.pe[:, : x.size(1)]


class BERTEmbedding(nn.Module):
    """Token embedding plus fixed sinusoidal position embedding."""

    def __init__(self, vocab_size: int, embed_size: int, max_len: int, dropout: float):
        super().__init__()
        self.token = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.position = PositionalEmbedding(embed_size, max_len)
        self.dropout = nn.Dropout(dropout)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.token(sequence) + self.position(sequence))


class MultiHeadedAttention(nn.Module):
    def __init__(self, heads: int, d_model: int, dropout: float):
        super().__init__()
        assert d_model % heads == 0
        self.d_k = d_model // heads
        self.h = heads
        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch = x.size(0)
        query, key, value = [
            layer(x).view(batch, -1, self.h, self.d_k).transpose(1, 2)
            for layer in self.linear_layers
        ]
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        scores = scores.masked_fill(~mask, -1e9)
        attn = self.dropout(torch.softmax(scores, dim=-1))
        out = torch.matmul(attn, value).transpose(1, 2).contiguous().view(batch, -1, self.h * self.d_k)
        return self.output_linear(out)


class TransformerBlock(nn.Module):
    """Pre-norm residual block: x + attn(norm(x)), then x + ffn(norm(x))."""

    def __init__(self, hidden: int, heads: int, ff_hidden: int, dropout: float):
        super().__init__()
        self.attention = MultiHeadedAttention(heads, hidden, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden, ff_hidden),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(ff_hidden, hidden),
        )
        self.norm1 = nn.LayerNorm(hidden, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.attention(self.norm1(x), mask))
        x = x + self.dropout(self.feed_forward(self.norm2(x)))
        return self.dropout(x)


class LogBERT(nn.Module):
    """BERT encoder plus masked log-key head (``BERTLog`` in the original).

    ``forward`` returns the log-softmax over the vocabulary for every
    position, and the ``[SOS]`` (position 0) representation that the
    hypersphere objective pulls towards the centre of normal sequences.
    """

    def __init__(self, config: LogBERTConfig):
        super().__init__()
        self.config = config
        self.embedding = BERTEmbedding(config.vocab_size, config.hidden, config.max_len, config.dropout)
        # TODO: should i do hidden * 2 or * 4 like with original BERT, eval later
        self.blocks = nn.ModuleList(
            TransformerBlock(config.hidden, config.attn_heads, config.hidden * 2, config.dropout)
            for _ in range(config.layers)
        )
        self.mask_lm = nn.Sequential(nn.Linear(config.hidden, config.vocab_size), nn.LogSoftmax(dim=-1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mask = (x > 0)[:, None, None, :]
        hidden = self.embedding(x)
        for block in self.blocks:
            hidden = block(hidden, mask)
        return self.mask_lm(hidden), hidden[:, 0]
