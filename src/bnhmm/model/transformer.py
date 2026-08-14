"""Causal Bayesian attention used by the market-state sequence encoder."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .bayesian import BayesianLinear, DeterministicLinear


class BayesianMultiheadAttention(nn.Module):
    def __init__(
        self, d_model: int, heads: int, dropout: float, *, bayesian: bool = True
    ) -> None:
        super().__init__()
        if d_model % heads:
            raise ValueError("d_model must be divisible by heads")
        self.d_model = d_model
        self.heads = heads
        self.head_dim = d_model // heads
        linear = BayesianLinear if bayesian else DeterministicLinear
        self.query = linear(d_model, d_model)
        self.key = linear(d_model, d_model)
        self.value = linear(d_model, d_model)
        self.output = linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        values: Tensor,
        *,
        valid_mask: Tensor,
        sample: bool,
    ) -> Tensor:
        batch, sequence, _ = values.shape

        def split_heads(tensor: Tensor) -> Tensor:
            return tensor.view(batch, sequence, self.heads, self.head_dim).transpose(1, 2)

        query = split_heads(self.query(values, sample=sample))
        key = split_heads(self.key(values, sample=sample))
        value = split_heads(self.value(values, sample=sample))
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)

        causal = torch.triu(
            torch.ones(sequence, sequence, device=values.device, dtype=torch.bool),
            diagonal=1,
        )
        scores = scores.masked_fill(causal, torch.finfo(scores.dtype).min)
        scores = scores.masked_fill(
            ~valid_mask[:, None, None, :], torch.finfo(scores.dtype).min
        )
        attention = self.dropout(F.softmax(scores, dim=-1))
        attended = torch.matmul(attention, value)
        attended = attended.transpose(1, 2).contiguous().view(batch, sequence, self.d_model)
        output = self.output(attended, sample=sample)
        return output * valid_mask.unsqueeze(-1)


class BayesianTransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        heads: int,
        feedforward_dim: int,
        dropout: float,
        *,
        bayesian: bool = True,
    ) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(d_model)
        self.attention = BayesianMultiheadAttention(
            d_model, heads, dropout, bayesian=bayesian
        )
        self.feedforward_norm = nn.LayerNorm(d_model)
        linear = BayesianLinear if bayesian else DeterministicLinear
        self.feedforward_in = linear(d_model, feedforward_dim)
        self.feedforward_out = linear(feedforward_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: Tensor, *, valid_mask: Tensor, sample: bool) -> Tensor:
        attended = self.attention(
            self.attention_norm(values), valid_mask=valid_mask, sample=sample
        )
        values = values + self.dropout(attended)
        hidden = self.feedforward_in(self.feedforward_norm(values), sample=sample)
        hidden = self.feedforward_out(F.gelu(hidden), sample=sample)
        values = values + self.dropout(hidden)
        return values * valid_mask.unsqueeze(-1)

