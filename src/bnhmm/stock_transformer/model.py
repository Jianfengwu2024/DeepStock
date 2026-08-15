"""Transformer that consumes conservation tokens without profile construction logic."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from ..config import REGIME_COUNT, REGIME_SCORES
from .tokenization import TOKEN_FEATURE_DIM


@dataclass(frozen=True, slots=True)
class StockTransformerConfig:
    feature_dim: int = TOKEN_FEATURE_DIM
    d_model: int = 64
    heads: int = 4
    layers: int = 3
    feedforward_dim: int = 128
    dropout: float = 0.1
    maximum_days: int = 10
    maximum_tokens: int = 16

    def __post_init__(self) -> None:
        if self.feature_dim != TOKEN_FEATURE_DIM:
            raise ValueError("StockTransformer requires the conservation token schema")
        if self.d_model < 8 or self.d_model % self.heads:
            raise ValueError("model width must be divisible by attention heads")
        if self.layers < 1 or self.feedforward_dim < self.d_model:
            raise ValueError("transformer depth and feedforward width are invalid")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.maximum_days < 2 or self.maximum_tokens < 2:
            raise ValueError("StockTransformer requires multiple days and tokens")

    def build(self, *, seed: int = 7) -> "StockTransformer":
        torch.manual_seed(seed)
        return StockTransformer(self)


@dataclass(slots=True)
class StockTransformerOutput:
    regime_logits: Tensor
    return_mean: Tensor
    return_log_variance: Tensor
    conservation_prediction: Tensor


@dataclass(slots=True)
class StockTransformerPosterior:
    probability_mean: Tensor
    aleatoric_variance: Tensor
    epistemic_variance: Tensor
    expected_score: Tensor
    expected_return: Tensor
    return_variance: Tensor
    aleatoric_return_variance: Tensor
    epistemic_return_variance: Tensor


class StockTransformer(nn.Module):
    """Encode variable conservation words with day, position, and sentence markers."""

    def __init__(self, config: StockTransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.input_projection = nn.Sequential(
            nn.Linear(config.feature_dim, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.GELU(),
        )
        self.day_embedding = nn.Embedding(config.maximum_days, config.d_model)
        self.token_embedding = nn.Embedding(config.maximum_tokens, config.d_model)
        self.sentence_embedding = nn.Embedding(2, config.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.heads,
            dim_feedforward=config.feedforward_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=config.layers, norm=nn.LayerNorm(config.d_model)
        )
        self.pool_score = nn.Linear(config.d_model, 1)
        self.regime_head = nn.Linear(config.d_model, REGIME_COUNT)
        self.return_head = nn.Linear(config.d_model, 2)
        self.conservation_head = nn.Linear(config.d_model, 1)
        self.register_buffer(
            "class_prior", torch.full((REGIME_COUNT,), 1.0 / REGIME_COUNT)
        )
        self.register_buffer("regime_scores", torch.tensor(REGIME_SCORES))

    def set_class_prior(self, targets: Tensor) -> None:
        counts = torch.bincount(
            targets.detach().to(torch.long).cpu(), minlength=REGIME_COUNT
        ).to(torch.float32)
        prior = (counts + 1.0) / (counts.sum() + REGIME_COUNT)
        self.class_prior.copy_(prior.to(self.class_prior.device))

    def forward(self, token_features: Tensor, token_mask: Tensor) -> StockTransformerOutput:
        if token_features.ndim != 4:
            raise ValueError("token features must have batch, day, token, feature axes")
        batch, days, tokens, features = token_features.shape
        if features != self.config.feature_dim:
            raise ValueError("token feature dimension does not match the model")
        if days > self.config.maximum_days or tokens > self.config.maximum_tokens:
            raise ValueError("token batch exceeds configured position embeddings")
        if token_mask.shape != (batch, days, tokens):
            raise ValueError("token mask shape does not match token features")
        # Mass change is an auxiliary conservation target, not an input shortcut.
        visible = token_features.clone()
        visible[..., 1] = 0.0
        encoded = self.input_projection(visible)
        day_ids = torch.arange(days, device=encoded.device).view(1, days, 1)
        token_ids = torch.arange(tokens, device=encoded.device).view(1, 1, tokens)
        sentence_start = (token_features[..., -1] > 0.5).to(torch.long)
        encoded = (
            encoded
            + self.day_embedding(day_ids)
            + self.token_embedding(token_ids)
            + self.sentence_embedding(sentence_start)
        )
        flat = encoded.reshape(batch, days * tokens, self.config.d_model)
        flat_mask = token_mask.reshape(batch, days * tokens)
        if torch.any(~flat_mask.any(dim=1)):
            raise ValueError("each sample must contain at least one token")
        contextual = self.encoder(flat, src_key_padding_mask=~flat_mask)
        score = self.pool_score(contextual).squeeze(-1)
        score = score.masked_fill(~flat_mask, -torch.inf)
        weight = torch.softmax(score, dim=-1)
        pooled = torch.sum(contextual * weight.unsqueeze(-1), dim=1)
        return_parameters = self.return_head(pooled)
        return StockTransformerOutput(
            regime_logits=self.class_prior.clamp_min(1e-8).log()
            + self.regime_head(pooled),
            return_mean=4.0 * torch.tanh(return_parameters[:, 0]),
            return_log_variance=return_parameters[:, 1].clamp(-6.0, 4.0),
            conservation_prediction=self.conservation_head(contextual)
            .squeeze(-1)
            .reshape(batch, days, tokens),
        )

    @torch.no_grad()
    def predict_posterior(
        self, token_features: Tensor, token_mask: Tensor, *, mc_samples: int = 16
    ) -> StockTransformerPosterior:
        if mc_samples < 2:
            raise ValueError("posterior estimation requires at least two samples")
        was_training = self.training
        self.train(True)
        outputs = [self(token_features, token_mask) for _ in range(mc_samples)]
        self.train(was_training)
        probability = torch.stack(
            [torch.softmax(item.regime_logits, dim=-1) for item in outputs]
        )
        return_mean = torch.stack([item.return_mean for item in outputs])
        return_variance = torch.stack(
            [item.return_log_variance.exp() for item in outputs]
        )
        probability_mean = probability.mean(dim=0)
        expected_return = return_mean.mean(dim=0)
        aleatoric_return = return_variance.mean(dim=0)
        epistemic_return = return_mean.var(dim=0, unbiased=False)
        return StockTransformerPosterior(
            probability_mean=probability_mean,
            aleatoric_variance=(probability * (1.0 - probability)).mean(dim=0),
            epistemic_variance=probability.var(dim=0, unbiased=False),
            expected_score=(probability_mean * self.regime_scores).sum(dim=-1),
            expected_return=expected_return,
            return_variance=aleatoric_return + epistemic_return,
            aleatoric_return_variance=aleatoric_return,
            epistemic_return_variance=epistemic_return,
        )
