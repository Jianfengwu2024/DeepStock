"""Training layer for StockTransformer; consumes tokenized states only."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from .model import StockTransformer
from .tokenization import TOKEN_FEATURE_DIM, TokenizedState


@dataclass(frozen=True, slots=True)
class StockWindow:
    symbol: str
    states: tuple[TokenizedState, ...]
    signal_time: np.datetime64
    execution_time: np.datetime64
    target_time: np.datetime64
    target: int
    forward_return: float
    standardized_return: float
    return_location: float
    return_scale: float


@dataclass(frozen=True, slots=True)
class StockTrainerConfig:
    epochs: int = 10
    batch_size: int = 16
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    classification_weight: float = 1.0
    brier_weight: float = 0.25
    conservation_weight: float = 0.5
    gradient_clip: float = 1.0
    seed: int = 7

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.batch_size < 1 or self.learning_rate <= 0:
            raise ValueError("StockTransformer training parameters are invalid")
        if min(
            self.classification_weight,
            self.brier_weight,
            self.conservation_weight,
        ) < 0:
            raise ValueError("training objective weights cannot be negative")


@dataclass(frozen=True, slots=True)
class StockTrainingResult:
    history: tuple[dict[str, float], ...]
    best_epoch: int
    best_validation_loss: float


def collate_stock_windows(samples: Sequence[StockWindow]) -> dict[str, Tensor]:
    if not samples:
        raise ValueError("cannot collate an empty StockTransformer batch")
    days = max(len(item.states) for item in samples)
    tokens = max(len(state.tokens) for item in samples for state in item.states)
    features = torch.zeros((len(samples), days, tokens, TOKEN_FEATURE_DIM))
    mask = torch.zeros((len(samples), days, tokens), dtype=torch.bool)
    for batch_index, sample in enumerate(samples):
        day_offset = days - len(sample.states)
        for day_index, state in enumerate(sample.states, start=day_offset):
            values = torch.as_tensor(state.features, dtype=torch.float32)
            features[batch_index, day_index, : len(values)] = values
            mask[batch_index, day_index, : len(values)] = True
    return {
        "token_features": features,
        "token_mask": mask,
        "targets": torch.tensor([item.target for item in samples], dtype=torch.long),
        "returns": torch.tensor(
            [item.standardized_return for item in samples], dtype=torch.float32
        ),
    }


class StockTransformerTrainer:
    def __init__(
        self,
        model: StockTransformer,
        config: StockTrainerConfig | None = None,
        *,
        device: str | torch.device = "cpu",
    ) -> None:
        self.model = model
        self.config = config or StockTrainerConfig()
        self.device = torch.device(device)
        self.model.to(self.device)

    def fit(
        self,
        train_samples: Sequence[StockWindow],
        validation_samples: Sequence[StockWindow],
    ) -> StockTrainingResult:
        if not train_samples or not validation_samples:
            raise ValueError("training and validation samples cannot be empty")
        self._seed()
        self.model.set_class_prior(
            torch.tensor([item.target for item in train_samples], dtype=torch.long)
        )
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        train_loader = self._loader(train_samples, shuffle=True)
        validation_loader = self._loader(validation_samples, shuffle=False)
        history: list[dict[str, float]] = []
        best_loss = float("inf")
        best_epoch = 1
        best_state: dict[str, Tensor] | None = None
        for epoch in range(1, self.config.epochs + 1):
            train = self._run_epoch(train_loader, optimizer)
            validation = self._run_epoch(validation_loader, None)
            row = {"epoch": float(epoch)}
            row.update({f"train_{key}": value for key, value in train.items()})
            row.update(
                {f"validation_{key}": value for key, value in validation.items()}
            )
            history.append(row)
            if validation["loss"] < best_loss:
                best_loss = validation["loss"]
                best_epoch = epoch
                best_state = copy.deepcopy(self.model.state_dict())
        assert best_state is not None
        self.model.load_state_dict(best_state)
        return StockTrainingResult(tuple(history), best_epoch, best_loss)

    def fit_fixed(self, samples: Sequence[StockWindow]) -> StockTrainingResult:
        if not samples:
            raise ValueError("fixed training samples cannot be empty")
        self._seed()
        self.model.set_class_prior(
            torch.tensor([item.target for item in samples], dtype=torch.long)
        )
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        loader = self._loader(samples, shuffle=True)
        history: list[dict[str, float]] = []
        for epoch in range(1, self.config.epochs + 1):
            metrics = self._run_epoch(loader, optimizer)
            history.append(
                {"epoch": float(epoch), **{f"train_{key}": value for key, value in metrics.items()}}
            )
        return StockTrainingResult(tuple(history), self.config.epochs, float("nan"))

    def _loader(self, samples: Sequence[StockWindow], *, shuffle: bool) -> DataLoader:
        generator = torch.Generator().manual_seed(self.config.seed)
        return DataLoader(
            list(samples),
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            generator=generator,
            collate_fn=collate_stock_windows,
        )

    def _run_epoch(
        self, loader: DataLoader, optimizer: torch.optim.Optimizer | None
    ) -> dict[str, float]:
        training = optimizer is not None
        self.model.train(training)
        totals = {key: 0.0 for key in ("loss", "classification", "return_nll", "brier", "conservation", "accuracy")}
        count = 0
        for cpu_batch in loader:
            batch = {key: value.to(self.device) for key, value in cpu_batch.items()}
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(training):
                output = self.model(batch["token_features"], batch["token_mask"])
                classification = F.cross_entropy(output.regime_logits, batch["targets"])
                variance = output.return_log_variance.exp().clamp_min(1e-6)
                return_nll = 0.5 * (
                    output.return_log_variance
                    + (batch["returns"] - output.return_mean).square() / variance
                ).mean()
                probability = torch.softmax(output.regime_logits, dim=-1)
                one_hot = F.one_hot(batch["targets"], probability.shape[-1]).to(probability.dtype)
                brier = (probability - one_hot).square().sum(dim=-1).mean()
                delta_target = batch["token_features"][..., 1]
                token_error = (
                    (output.conservation_prediction - delta_target).square()
                    * batch["token_mask"].to(delta_target.dtype)
                ).sum() / batch["token_mask"].sum().clamp_min(1)
                predicted_mass = (
                    output.conservation_prediction
                    * batch["token_mask"].to(delta_target.dtype)
                ).sum(dim=-1)
                day_mask = batch["token_mask"].any(dim=-1)
                mass_error = (
                    predicted_mass.square() * day_mask.to(delta_target.dtype)
                ).sum() / day_mask.sum().clamp_min(1)
                conservation = token_error + mass_error
                loss = (
                    return_nll
                    + self.config.classification_weight * classification
                    + self.config.brier_weight * brier
                    + self.config.conservation_weight * conservation
                )
                if optimizer is not None:
                    loss.backward()
                    clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                    optimizer.step()
            size = len(batch["targets"])
            totals["loss"] += float(loss.detach()) * size
            totals["classification"] += float(classification.detach()) * size
            totals["return_nll"] += float(return_nll.detach()) * size
            totals["brier"] += float(brier.detach()) * size
            totals["conservation"] += float(conservation.detach()) * size
            totals["accuracy"] += float(
                (probability.argmax(dim=-1) == batch["targets"]).sum().detach()
            )
            count += size
        return {key: value / max(count, 1) for key, value in totals.items()}

    def _seed(self) -> None:
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)
