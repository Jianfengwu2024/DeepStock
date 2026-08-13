"""Offline training loop for the Bayesian profile Transformer."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import Tensor
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset

from .data.dataset import ProfileSequence
from .model import BNHMMNetwork, evidential_classification_loss


@dataclass(frozen=True, slots=True)
class ModelConfig:
    max_profile_bins: int = 512
    profile_channels: int = 3
    state_feature_dim: int = 9
    d_model: int = 96
    motif_dim: int = 32
    heads: int = 4
    layers: int = 3
    feedforward_dim: int = 192
    dropout: float = 0.1
    max_sequence_length: int = 128

    def build(self) -> BNHMMNetwork:
        return BNHMMNetwork(**asdict(self))


@dataclass(frozen=True, slots=True)
class TrainerConfig:
    epochs: int = 10
    batch_size: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    kl_weight: float = 1e-3
    kl_anneal_epochs: int = 10
    brier_weight: float = 0.25
    latent_balance_weight: float = 0.05
    gradient_clip: float = 1.0
    seed: int = 7

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("epochs and batch_size must be positive")
        if self.learning_rate <= 0 or self.gradient_clip <= 0:
            raise ValueError("learning_rate and gradient_clip must be positive")
        if self.kl_weight < 0 or self.kl_anneal_epochs < 1:
            raise ValueError("KL weight must be non-negative and annealing must be positive")


@dataclass(frozen=True, slots=True)
class TrainingResult:
    history: tuple[dict[str, float], ...]
    best_epoch: int
    best_validation_loss: float


class _SequenceDataset(Dataset[ProfileSequence]):
    def __init__(self, sequences: Sequence[ProfileSequence]) -> None:
        self.sequences = list(sequences)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> ProfileSequence:
        return self.sequences[index]


def collate_profile_sequences(items: Sequence[ProfileSequence]) -> dict[str, Tensor]:
    if not items:
        raise ValueError("cannot collate an empty batch")
    batch = len(items)
    sequence = max(len(item) for item in items)
    channels = items[0].profiles[0].shape[-1]
    bins = max(profile.shape[0] for item in items for profile in item.profiles)
    feature_dim = items[0].state_features.shape[-1]
    profiles = torch.zeros(batch, sequence, bins, channels, dtype=torch.float32)
    bin_coordinates = torch.zeros(batch, sequence, bins, dtype=torch.float32)
    profile_mask = torch.zeros(batch, sequence, bins, dtype=torch.bool)
    state_features = torch.zeros(batch, sequence, feature_dim, dtype=torch.float32)
    timeframe_ids = torch.zeros(batch, sequence, dtype=torch.long)
    targets = torch.full((batch, sequence), 2, dtype=torch.long)
    valid_mask = torch.zeros(batch, sequence, dtype=torch.bool)
    for index, item in enumerate(items):
        if len(item.profiles) != len(item) or len(item.bin_coordinates) != len(item):
            raise ValueError("profile arrays and coordinates must match sequence length")
        if item.state_features.shape[-1] != feature_dim:
            raise ValueError("all state features in a batch must use the same dimension")
        length = len(item)
        for state_index, (profile, coordinates) in enumerate(
            zip(item.profiles, item.bin_coordinates, strict=True)
        ):
            if profile.ndim != 2 or profile.shape[-1] != channels:
                raise ValueError("all profiles must use the same channel count")
            if coordinates.shape != (len(profile),):
                raise ValueError("profile coordinates must align with profile bins")
            profile_bins = len(profile)
            profiles[index, state_index, :profile_bins] = torch.from_numpy(profile)
            bin_coordinates[index, state_index, :profile_bins] = torch.from_numpy(
                coordinates
            )
            profile_mask[index, state_index, :profile_bins] = True
        state_features[index, :length] = torch.from_numpy(item.state_features)
        timeframe_ids[index, :length] = torch.from_numpy(item.timeframe_ids)
        targets[index, :length] = torch.from_numpy(item.targets)
        valid_mask[index, :length] = True
    return {
        "profiles": profiles,
        "bin_coordinates": bin_coordinates,
        "profile_mask": profile_mask,
        "state_features": state_features,
        "timeframe_ids": timeframe_ids,
        "targets": targets,
        "valid_mask": valid_mask,
    }


class BNHMMTrainer:
    def __init__(
        self,
        model: BNHMMNetwork,
        config: TrainerConfig | None = None,
        *,
        device: str | torch.device = "cpu",
    ) -> None:
        self.model = model
        self.config = config or TrainerConfig()
        self.device = torch.device(device)
        self.model.to(self.device)

    def fit(
        self,
        train_sequences: Sequence[ProfileSequence],
        validation_sequences: Sequence[ProfileSequence],
    ) -> TrainingResult:
        if not train_sequences or not validation_sequences:
            raise ValueError("training and validation sequences cannot be empty")
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)
        generator = torch.Generator().manual_seed(self.config.seed)
        train_loader = DataLoader(
            _SequenceDataset(train_sequences),
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=collate_profile_sequences,
            generator=generator,
        )
        validation_loader = DataLoader(
            _SequenceDataset(validation_sequences),
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=collate_profile_sequences,
        )
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        history: list[dict[str, float]] = []
        best_loss = float("inf")
        best_epoch = 0
        best_state: dict[str, Tensor] | None = None

        for epoch in range(1, self.config.epochs + 1):
            kl_scale = min(epoch / self.config.kl_anneal_epochs, 1.0)
            train_metrics = self._run_epoch(train_loader, optimizer, kl_scale=kl_scale)
            validation_metrics = self._run_epoch(validation_loader, None, kl_scale=0.0)
            row = {"epoch": float(epoch)}
            row.update({f"train_{key}": value for key, value in train_metrics.items()})
            row.update(
                {f"validation_{key}": value for key, value in validation_metrics.items()}
            )
            history.append(row)
            if validation_metrics["predictive_loss"] < best_loss:
                best_loss = validation_metrics["predictive_loss"]
                best_epoch = epoch
                best_state = copy.deepcopy(self.model.state_dict())

        if best_state is not None:
            self.model.load_state_dict(best_state)
        return TrainingResult(tuple(history), best_epoch, best_loss)

    def fit_fixed(self, train_sequences: Sequence[ProfileSequence]) -> TrainingResult:
        """Train for an already selected epoch count without touching outer test data."""
        if not train_sequences:
            raise ValueError("training sequences cannot be empty")
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)
        loader = DataLoader(
            _SequenceDataset(train_sequences),
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=collate_profile_sequences,
            generator=torch.Generator().manual_seed(self.config.seed),
        )
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        history: list[dict[str, float]] = []
        for epoch in range(1, self.config.epochs + 1):
            kl_scale = min(epoch / self.config.kl_anneal_epochs, 1.0)
            metrics = self._run_epoch(loader, optimizer, kl_scale=kl_scale)
            row = {"epoch": float(epoch)}
            row.update({f"train_{key}": value for key, value in metrics.items()})
            history.append(row)
        return TrainingResult(tuple(history), self.config.epochs, float("nan"))

    def _run_epoch(
        self,
        loader: DataLoader[dict[str, Tensor]],
        optimizer: torch.optim.Optimizer | None,
        *,
        kl_scale: float,
    ) -> dict[str, float]:
        training = optimizer is not None
        self.model.train(training)
        totals = {
            "loss": 0.0,
            "predictive_loss": 0.0,
            "nll": 0.0,
            "brier": 0.0,
            "kl": 0.0,
            "latent_balance": 0.0,
            "accuracy": 0.0,
        }
        token_count = 0
        for cpu_batch in loader:
            batch = {key: value.to(self.device) for key, value in cpu_batch.items()}
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(training):
                output = self.model(
                    batch["profiles"],
                    batch["timeframe_ids"],
                    batch["state_features"],
                    batch["bin_coordinates"],
                    batch["profile_mask"],
                    valid_mask=batch["valid_mask"],
                    sample=training,
                )
                loss = evidential_classification_loss(
                    output,
                    batch["targets"],
                    valid_mask=batch["valid_mask"],
                    kl_weight=self.config.kl_weight * kl_scale,
                    brier_weight=self.config.brier_weight,
                    latent_balance_weight=self.config.latent_balance_weight,
                )
                if optimizer is not None:
                    loss.total.backward()
                    clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                    optimizer.step()

            tokens = int(batch["valid_mask"].sum().item())
            probability = output.alpha / output.alpha.sum(dim=-1, keepdim=True)
            correct = (
                (probability.argmax(dim=-1) == batch["targets"])
                & batch["valid_mask"]
            ).sum()
            totals["loss"] += float(loss.total.detach()) * tokens
            predictive_loss = (
                loss.evidential_nll + self.config.brier_weight * loss.brier
            )
            totals["predictive_loss"] += float(predictive_loss.detach()) * tokens
            totals["nll"] += float(loss.evidential_nll.detach()) * tokens
            totals["brier"] += float(loss.brier.detach()) * tokens
            totals["kl"] += float(loss.kl.detach()) * tokens
            totals["latent_balance"] += float(loss.latent_balance.detach()) * tokens
            totals["accuracy"] += float(correct.detach())
            token_count += tokens
        if not token_count:
            raise ValueError("data loader produced no valid tokens")
        return {key: value / token_count for key, value in totals.items()}


def save_checkpoint(
    path: str | Path,
    model: BNHMMNetwork,
    model_config: ModelConfig,
    trainer_config: TrainerConfig,
    result: TrainingResult,
    metadata: dict[str, Any],
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 2,
            "model_state": model.state_dict(),
            "model_config": asdict(model_config),
            "trainer_config": asdict(trainer_config),
            "history": list(result.history),
            "best_epoch": result.best_epoch,
            "best_validation_loss": result.best_validation_loss,
            "metadata": metadata,
        },
        destination,
    )
