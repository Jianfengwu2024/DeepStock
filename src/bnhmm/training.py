"""Offline training loop for the Bayesian profile Transformer."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from math import isqrt
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import Tensor
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset

from .config import PROFILE_CHANNELS, REGIME_COUNT, REGIME_NAMES, REGIME_PROTOCOL
from .data.dataset import ProfileSequence
from .model import (
    BNHMMNetwork,
    return_distribution_loss,
)


@dataclass(frozen=True, slots=True)
class ModelConfig:
    max_profile_bins: int = 512
    profile_channels: int = PROFILE_CHANNELS
    state_feature_dim: int = 9
    latent_states: int = 20
    output_states: int = REGIME_COUNT
    feature_mode: str = "price-velocity"
    auxiliary_gate_floor: float = 0.25
    auxiliary_gate_initial: float = 0.75
    bayesian_backbone: bool = False
    transition_rank: int = 4
    strong_bear_threshold: float = -0.75
    direction_threshold: float = 0.0
    strong_bull_threshold: float = 0.75
    d_model: int = 48
    motif_dim: int = 16
    heads: int = 4
    layers: int = 2
    feedforward_dim: int = 96
    dropout: float = 0.1
    max_sequence_length: int = 5

    def build(self, *, seed: int | None = None) -> BNHMMNetwork:
        if seed is None:
            return BNHMMNetwork(**asdict(self))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            return BNHMMNetwork(**asdict(self))


@dataclass(frozen=True, slots=True)
class TrainerConfig:
    epochs: int = 10
    batch_size: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    kl_weight: float = 1e-3
    kl_anneal_epochs: int = 10
    classification_weight: float = 1.0
    brier_weight: float = 0.25
    ordinal_weight: float = 0.25
    consistency_weight: float = 0.25
    symmetry_weight: float = 0.2
    pretrain_epochs: int = 10
    auxiliary_warmup_epochs: int = 3
    pretrain_mask_probability: float = 0.15
    pretrain_next_weight: float = 1.0
    pretrain_profile_weight: float = 1.0
    latent_information_weight: float = 0.05
    student_df: float = 4.0
    max_tokens_per_batch: int = 256
    sample_during_training: bool = True
    gradient_clip: float = 1.0
    seed: int = 7

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("epochs and batch_size must be positive")
        if self.learning_rate <= 0 or self.gradient_clip <= 0:
            raise ValueError("learning_rate and gradient_clip must be positive")
        if self.kl_weight < 0 or self.kl_anneal_epochs < 1:
            raise ValueError("KL weight must be non-negative and annealing must be positive")
        objective_weights = (
            self.classification_weight,
            self.brier_weight,
            self.ordinal_weight,
            self.consistency_weight,
            self.symmetry_weight,
            self.latent_information_weight,
        )
        if any(value < 0 for value in objective_weights):
            raise ValueError("objective weights cannot be negative")
        if self.pretrain_epochs < 0:
            raise ValueError("pretraining epochs cannot be negative")
        if self.auxiliary_warmup_epochs < 0:
            raise ValueError("auxiliary warmup epochs cannot be negative")
        if not 0.0 < self.pretrain_mask_probability < 1.0:
            raise ValueError("pretraining mask probability must be between zero and one")
        if self.pretrain_next_weight < 0:
            raise ValueError("next-state pretraining weight cannot be negative")
        if self.pretrain_profile_weight < 0:
            raise ValueError("profile pretraining weight cannot be negative")
        if self.student_df != 4.0:
            raise ValueError("BNHMM currently requires Student-t degrees of freedom = 4")
        if self.max_tokens_per_batch < 1:
            raise ValueError("maximum tokens per batch must be positive")


@dataclass(frozen=True, slots=True)
class TrainingResult:
    history: tuple[dict[str, float], ...]
    best_epoch: int
    best_validation_loss: float
    pretraining_history: tuple[dict[str, float], ...] = ()


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
    targets = torch.zeros(batch, sequence, dtype=torch.long)
    forward_returns = torch.zeros(batch, sequence, dtype=torch.float32)
    standardized_returns = torch.zeros(batch, sequence, dtype=torch.float32)
    return_locations = torch.zeros(batch, sequence, dtype=torch.float32)
    return_scales = torch.ones(batch, sequence, dtype=torch.float32)
    valid_mask = torch.zeros(batch, sequence, dtype=torch.bool)
    target_mask = torch.zeros(batch, sequence, dtype=torch.bool)
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
        forward_returns[index, :length] = torch.from_numpy(
            item.forward_returns.astype(np.float32)
        )
        standardized_returns[index, :length] = torch.from_numpy(
            item.standardized_returns.astype(np.float32)
        )
        return_locations[index, :length] = torch.from_numpy(
            item.return_locations.astype(np.float32)
        )
        return_scales[index, :length] = torch.from_numpy(
            item.return_scales.astype(np.float32)
        )
        valid_mask[index, :length] = True
        target_mask[index, length - 1] = True
    return {
        "profiles": profiles,
        "bin_coordinates": bin_coordinates,
        "profile_mask": profile_mask,
        "state_features": state_features,
        "timeframe_ids": timeframe_ids,
        "targets": targets,
        "forward_returns": forward_returns,
        "standardized_returns": standardized_returns,
        "return_locations": return_locations,
        "return_scales": return_scales,
        "valid_mask": valid_mask,
        "target_mask": target_mask,
    }


def profile_summary_targets(batch: dict[str, Tensor]) -> Tensor:
    """Produce fixed-size mass/location/spread targets from variable profiles."""
    profiles = batch["profiles"]
    mask = batch["profile_mask"].unsqueeze(-1).to(profiles.dtype)
    coordinates = torch.clamp(
        batch["bin_coordinates"] / 64.0, -4.0, 4.0
    ).unsqueeze(-1)
    absolute = profiles.abs() * mask
    denominator = absolute.sum(dim=-2).clamp_min(1e-8)
    mass = torch.clamp((profiles * mask).sum(dim=-2), -4.0, 4.0)
    location = (absolute * coordinates).sum(dim=-2) / denominator
    spread = torch.sqrt(
        (
            absolute
            * (coordinates - location.unsqueeze(-2)).square()
        ).sum(dim=-2)
        / denominator
        + 1e-8
    )
    return torch.cat((mass, location, spread), dim=-1)


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
        self._set_training_prior(train_sequences)
        generator = torch.Generator().manual_seed(self.config.seed)
        pretraining_generator = torch.Generator().manual_seed(self.config.seed + 10_000)
        dataset = _SequenceDataset(train_sequences)
        effective_batch_size = self._effective_batch_size(train_sequences)
        train_loader = DataLoader(
            dataset,
            batch_size=effective_batch_size,
            shuffle=True,
            collate_fn=collate_profile_sequences,
            generator=generator,
        )
        pretraining_loader = DataLoader(
            dataset,
            batch_size=effective_batch_size,
            shuffle=True,
            collate_fn=collate_profile_sequences,
            generator=pretraining_generator,
        )
        validation_loader = DataLoader(
            _SequenceDataset(validation_sequences),
            batch_size=self._effective_batch_size(validation_sequences),
            shuffle=False,
            collate_fn=collate_profile_sequences,
        )
        pretraining_history = self._pretrain(pretraining_loader)
        self._reset_training_seed()
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
            self.model.tokenizer.set_auxiliary_warmup(
                epoch <= self.config.auxiliary_warmup_epochs
            )
            kl_scale = min(epoch / self.config.kl_anneal_epochs, 1.0)
            train_metrics = self._run_epoch(train_loader, optimizer, kl_scale=kl_scale)
            validation_metrics = self._run_epoch(validation_loader, None, kl_scale=0.0)
            row = {"epoch": float(epoch)}
            row.update({f"train_{key}": value for key, value in train_metrics.items()})
            row.update(
                {f"validation_{key}": value for key, value in validation_metrics.items()}
            )
            row.update(self._auxiliary_gate_metrics())
            history.append(row)
            if validation_metrics["predictive_loss"] < best_loss:
                best_loss = validation_metrics["predictive_loss"]
                best_epoch = epoch
                best_state = copy.deepcopy(self.model.state_dict())

        if best_state is not None:
            self.model.load_state_dict(best_state)
        return TrainingResult(
            tuple(history), best_epoch, best_loss, pretraining_history
        )

    def fit_fixed(self, train_sequences: Sequence[ProfileSequence]) -> TrainingResult:
        """Train for an already selected epoch count without touching outer test data."""
        if not train_sequences:
            raise ValueError("training sequences cannot be empty")
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)
        self._set_training_prior(train_sequences)
        dataset = _SequenceDataset(train_sequences)
        effective_batch_size = self._effective_batch_size(train_sequences)
        loader = DataLoader(
            dataset,
            batch_size=effective_batch_size,
            shuffle=True,
            collate_fn=collate_profile_sequences,
            generator=torch.Generator().manual_seed(self.config.seed),
        )
        pretraining_loader = DataLoader(
            dataset,
            batch_size=effective_batch_size,
            shuffle=True,
            collate_fn=collate_profile_sequences,
            generator=torch.Generator().manual_seed(self.config.seed + 10_000),
        )
        pretraining_history = self._pretrain(pretraining_loader)
        self._reset_training_seed()
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        history: list[dict[str, float]] = []
        for epoch in range(1, self.config.epochs + 1):
            self.model.tokenizer.set_auxiliary_warmup(
                epoch <= self.config.auxiliary_warmup_epochs
            )
            kl_scale = min(epoch / self.config.kl_anneal_epochs, 1.0)
            metrics = self._run_epoch(loader, optimizer, kl_scale=kl_scale)
            row = {"epoch": float(epoch)}
            row.update({f"train_{key}": value for key, value in metrics.items()})
            row.update(self._auxiliary_gate_metrics())
            history.append(row)
        return TrainingResult(
            tuple(history),
            self.config.epochs,
            float("nan"),
            pretraining_history,
        )

    def _effective_batch_size(
        self, sequences: Sequence[ProfileSequence]
    ) -> int:
        longest = max(len(item) for item in sequences)
        token_limited = max(1, self.config.max_tokens_per_batch // longest)
        return min(self.config.batch_size, token_limited)

    def _set_training_prior(self, sequences: Sequence[ProfileSequence]) -> None:
        counts = torch.ones(REGIME_COUNT, dtype=torch.float32, device=self.device)
        for sequence in sequences:
            counts[int(sequence.targets[-1])] += 1.0
        self.model.set_class_prior(counts / counts.sum())

    def _pretrain(
        self, loader: DataLoader[dict[str, Tensor]]
    ) -> tuple[dict[str, float], ...]:
        if self.config.pretrain_epochs == 0:
            return ()
        self.model.tokenizer.set_auxiliary_warmup(True)
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        history: list[dict[str, float]] = []
        for epoch in range(1, self.config.pretrain_epochs + 1):
            self.model.train()
            masked_total = 0.0
            next_total = 0.0
            masked_profile_total = 0.0
            next_profile_total = 0.0
            batch_count = 0
            for cpu_batch in loader:
                batch = {key: value.to(self.device) for key, value in cpu_batch.items()}
                masked_states = (
                    torch.rand_like(batch["valid_mask"], dtype=torch.float32)
                    < self.config.pretrain_mask_probability
                ) & batch["valid_mask"]
                if not masked_states.any():
                    first_valid = batch["valid_mask"].nonzero()[0]
                    masked_states[first_valid[0], first_valid[1]] = True

                profile_targets = profile_summary_targets(batch)
                profiles = batch["profiles"].clone()
                state_features = batch["state_features"].clone()
                bin_coordinates = batch["bin_coordinates"].clone()
                profiles[masked_states] = 0.0
                state_features[masked_states] = 0.0
                bin_coordinates[masked_states] = 0.0
                optimizer.zero_grad(set_to_none=True)
                output = self.model(
                    profiles,
                    batch["timeframe_ids"],
                    state_features,
                    bin_coordinates,
                    batch["profile_mask"],
                    valid_mask=batch["valid_mask"],
                    sample=False,
                )
                features = self.model.pretraining_features(output)
                reconstructed = self.model.masked_state_reconstruction(features)
                masked_loss = torch.nn.functional.smooth_l1_loss(
                    reconstructed[masked_states],
                    batch["state_features"][masked_states],
                )
                reconstructed_profile = (
                    self.model.masked_profile_reconstruction(features)
                )
                masked_profile_loss = torch.nn.functional.smooth_l1_loss(
                    reconstructed_profile[masked_states],
                    profile_targets[masked_states],
                )

                pair_mask = (
                    batch["valid_mask"][:, :-1] & batch["valid_mask"][:, 1:]
                )
                predicted_next = self.model.next_state_prediction(features[:, :-1])
                predicted_next_profile = self.model.next_profile_prediction(
                    features[:, :-1]
                )
                if pair_mask.any():
                    next_loss = torch.nn.functional.smooth_l1_loss(
                        predicted_next[pair_mask],
                        batch["state_features"][:, 1:][pair_mask],
                    )
                    next_profile_loss = torch.nn.functional.smooth_l1_loss(
                        predicted_next_profile[pair_mask],
                        profile_targets[:, 1:][pair_mask],
                    )
                else:
                    next_loss = predicted_next.sum() * 0.0
                    next_profile_loss = predicted_next_profile.sum() * 0.0
                total = (
                    masked_loss
                    + self.config.pretrain_next_weight * next_loss
                    + self.config.pretrain_profile_weight
                    * (masked_profile_loss + next_profile_loss)
                )
                total.backward()
                clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                optimizer.step()
                masked_total += float(masked_loss.detach())
                next_total += float(next_loss.detach())
                masked_profile_total += float(masked_profile_loss.detach())
                next_profile_total += float(next_profile_loss.detach())
                batch_count += 1
            history.append(
                {
                    "epoch": float(epoch),
                    "masked_state_loss": masked_total / batch_count,
                    "next_state_loss": next_total / batch_count,
                    "masked_profile_loss": masked_profile_total / batch_count,
                    "next_profile_loss": next_profile_total / batch_count,
                }
            )
        self.model.tokenizer.set_auxiliary_warmup(False)
        return tuple(history)

    def _auxiliary_gate_metrics(self) -> dict[str, float]:
        gates = self.model.tokenizer.effective_auxiliary_gates().detach().cpu()
        return {
            "auxiliary_warmup": float(
                bool(self.model.tokenizer.auxiliary_force_full)
            ),
            "relative_volume_gate": float(gates[0]),
            "joint_velocity_gate": float(gates[1]),
        }

    def _reset_training_seed(self) -> None:
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)

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
            "return_nll": 0.0,
            "return_mae": 0.0,
            "classification": 0.0,
            "brier": 0.0,
            "ordinal": 0.0,
            "consistency": 0.0,
            "symmetry": 0.0,
            "kl": 0.0,
            "latent_information": 0.0,
            "accuracy": 0.0,
        }
        token_count = 0
        for cpu_batch in loader:
            batch = {key: value.to(self.device) for key, value in cpu_batch.items()}
            original_batch_size = len(batch["profiles"])
            symmetry_active = training and self.config.symmetry_weight > 0.0
            if symmetry_active:
                batch = self._augment_z2_batch(batch)
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
                    sample=training and self.config.sample_during_training,
                )
                loss = return_distribution_loss(
                    output,
                    batch["standardized_returns"],
                    batch["targets"],
                    valid_mask=batch["target_mask"],
                    kl_weight=self.config.kl_weight * kl_scale,
                    classification_weight=self.config.classification_weight,
                    brier_weight=self.config.brier_weight,
                    ordinal_weight=self.config.ordinal_weight,
                    consistency_weight=self.config.consistency_weight,
                    latent_information_weight=self.config.latent_information_weight,
                    class_weights=(
                        None if symmetry_active else self.model.class_weights
                    ),
                    student_df=self.config.student_df,
                    strong_bear_threshold=self.model.strong_bear_threshold,
                    direction_threshold=self.model.direction_threshold,
                    strong_bull_threshold=self.model.strong_bull_threshold,
                )
                symmetry = (
                    self._z2_symmetry_loss(output, batch, original_batch_size)
                    if symmetry_active
                    else output.return_mean.new_zeros(())
                )
                total_loss = loss.total + self.config.symmetry_weight * symmetry
                if optimizer is not None:
                    total_loss.backward()
                    clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                    optimizer.step()

            tokens = int(batch["target_mask"].sum().item())
            probability = torch.softmax(output.regime_logits, dim=-1)
            correct = (
                (probability.argmax(dim=-1) == batch["targets"])
                & batch["target_mask"]
            ).sum()
            totals["loss"] += float(total_loss.detach()) * tokens
            predictive_loss = (
                loss.return_nll
                + self.config.classification_weight * loss.classification
                + self.config.brier_weight * loss.brier
                + self.config.ordinal_weight * loss.ordinal
                + self.config.consistency_weight * loss.consistency
                + self.config.symmetry_weight * symmetry
            )
            totals["predictive_loss"] += float(predictive_loss.detach()) * tokens
            totals["return_nll"] += float(loss.return_nll.detach()) * tokens
            totals["return_mae"] += float(loss.return_mae.detach()) * tokens
            totals["classification"] += float(loss.classification.detach()) * tokens
            totals["brier"] += float(loss.brier.detach()) * tokens
            totals["ordinal"] += float(loss.ordinal.detach()) * tokens
            totals["consistency"] += float(loss.consistency.detach()) * tokens
            totals["symmetry"] += float(symmetry.detach()) * tokens
            totals["kl"] += float(loss.kl.detach()) * tokens
            totals["latent_information"] += float(loss.latent_information.detach()) * tokens
            totals["accuracy"] += float(correct.detach())
            token_count += tokens
        if not token_count:
            raise ValueError("data loader produced no valid tokens")
        return {key: value / token_count for key, value in totals.items()}

    def _augment_z2_batch(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Append the exact up/down mirror of each state using the same model draw."""
        mirrored = {key: value.clone() for key, value in batch.items()}
        velocity_channels = batch["profiles"].shape[-1] - 4
        velocity_bins = isqrt(velocity_channels)
        if velocity_bins**2 != velocity_channels:
            raise ValueError("Z2 augmentation requires a square velocity grid")
        mirrored["profiles"].zero_()
        mirrored["bin_coordinates"].zero_()
        for batch_index in range(len(batch["profiles"])):
            for state_index in range(batch["profiles"].shape[1]):
                valid_bins = int(
                    batch["profile_mask"][batch_index, state_index].sum().item()
                )
                if not valid_bins:
                    continue
                profile = batch["profiles"][
                    batch_index, state_index, :valid_bins
                ]
                scalar = profile[:, :4].flip(dims=(0,))
                scalar[:, 3] = -scalar[:, 3]
                velocity = profile[:, 4:].reshape(
                    valid_bins, velocity_bins, velocity_bins
                )
                velocity = velocity.flip(dims=(0, 1)).reshape(
                    valid_bins, velocity_channels
                )
                mirrored["profiles"][
                    batch_index, state_index, :valid_bins
                ] = torch.cat((scalar, velocity), dim=-1)
                mirrored["bin_coordinates"][
                    batch_index, state_index, :valid_bins
                ] = -batch["bin_coordinates"][
                    batch_index, state_index, :valid_bins
                ].flip(dims=(0,))

        features = batch["state_features"]
        mirrored_features = mirrored["state_features"]
        mirrored_features[..., 0:3] = -features[..., 0:3]
        mirrored_features[..., 3] = -features[..., 4]
        mirrored_features[..., 4] = -features[..., 3]
        mirrored["targets"] = REGIME_COUNT - 1 - batch["targets"]
        mirrored["forward_returns"] = -batch["forward_returns"]
        mirrored["standardized_returns"] = -batch["standardized_returns"]
        mirrored["return_locations"] = -batch["return_locations"]
        return {
            key: torch.cat((batch[key], mirrored[key]), dim=0) for key in batch
        }

    @staticmethod
    def _z2_symmetry_loss(
        output: Any, batch: dict[str, Tensor], original_batch_size: int
    ) -> Tensor:
        probability = torch.softmax(output.regime_logits, dim=-1)
        original_probability = probability[:original_batch_size]
        mirrored_probability = probability[original_batch_size:]
        mask = batch["valid_mask"][:original_batch_size].to(probability.dtype)
        denominator = mask.sum().clamp_min(1.0)
        class_error = (
            mirrored_probability - original_probability.flip(dims=(-1,))
        ).square().mean(dim=-1)
        return_error = torch.nn.functional.smooth_l1_loss(
            output.return_mean[original_batch_size:],
            -output.return_mean[:original_batch_size],
            reduction="none",
        )
        variance_error = torch.nn.functional.smooth_l1_loss(
            output.return_log_variance[original_batch_size:],
            output.return_log_variance[:original_batch_size],
            reduction="none",
        )
        return (
            ((class_error + return_error + 0.25 * variance_error) * mask).sum()
            / denominator
        )


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
            "format_version": 16,
            "regime_protocol": REGIME_PROTOCOL,
            "regime_names": list(REGIME_NAMES),
            "model_state": model.state_dict(),
            "model_config": asdict(model_config),
            "trainer_config": asdict(trainer_config),
            "history": list(result.history),
            "pretraining_history": list(result.pretraining_history),
            "best_epoch": result.best_epoch,
            "best_validation_loss": result.best_validation_loss,
            "metadata": metadata,
        },
        destination,
    )
