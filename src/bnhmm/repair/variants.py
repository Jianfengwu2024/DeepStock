"""Immutable contracts for the cumulative HoloBNHMM repair experiments."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RepairVariant:
    name: str
    stage: int
    control_mode: str = "full"
    state_representation: str = "price-volume"
    daily_encoder: str = "shallow-cnn"
    temporal_encoder: str = "transformer"
    latent_filter: str = "conditional"
    output_head: str = "dual"
    soft_band: bool = False
    ensemble_members: int = 1
    description: str = ""


class RepairVariantCatalog:
    """Resolve a public model name into one explicit cumulative contract."""

    _variants = (
        RepairVariant("HoloProbe-Prior", 0, control_mode="prior-only", description="Training-fold class prior only."),
        RepairVariant("HoloProbe-Metadata", 1, control_mode="metadata-only", description="K-line metadata and direct recent returns only."),
        RepairVariant("HoloProbe-Image", 2, control_mode="image-only", description="Two-dimensional field only, with every bypass disabled."),
        RepairVariant("HoloProbe-Shuffled", 3, control_mode="shuffled-image", description="Full model with images reassigned across each batch."),
        RepairVariant("HoloPhase", 4, control_mode="image-only", state_representation="price-velocity", description="Price-position by price-velocity phase image weighted by volume mass."),
        RepairVariant("HoloResNet", 5, control_mode="image-only", state_representation="price-velocity", daily_encoder="masked-resnet", description="Masked residual daily image encoder."),
        RepairVariant("HoloResTCN", 6, control_mode="image-only", state_representation="price-velocity", daily_encoder="masked-resnet", temporal_encoder="tcn", description="Residual image encoder with a short causal temporal convolution."),
        RepairVariant("HoloSticky", 7, control_mode="image-only", state_representation="price-velocity", daily_encoder="masked-resnet", temporal_encoder="tcn", latent_filter="sticky", description="Sparse sticky latent transition process."),
        RepairVariant("HoloStudent", 8, control_mode="image-only", state_representation="price-velocity", daily_encoder="masked-resnet", temporal_encoder="tcn", latent_filter="sticky", output_head="student-t", description="One Student-t posterior with distribution-derived regimes."),
        RepairVariant("HoloSoftBand", 9, control_mode="image-only", state_representation="price-velocity", daily_encoder="masked-resnet", temporal_encoder="tcn", latent_filter="sticky", output_head="student-t", soft_band=True, description="Low-edge observations receive soft directional supervision."),
        RepairVariant("HoloEnsemble", 10, control_mode="image-only", state_representation="price-velocity", daily_encoder="masked-resnet", temporal_encoder="tcn", latent_filter="sticky", output_head="student-t", soft_band=True, ensemble_members=5, description="Five-member deep ensemble for epistemic uncertainty."),
    )
    _by_name = {item.name.lower(): item for item in _variants}

    @classmethod
    def resolve(cls, name: str) -> RepairVariant:
        try:
            return cls._by_name[name.strip().lower()]
        except KeyError as error:
            raise ValueError(f"unknown Holo repair variant: {name}") from error

    @classmethod
    def all(cls) -> tuple[RepairVariant, ...]:
        return cls._variants

    @classmethod
    def names(cls) -> tuple[str, ...]:
        return tuple(item.name for item in cls._variants)
