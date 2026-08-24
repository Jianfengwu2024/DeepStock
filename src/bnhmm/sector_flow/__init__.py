"""Bayesian sector-flow inference from synchronized one-minute bars.

Public objects are loaded on first access so data-only consumers do not need the
optional PyTorch regime-model dependency.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORT_MODULES = {
    "AttributionConfig": "attribution",
    "FlowAttributionModel": "attribution",
    "SectorMinutePanelBuilder": "panel",
    "DailyRiverSliceBuilder": "river",
    "daily_statistics": "river",
    "SectorFlowConfig": "pipeline",
    "SectorFlowPipeline": "pipeline",
    "SectorFlowRunResult": "pipeline",
    "BayesianSectorFactor": "factor",
    "SectorFactorConfig": "factor",
    "GraphStateConfig": "graph_state",
    "SectorGraphStateModel": "graph_state",
    "CausalSectorNormalizer": "normalization",
    "SectorNormalizationConfig": "normalization",
    "PathEfficiencyAnalyzer": "ranking",
    "RankingConfig": "ranking",
    "SourceProximityRanker": "ranking",
    "SectorRegimeDataConfig": "regime_data",
    "SectorRegimeDatasetBuilder": "regime_data",
    "SectorRegimeSample": "regime_data",
    "SectorRegimeState": "regime_data",
    "SectorStateImage": "regime_data",
    "build_sector_state_image": "regime_data",
    "BayesianSectorRegimeHMM": "regime_model",
    "SectorRegimeModelConfig": "regime_model",
    "SectorRegimePass": "regime_model",
    "SectorRegimePosterior": "regime_model",
    "SectorRegimePredictionRecord": "regime_evaluation",
    "evaluate_sector_regimes": "regime_evaluation",
    "SectorRegimeTrainer": "regime_training",
    "SectorRegimeTrainerConfig": "regime_training",
    "SectorRegimeTrainingResult": "regime_training",
    "state_reconstruction_loss": "regime_training",
    "SectorPolicyObservation": "regime_policy",
    "SectorRegimePolicyConfig": "regime_policy",
    "StockSelectionSnapshot": "regime_policy",
    "apply_sector_regime_policy": "regime_policy",
    "evaluate_sector_regime_policy": "regime_policy",
    "RiverTopologyBuilder": "topology",
    "TopologyConfig": "topology",
    "DailyGraphState": "types",
    "DailyRiverSlice": "types",
    "FlowAttributionParameters": "types",
    "FlowAttributionPosterior": "types",
    "GraphStateParameters": "types",
    "MonthlyRiverTopology": "types",
    "NormalizedSectorPanel": "types",
    "PathWindowMetrics": "types",
    "SectorFactorParameters": "types",
    "SectorFlowPosterior": "types",
    "SectorGraphPrediction": "types",
    "SectorMinutePanel": "types",
    "SourceProximityPosterior": "types",
    "StockFlowAttribution": "types",
    "StockPathMetrics": "types",
}

__all__ = sorted(_EXPORT_MODULES)


def __getattr__(name: str) -> Any:
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"{__name__}.{module_name}"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted((*globals(), *__all__))
