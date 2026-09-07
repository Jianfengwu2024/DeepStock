"""Causal two-branch Feynman-Kac model components.

Exports are lazy so the synthetic-index and normalization layers remain usable
without importing the optional clustering stack.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORT_MODULES = {
    "BayesianBoundaryPosterior": "bayesian_state",
    "BayesianBoundaryStateConfig": "bayesian_state",
    "BayesianBoundaryStateModel": "bayesian_state",
    "BoundaryStateLabels": "bayesian_state",
    "BoundaryFeynmanKacConfig": "feynman_kac",
    "BoundaryFeynmanKacForecast": "feynman_kac",
    "BoundaryFeynmanKacForecaster": "feynman_kac",
    "BranchFeynmanKacForecaster": "feynman_kac",
    "BayesianBranchModel": "branch_model",
    "BayesianBranchModelConfig": "branch_model",
    "BayesianBranchPosterior": "branch_model",
    "TwoBranchDailyStateBuilder": "branch_state",
    "TwoBranchDailyStateConfig": "branch_state",
    "BoundaryPolicy": "policy",
    "BoundaryPolicyConfig": "policy",
    "BoundaryPolicyDecision": "policy",
    "BoundaryEvaluationConfig": "evaluation",
    "BoundaryEvaluationMetrics": "evaluation",
    "BoundaryForecastObservation": "evaluation",
    "evaluate_boundary_forecasts": "evaluation",
    "BOUNDARY_MODEL_NAMES": "pipeline",
    "BoundaryFlowPipeline": "pipeline",
    "BoundaryFlowPipelineConfig": "pipeline",
    "BoundaryFlowRunResult": "pipeline",
    "BoundaryStageResult": "pipeline",
    "BoundaryWalkForwardData": "pipeline",
    "build_boundary_experiment_data": "experiment",
    "run_boundary_flow_experiment": "experiment",
    "BoundaryDailyStateBuilder": "daily_state",
    "BoundaryDailyStateConfig": "daily_state",
    "BoundaryTopologyBuilder": "daily_state",
    "BoundaryFieldConfig": "moments",
    "BoundaryFieldExtractor": "moments",
    "BoundaryNormalizationConfig": "normalization",
    "CausalBoundaryNormalizer": "normalization",
    "SyntheticSectorIndexBuilder": "synthetic_index",
    "BoundaryDailyStates": "types",
    "BoundaryMinuteMoments": "types",
    "MonthlyIndexWeights": "types",
    "NormalizedBoundaryPanel": "types",
    "SyntheticSectorIndex": "types",
    "MonthlyBranchMembership": "types",
    "PeriodicBranchMembership": "types",
    "BranchDailyStates": "types",
    "BranchLabels": "types",
    "BranchDataset": "types",
    "TwoBranchDailyStates": "types",
    "TwoBranchDataset": "types",
    "TwoBranchLabels": "types",
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
