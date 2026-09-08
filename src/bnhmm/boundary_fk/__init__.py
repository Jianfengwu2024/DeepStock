"""Causal whole-sector Bayesian Feynman-Kac model components."""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORT_MODULES = {
    "BoundaryFeynmanKacConfig": "feynman_kac",
    "BoundaryFeynmanKacForecast": "feynman_kac",
    "BranchFeynmanKacForecaster": "feynman_kac",
    "BayesianBranchModel": "branch_model",
    "BayesianBranchModelConfig": "branch_model",
    "BayesianBranchPosterior": "branch_model",
    "BoundaryPolicy": "policy",
    "BoundaryPolicyConfig": "policy",
    "BoundaryPolicyDecision": "policy",
    "BoundaryEvaluationConfig": "evaluation",
    "BoundaryEvaluationMetrics": "evaluation",
    "BoundaryFlowPipeline": "pipeline",
    "BoundaryFlowPipelineConfig": "pipeline",
    "BoundaryFlowRunResult": "pipeline",
    "BoundaryStageResult": "pipeline",
    "BoundaryWalkForwardData": "pipeline",
    "build_boundary_experiment_data": "experiment",
    "run_boundary_flow_experiment": "experiment",
    "BoundaryNormalizationConfig": "normalization",
    "CausalBoundaryNormalizer": "normalization",
    "SyntheticSectorIndexBuilder": "synthetic_index",
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
