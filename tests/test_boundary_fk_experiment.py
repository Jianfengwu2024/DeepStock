from __future__ import annotations

from datetime import date

import numpy as np

from bnhmm.boundary_fk.pipeline import BoundaryFlowPipeline, BoundaryFlowPipelineConfig
from test_boundary_fk_pipeline import _walk_forward_data


def test_requested_evaluation_range_limits_sample_out_months(tmp_path) -> None:
    data = _walk_forward_data(220)
    start = np.datetime64("2024-07-01")
    end = np.datetime64("2024-09-30")
    ranged = type(data)(
        states=data.states,
        labels=data.labels,
        memberships=data.memberships,
        symbols=data.symbols,
        member_next_returns=data.member_next_returns,
        member_next_valid=data.member_next_valid,
        evaluation_start=start,
        evaluation_end=end,
    )

    result = BoundaryFlowPipeline(
        BoundaryFlowPipelineConfig(minimum_train_sessions=60, max_folds=None, fk_paths=1000)
    ).run(ranged, tmp_path)

    assert result.stages[0].test_months
    assert all("2024-07" <= month <= "2024-09" for month in result.stages[0].test_months)
