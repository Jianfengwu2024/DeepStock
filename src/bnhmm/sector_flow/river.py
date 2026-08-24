"""Project completed sessions onto a frozen monthly river topology."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .types import (
    DailyRiverSlice,
    MonthlyRiverTopology,
    NormalizedSectorPanel,
    SectorFactorParameters,
    SectorFlowPosterior,
)


class DailyRiverSliceBuilder:
    def build(
        self,
        panel: NormalizedSectorPanel,
        factor_parameters: SectorFactorParameters,
        flow: SectorFlowPosterior,
        topology: MonthlyRiverTopology,
        session: np.datetime64,
    ) -> DailyRiverSlice:
        day = np.datetime64(session, "D")
        if day.astype("datetime64[M]") != topology.month:
            raise ValueError("session does not belong to the frozen topology month")
        if panel.symbols != flow.symbols:
            raise ValueError("normalized panel and sector flow symbols do not align")
        if factor_parameters.loadings.shape != (len(panel.symbols),):
            raise ValueError("factor loadings do not align with the normalized panel")
        panel_index = {symbol: index for index, symbol in enumerate(panel.symbols)}
        if set(panel_index) != set(topology.symbols):
            raise ValueError("topology and normalized panel symbols do not align")
        rows = np.flatnonzero(panel.timestamps.astype("datetime64[D]") == day)
        if not len(rows):
            raise ValueError("completed session is unavailable in the normalized panel")
        ordered_symbols = tuple(
            topology.symbols[index] for index in topology.row_order
        )
        columns = np.asarray(
            [panel_index[symbol] for symbol in ordered_symbols], dtype=np.int64
        )
        velocity = panel.velocity[np.ix_(rows, columns)].T.copy()
        valid = panel.valid[np.ix_(rows, columns)].T.copy()
        valid &= flow.valid[rows][None, :]
        activity_valid = panel.activity_valid[np.ix_(rows, columns)].T
        activity = panel.activity[np.ix_(rows, columns)].T.copy()
        loadings = factor_parameters.loadings[columns]
        residual = velocity - loadings[:, None] * flow.factor_mean[rows][None, :]
        velocity[~valid] = np.nan
        residual[~valid] = np.nan
        activity[~activity_valid] = np.nan
        return DailyRiverSlice(
            session=day,
            topology_month=topology.month,
            symbols=ordered_symbols,
            timestamps=panel.timestamps[rows],
            velocity=velocity,
            residual_velocity=residual,
            activity=activity,
            valid=valid,
        )


def daily_statistics(
    river_slice: DailyRiverSlice,
    topology: MonthlyRiverTopology,
) -> NDArray[np.float64]:
    """Return six fixed, interpretable statistics per stock."""

    topology_index = {symbol: index for index, symbol in enumerate(topology.symbols)}
    if set(topology_index) != set(river_slice.symbols):
        raise ValueError("river slice and topology symbols do not align")
    count = len(river_slice.symbols)
    result = np.zeros((count, 6), dtype=np.float64)
    displacement = np.zeros(count, dtype=np.float64)
    for row in range(count):
        valid = river_slice.valid[row]
        velocity = river_slice.velocity[row, valid]
        residual = river_slice.residual_velocity[row, valid]
        if len(velocity):
            displacement[row] = float(np.sum(velocity) / np.sqrt(len(velocity)))
            path = float(np.sum(np.abs(velocity)))
            result[row, 3] = abs(float(np.sum(velocity))) / max(path, 1e-12)
            result[row, 4] = float(np.mean(np.square(residual)))
        activity = river_slice.activity[row]
        finite_activity = activity[np.isfinite(activity)]
        if len(finite_activity):
            result[row, 5] = float(np.median(finite_activity))
    result[:, 0] = displacement

    bands = np.asarray(
        [topology.band_ids[topology_index[symbol]] for symbol in river_slice.symbols],
        dtype=np.int64,
    )
    for row, band in enumerate(bands):
        peers = bands == band if band >= 0 else np.zeros(count, dtype=np.bool_)
        result[row, 1] = float(np.mean(displacement[peers])) if peers.any() else 0.0
    result[:, 2] = result[:, 0] - result[:, 1]
    return result
