"""Build a causal minute sector index from constituent market capitalizations."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..data.source import MarketCapRecord
from ..sector_flow.types import SectorMinutePanel
from .types import MonthlyIndexWeights, SyntheticSectorIndex


class SyntheticSectorIndexBuilder:
    def __init__(
        self,
        *,
        min_active_weight: float = 0.5,
        min_active_fraction: float = 0.5,
        min_constituents: int = 3,
    ) -> None:
        if not 0.0 < min_active_weight <= 1.0:
            raise ValueError("min_active_weight must lie in (0, 1]")
        if not 0.0 < min_active_fraction <= 1.0:
            raise ValueError("min_active_fraction must lie in (0, 1]")
        if min_constituents < 1:
            raise ValueError("min_constituents must be positive")
        self.min_active_weight = float(min_active_weight)
        self.min_active_fraction = float(min_active_fraction)
        self.min_constituents = int(min_constituents)

    def build(
        self,
        panel: SectorMinutePanel,
        market_caps: Sequence[MarketCapRecord],
    ) -> SyntheticSectorIndex:
        has_minutes = np.any(panel.valid, axis=0)
        included = tuple(
            symbol for symbol, available in zip(panel.symbols, has_minutes, strict=True) if available
        )
        excluded = tuple(
            symbol for symbol, available in zip(panel.symbols, has_minutes, strict=True) if not available
        )
        if len(included) < self.min_constituents:
            raise ValueError(
                f"剔除无一分钟线股票后有效成分少于 {self.min_constituents} 只："
                + ", ".join(excluded)
            )

        caps_by_symbol = self._group_caps(market_caps, set(included))
        monthly_weights = self._build_monthly_weights(
            panel,
            included,
            excluded,
            caps_by_symbol,
        )
        if not monthly_weights:
            raise ValueError("请求区间没有可用的上月末正流通市值，禁止等权回退")

        velocity, valid, active_weight, active_fraction = self._minute_returns(
            panel,
            monthly_weights,
        )
        (
            daily_sessions,
            daily_return,
            daily_valid,
            daily_weight,
            daily_fraction,
        ) = self._daily_returns(panel, monthly_weights)
        return SyntheticSectorIndex(
            timestamps=panel.timestamps.copy(),
            velocity=velocity,
            valid=valid,
            active_weight=active_weight,
            active_fraction=active_fraction,
            monthly_weights=monthly_weights,
            daily_sessions=daily_sessions,
            daily_open_to_close_return=daily_return,
            daily_valid=daily_valid,
            daily_active_weight=daily_weight,
            daily_active_fraction=daily_fraction,
            excluded_symbols=excluded,
        )

    @staticmethod
    def _group_caps(
        records: Sequence[MarketCapRecord],
        symbols: set[str],
    ) -> dict[str, tuple[MarketCapRecord, ...]]:
        grouped: dict[str, list[MarketCapRecord]] = {symbol: [] for symbol in symbols}
        seen: set[tuple[str, int]] = set()
        for record in records:
            if record.symbol not in symbols:
                continue
            key = (record.symbol, int(record.session.astype("datetime64[D]").astype(np.int64)))
            if key in seen:
                raise ValueError(
                    f"重复流通市值记录：{record.symbol}/{record.session.astype('datetime64[D]')}"
                )
            seen.add(key)
            grouped[record.symbol].append(record)
        return {
            symbol: tuple(sorted(rows, key=lambda row: row.session))
            for symbol, rows in grouped.items()
        }

    def _build_monthly_weights(
        self,
        panel: SectorMinutePanel,
        included: tuple[str, ...],
        globally_excluded: tuple[str, ...],
        caps_by_symbol: dict[str, tuple[MarketCapRecord, ...]],
    ) -> tuple[MonthlyIndexWeights, ...]:
        months = np.unique(panel.timestamps.astype("datetime64[M]"))
        panel_columns = {symbol: column for column, symbol in enumerate(panel.symbols)}
        result: list[MonthlyIndexWeights] = []
        for month_index, month in enumerate(months):
            month_start = month.astype("datetime64[ns]")
            symbols: list[str] = []
            values: list[float] = []
            cap_dates: list[np.datetime64] = []
            for symbol in included:
                column = panel_columns[symbol]
                observed_before_month = np.any(
                    panel.valid[:, column] & (panel.timestamps < month_start)
                )
                if not observed_before_month:
                    continue
                prior = [
                    row
                    for row in caps_by_symbol.get(symbol, ())
                    if row.session.astype("datetime64[ns]") < month_start
                ]
                if not prior:
                    continue
                latest = prior[-1]
                symbols.append(symbol)
                values.append(latest.circ_mv)
                cap_dates.append(latest.session.astype("datetime64[D]"))

            if not symbols:
                if month_index < len(months) - 1:
                    continue
                raise ValueError(
                    f"{month} 缺少上月末有效正流通市值，禁止等权回退"
                )
            if len(symbols) < self.min_constituents:
                raise ValueError(
                    f"{month} 有历史分钟线和上月末流通市值的成分少于 "
                    f"{self.min_constituents} 只"
                )

            raw = np.asarray(values, dtype=np.float64)
            weights = raw / float(raw.sum())
            month_excluded = tuple(
                symbol
                for symbol in panel.symbols
                if symbol not in set(symbols)
            )
            weight_date = max(cap_dates)
            result.append(
                MonthlyIndexWeights(
                    month=month,
                    weight_date=weight_date,
                    symbols=tuple(symbols),
                    weights=weights,
                    training_cutoff=weight_date,
                    excluded_symbols=tuple(
                        dict.fromkeys((*globally_excluded, *month_excluded))
                    ),
                )
            )
        return tuple(result)

    def _minute_returns(
        self,
        panel: SectorMinutePanel,
        monthly_weights: tuple[MonthlyIndexWeights, ...],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        returns, return_valid = panel.log_returns()
        rows = len(panel.timestamps)
        velocity = np.full(rows, np.nan, dtype=np.float64)
        valid = np.zeros(rows, dtype=np.bool_)
        active_weight = np.zeros(rows, dtype=np.float64)
        active_fraction = np.zeros(rows, dtype=np.float64)
        panel_columns = {symbol: column for column, symbol in enumerate(panel.symbols)}
        timestamp_months = panel.timestamps.astype("datetime64[M]")

        for frozen in monthly_weights:
            row_indices = np.flatnonzero(timestamp_months == frozen.month)
            columns = np.asarray([panel_columns[symbol] for symbol in frozen.symbols])
            for row in row_indices:
                available = return_valid[row, columns]
                weight_sum = float(frozen.weights[available].sum())
                fraction = float(np.mean(available))
                active_weight[row] = weight_sum
                active_fraction[row] = fraction
                if (
                    weight_sum + 1e-12 < self.min_active_weight
                    or fraction + 1e-12 < self.min_active_fraction
                ):
                    continue
                velocity[row] = float(
                    np.dot(frozen.weights[available], returns[row, columns][available])
                    / weight_sum
                )
                valid[row] = True
        return velocity, valid, active_weight, active_fraction

    def _daily_returns(
        self,
        panel: SectorMinutePanel,
        monthly_weights: tuple[MonthlyIndexWeights, ...],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        sessions = np.unique(panel.timestamps.astype("datetime64[D]"))
        returns = np.full(len(sessions), np.nan, dtype=np.float64)
        valid = np.zeros(len(sessions), dtype=np.bool_)
        active_weight = np.zeros(len(sessions), dtype=np.float64)
        active_fraction = np.zeros(len(sessions), dtype=np.float64)
        by_month = {item.month: item for item in monthly_weights}
        panel_columns = {symbol: column for column, symbol in enumerate(panel.symbols)}
        timestamp_sessions = panel.timestamps.astype("datetime64[D]")

        for destination, session in enumerate(sessions):
            frozen = by_month.get(session.astype("datetime64[M]"))
            if frozen is None:
                continue
            session_rows = np.flatnonzero(timestamp_sessions == session)
            stock_returns = np.full(len(frozen.symbols), np.nan, dtype=np.float64)
            available = np.zeros(len(frozen.symbols), dtype=np.bool_)
            for position, symbol in enumerate(frozen.symbols):
                column = panel_columns[symbol]
                observed = session_rows[panel.valid[session_rows, column]]
                if not len(observed):
                    continue
                first, last = int(observed[0]), int(observed[-1])
                stock_returns[position] = np.log(
                    panel.close[last, column] / panel.open[first, column]
                )
                available[position] = True
            weight_sum = float(frozen.weights[available].sum())
            fraction = float(np.mean(available))
            active_weight[destination] = weight_sum
            active_fraction[destination] = fraction
            if (
                weight_sum + 1e-12 < self.min_active_weight
                or fraction + 1e-12 < self.min_active_fraction
            ):
                continue
            returns[destination] = float(
                np.dot(frozen.weights[available], stock_returns[available]) / weight_sum
            )
            valid[destination] = True
        return sessions, returns, valid, active_weight, active_fraction
