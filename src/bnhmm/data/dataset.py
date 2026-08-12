"""Leakage-safe labels and sequence windows for profile-state training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from ..config import LabelConfig, SequenceConfig
from ..types import ProfileState


@dataclass(frozen=True, slots=True)
class ProfileObservations:
    symbol: str
    profiles: NDArray[np.float32]
    timeframe_ids: NDArray[np.int64]
    targets: NDArray[np.int64]
    signal_times: NDArray[np.datetime64]
    execution_times: NDArray[np.datetime64]
    target_times: NDArray[np.datetime64]
    entry_prices: NDArray[np.float64]
    forward_returns: NDArray[np.float64]

    def __len__(self) -> int:
        return len(self.targets)

    def take(self, indices: NDArray[np.integer]) -> "ProfileObservations":
        return ProfileObservations(
            symbol=self.symbol,
            profiles=self.profiles[indices],
            timeframe_ids=self.timeframe_ids[indices],
            targets=self.targets[indices],
            signal_times=self.signal_times[indices],
            execution_times=self.execution_times[indices],
            target_times=self.target_times[indices],
            entry_prices=self.entry_prices[indices],
            forward_returns=self.forward_returns[indices],
        )

    def temporal_split(
        self, cutoff: str | np.datetime64
    ) -> tuple["ProfileObservations", "ProfileObservations"]:
        """Split while purging training labels that mature after the cutoff."""
        cutoff_time = np.datetime64(cutoff, "ns")
        train = np.flatnonzero(self.target_times < cutoff_time)
        validation = np.flatnonzero(self.signal_times >= cutoff_time)
        return self.take(train), self.take(validation)


@dataclass(frozen=True, slots=True)
class ProfileSequence:
    symbol: str
    profiles: NDArray[np.float32]
    timeframe_ids: NDArray[np.int64]
    targets: NDArray[np.int64]
    signal_times: NDArray[np.datetime64]
    execution_times: NDArray[np.datetime64]
    target_times: NDArray[np.datetime64]
    entry_prices: NDArray[np.float64]
    forward_returns: NDArray[np.float64]

    def __len__(self) -> int:
        return len(self.targets)


class ProfileLabelBuilder:
    """Map state-end prices to future-session return regimes."""

    def __init__(self, config: LabelConfig | None = None) -> None:
        self.config = config or LabelConfig()

    def build(self, states: Sequence[ProfileState]) -> ProfileObservations:
        if not states:
            raise ValueError("at least one profile state is required")
        symbol = states[0].symbol
        if any(state.symbol != symbol for state in states):
            raise ValueError("all states must belong to one symbol")

        ordered = sorted(
            states,
            key=lambda state: (state.period_end, state.timeframe == "1d"),
        )
        daily = [state for state in ordered if state.timeframe == "1d"]
        session_index = {
            state.period_start.astype("datetime64[D]"): index
            for index, state in enumerate(daily)
        }
        rows: list[tuple[ProfileState, ProfileState, ProfileState, float, int]] = []
        for position, state in enumerate(ordered):
            index = session_index.get(state.period_start.astype("datetime64[D]"))
            if index is None:
                continue
            target_index = index + self.config.forward_sessions
            if target_index >= len(daily):
                continue
            execution_state = None
            for candidate_position in range(position + 1, len(ordered)):
                candidate = ordered[candidate_position]
                if candidate.period_start >= state.period_end:
                    execution_state = candidate
                    break
            if execution_state is None:
                continue
            target_state = daily[target_index]
            if execution_state.period_start > target_state.period_end:
                continue
            forward_return = target_state.end_price / execution_state.start_price - 1.0
            regime = int(np.digitize(forward_return, self.config.thresholds))
            rows.append(
                (state, execution_state, target_state, forward_return, regime)
            )

        if not rows:
            raise ValueError("profile states do not contain enough future sessions")
        return ProfileObservations(
            symbol=symbol,
            profiles=np.asarray(
                [row[0].distribution for row in rows], dtype=np.float32
            ),
            timeframe_ids=np.asarray(
                [0 if row[0].timeframe == "60m" else 1 for row in rows],
                dtype=np.int64,
            ),
            targets=np.asarray([row[4] for row in rows], dtype=np.int64),
            signal_times=np.asarray(
                [row[0].period_end for row in rows], dtype="datetime64[ns]"
            ),
            execution_times=np.asarray(
                [row[1].first_bar_time for row in rows], dtype="datetime64[ns]"
            ),
            target_times=np.asarray(
                [row[2].period_end for row in rows], dtype="datetime64[ns]"
            ),
            entry_prices=np.asarray([row[1].start_price for row in rows]),
            forward_returns=np.asarray([row[3] for row in rows], dtype=np.float64),
        )


class SequenceWindowBuilder:
    def __init__(self, config: SequenceConfig | None = None) -> None:
        self.config = config or SequenceConfig()

    def build(self, observations: ProfileObservations) -> list[ProfileSequence]:
        size = len(observations)
        if size < self.config.minimum_length:
            return []
        starts = list(range(0, max(size - self.config.minimum_length + 1, 1), self.config.stride))
        tail_start = max(0, size - self.config.length)
        if not starts or starts[-1] != tail_start:
            starts.append(tail_start)

        result: list[ProfileSequence] = []
        seen: set[tuple[int, int]] = set()
        for start in starts:
            end = min(start + self.config.length, size)
            if end - start < self.config.minimum_length or (start, end) in seen:
                continue
            seen.add((start, end))
            section = slice(start, end)
            result.append(
                ProfileSequence(
                    symbol=observations.symbol,
                    profiles=observations.profiles[section],
                    timeframe_ids=observations.timeframe_ids[section],
                    targets=observations.targets[section],
                    signal_times=observations.signal_times[section],
                    execution_times=observations.execution_times[section],
                    target_times=observations.target_times[section],
                    entry_prices=observations.entry_prices[section],
                    forward_returns=observations.forward_returns[section],
                )
            )
        return result
