"""Leakage-safe labels and sequence windows for profile-state training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from ..config import LabelConfig, SequenceConfig
from ..types import ProfileState


STATE_FEATURE_DIM = 9


@dataclass(frozen=True, slots=True)
class ProfileObservations:
    symbol: str
    profiles: tuple[NDArray[np.float32], ...]
    bin_coordinates: tuple[NDArray[np.float32], ...]
    state_features: NDArray[np.float32]
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
            profiles=tuple(self.profiles[int(index)] for index in indices),
            bin_coordinates=tuple(
                self.bin_coordinates[int(index)] for index in indices
            ),
            state_features=self.state_features[indices],
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
    profiles: tuple[NDArray[np.float32], ...]
    bin_coordinates: tuple[NDArray[np.float32], ...]
    state_features: NDArray[np.float32]
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
            if forward_return < self.config.bear_threshold:
                regime = 0
            elif forward_return <= self.config.bull_threshold:
                regime = 1
            else:
                regime = 2
            rows.append(
                (state, execution_state, target_state, forward_return, regime)
            )

        if not rows:
            raise ValueError("profile states do not contain enough future sessions")
        return ProfileObservations(
            symbol=symbol,
            profiles=tuple(self._profile_channels(row[0]) for row in rows),
            bin_coordinates=tuple(
                row[0].bin_coordinates.astype(np.float32) for row in rows
            ),
            state_features=np.asarray(
                [self._state_features(row[0]) for row in rows], dtype=np.float32
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

    @staticmethod
    def _profile_channels(state: ProfileState) -> NDArray[np.float32]:
        return np.concatenate(
            (
                state.price_distribution[:, None],
                state.distribution[:, None],
                state.normalized_relative_volume_level[:, None],
                state.normalized_bin_coordinates[:, None],
                state.joint_velocity_distribution.reshape(
                    len(state.distribution), -1
                ),
            ),
            axis=1,
        ).astype(np.float32)

    @staticmethod
    def _state_features(state: ProfileState) -> NDArray[np.float32]:
        relative_volume_mean = state.total_relative_volume / max(state.bar_count, 1)

        def scaled_log(numerator: float, denominator: float) -> float:
            return float(
                np.clip(
                    np.log(numerator / denominator) / state.location_scale,
                    -4.0,
                    4.0,
                )
            )

        normalized_relative_volume = float(
            np.clip(
                (
                    np.log1p(max(relative_volume_mean, 0.0))
                    - state.relative_volume_mean
                )
                / state.relative_volume_scale,
                -4.0,
                4.0,
            )
        )

        return np.asarray(
            (
                scaled_log(state.start_price, state.anchor_price),
                scaled_log(state.end_price, state.anchor_price),
                scaled_log(state.end_price, state.start_price),
                scaled_log(state.high_price, state.anchor_price),
                scaled_log(state.low_price, state.anchor_price),
                float(
                    np.clip(
                        np.log(state.high_price / state.low_price)
                        / state.location_scale,
                        0.0,
                        4.0,
                    )
                ),
                normalized_relative_volume,
                float(np.clip(state.bar_count / 240.0, 0.0, 1.0)),
                float(state.profile_clipped),
            ),
            dtype=np.float32,
        )


class SequenceWindowBuilder:
    def __init__(self, config: SequenceConfig | None = None) -> None:
        self.config = config or SequenceConfig()

    def build(self, observations: ProfileObservations) -> list[ProfileSequence]:
        size = len(observations)
        if size < self.config.length:
            return []
        ends = list(range(self.config.length, size + 1, self.config.stride))
        if ends[-1] != size:
            ends.append(size)

        result: list[ProfileSequence] = []
        for end in ends:
            start = end - self.config.length
            section = slice(start, end)
            result.append(
                ProfileSequence(
                    symbol=observations.symbol,
                    profiles=observations.profiles[section],
                    bin_coordinates=observations.bin_coordinates[section],
                    state_features=observations.state_features[section],
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
