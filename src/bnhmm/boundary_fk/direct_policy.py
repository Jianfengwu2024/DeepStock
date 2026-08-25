"""Direct long-only policy driven only by signed path efficiency."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


BranchDirection = Literal["bull", "bear", "hold"]
BRANCH_EXPOSURE = 0.5


@dataclass(frozen=True, slots=True)
class DirectPathDecision:
    branch_directions: tuple[BranchDirection, BranchDirection]
    branch_target_exposures: tuple[float, float]
    selected_symbols: tuple[str, ...]
    target_weights: tuple[float, ...]

    @property
    def open_position(self) -> bool:
        return bool(self.selected_symbols)

    @property
    def held_branches(self) -> tuple[bool, bool]:
        return tuple(value > 0.0 for value in self.branch_target_exposures)


class DirectPathPolicy:
    def decide(
        self,
        path_efficiencies: tuple[float, float],
        *,
        branches: tuple[tuple[str, ...], tuple[str, ...]],
        previous_exposures: tuple[float, float],
    ) -> DirectPathDecision:
        if len(path_efficiencies) != 2 or len(branches) != 2:
            raise ValueError("direct path policy requires exactly two branches")
        if not np.all(np.isfinite(path_efficiencies)):
            raise ValueError("path efficiencies must be finite")
        if any(value not in (0.0, BRANCH_EXPOSURE) for value in previous_exposures):
            raise ValueError("previous branch exposures must be 0% or 50%")

        directions: list[BranchDirection] = []
        exposures: list[float] = []
        for value, previous in zip(
            path_efficiencies, previous_exposures, strict=True
        ):
            if value > 0.0:
                directions.append("bull")
                exposures.append(BRANCH_EXPOSURE)
            elif value < 0.0:
                directions.append("bear")
                exposures.append(0.0)
            else:
                directions.append("hold")
                exposures.append(previous)

        selected: list[str] = []
        weights: list[float] = []
        for symbols, exposure in zip(branches, exposures, strict=True):
            if exposure == 0.0:
                continue
            if not symbols:
                raise ValueError("a held branch must contain symbols")
            selected.extend(symbols)
            weights.extend((exposure / len(symbols),) * len(symbols))
        if len(selected) != len(set(selected)):
            raise ValueError("a held symbol cannot appear in both branches")

        return DirectPathDecision(
            branch_directions=(directions[0], directions[1]),
            branch_target_exposures=(exposures[0], exposures[1]),
            selected_symbols=tuple(selected),
            target_weights=tuple(weights),
        )
