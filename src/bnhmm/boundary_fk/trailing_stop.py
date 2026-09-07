"""Branch-level trailing equity stop for dynamically clustered FK branches."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True, slots=True)
class BranchEquityState:
    equity: float = 1.0
    peak: float = 1.0
    active: bool = False


@dataclass(frozen=True, slots=True)
class TrailingStopSnapshot:
    branch_equities: tuple[float, ...]
    branch_peaks: tuple[float, ...]
    branch_drawdowns: tuple[float, ...]
    newly_triggered: tuple[bool, ...]
    liquidation_pending: bool


class BranchTrailingStop:
    def __init__(self, branch_count: int = 2, maximum_drawdown: float = 0.05) -> None:
        if not math.isfinite(maximum_drawdown) or not 0.0 < maximum_drawdown < 1.0:
            raise ValueError("maximum branch drawdown must lie in (0, 1)")
        if branch_count < 1:
            raise ValueError("at least one branch is required")
        self.maximum_drawdown = maximum_drawdown
        self.branch_count = branch_count
        self._states = (BranchEquityState(),) * branch_count
        self._liquidation_pending = False

    @property
    def liquidation_pending(self) -> bool:
        return self._liquidation_pending

    def update(
        self,
        branch_returns: tuple[float, ...],
        held_branches: tuple[bool, ...],
    ) -> TrailingStopSnapshot:
        if len(branch_returns) != self.branch_count or len(held_branches) != self.branch_count:
            raise ValueError("trailing stop inputs must align with branch count")
        states: list[BranchEquityState] = []
        drawdowns: list[float] = []
        triggered: list[bool] = []
        for previous, daily_return, held in zip(
            self._states, branch_returns, held_branches, strict=True
        ):
            if not math.isfinite(daily_return):
                raise ValueError("branch return must be finite")
            if not held:
                state = BranchEquityState()
                drawdown = 0.0
            else:
                prior_equity = previous.equity if previous.active else 1.0
                prior_peak = previous.peak if previous.active else 1.0
                equity = prior_equity * (1.0 + daily_return)
                if equity <= 0.0:
                    raise ValueError("branch equity must remain positive")
                peak = max(prior_peak, equity)
                state = BranchEquityState(equity=equity, peak=peak, active=True)
                drawdown = max(0.0, 1.0 - equity / peak)
            states.append(state)
            drawdowns.append(drawdown)
            triggered.append(
                held and drawdown > self.maximum_drawdown + 1e-12
            )

        newly_triggered = tuple(bool(item and not self._liquidation_pending) for item in triggered)
        if any(triggered):
            self._liquidation_pending = True
        if self._liquidation_pending and not any(held_branches):
            self._liquidation_pending = False
            states = [BranchEquityState()] * self.branch_count
            drawdowns = [0.0] * self.branch_count
        self._states = tuple(states)
        return TrailingStopSnapshot(
            branch_equities=tuple(item.equity for item in states),
            branch_peaks=tuple(item.peak for item in states),
            branch_drawdowns=tuple(drawdowns),
            newly_triggered=newly_triggered,
            liquidation_pending=self._liquidation_pending,
        )


TwoBranchTrailingStop = BranchTrailingStop
