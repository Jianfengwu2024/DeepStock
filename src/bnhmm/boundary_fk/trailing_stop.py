"""Branch-level trailing equity stop for the two-branch FK strategy."""

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
    branch_equities: tuple[float, float]
    branch_peaks: tuple[float, float]
    branch_drawdowns: tuple[float, float]
    newly_triggered: tuple[bool, bool]
    liquidation_pending: bool


class TwoBranchTrailingStop:
    def __init__(self, maximum_drawdown: float = 0.05) -> None:
        if not math.isfinite(maximum_drawdown) or not 0.0 < maximum_drawdown < 1.0:
            raise ValueError("maximum branch drawdown must lie in (0, 1)")
        self.maximum_drawdown = maximum_drawdown
        self._states = (BranchEquityState(), BranchEquityState())
        self._liquidation_pending = False

    @property
    def liquidation_pending(self) -> bool:
        return self._liquidation_pending

    def update(
        self,
        branch_returns: tuple[float, float],
        held_branches: tuple[bool, bool],
    ) -> TrailingStopSnapshot:
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

        newly_triggered = (
            bool(triggered[0] and not self._liquidation_pending),
            bool(triggered[1] and not self._liquidation_pending),
        )
        if any(triggered):
            self._liquidation_pending = True
        if self._liquidation_pending and not any(held_branches):
            self._liquidation_pending = False
            states = [BranchEquityState(), BranchEquityState()]
            drawdowns = [0.0, 0.0]
        self._states = (states[0], states[1])
        return TrailingStopSnapshot(
            branch_equities=(states[0].equity, states[1].equity),
            branch_peaks=(states[0].peak, states[1].peak),
            branch_drawdowns=(drawdowns[0], drawdowns[1]),
            newly_triggered=newly_triggered,
            liquidation_pending=self._liquidation_pending,
        )
