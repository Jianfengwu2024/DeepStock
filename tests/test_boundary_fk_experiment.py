from __future__ import annotations

import numpy as np

from bnhmm.boundary_fk.experiment import _build_labels_and_side_data
from test_boundary_fk_bayesian_state import _dataset


def test_future_returns_cannot_change_earlier_labels_or_risk_quantiles() -> None:
    states, _ = _dataset(90)
    rng = np.random.default_rng(31)
    daily_returns = rng.normal(0.0, 0.01, len(states.sessions))
    valid = np.ones(len(states.sessions), dtype=np.bool_)

    base = _build_labels_and_side_data(
        states, states.sessions, daily_returns, valid
    )
    changed_returns = daily_returns.copy()
    changed_returns[70:] = 10.0
    changed = _build_labels_and_side_data(
        states, states.sessions, changed_returns, valid
    )

    np.testing.assert_allclose(
        base.labels.historical_volatility[:60],
        changed.labels.historical_volatility[:60],
        equal_nan=True,
    )
    np.testing.assert_allclose(
        base.labels.next_return[:60], changed.labels.next_return[:60], equal_nan=True
    )
    np.testing.assert_allclose(
        base.thickness_risk_quantile[:60], changed.thickness_risk_quantile[:60]
    )


def test_labels_point_to_the_next_available_trading_session() -> None:
    states, _ = _dataset(40)
    returns = np.linspace(-0.01, 0.01, len(states.sessions))
    side = _build_labels_and_side_data(
        states, states.sessions, returns, np.ones(len(returns), dtype=np.bool_)
    )

    assert side.labels.target_time[25] == states.sessions[26].astype("datetime64[ns]")
    assert side.labels.next_return[25] == returns[26]
