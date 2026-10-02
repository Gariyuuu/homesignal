import numpy as np
import pandas as pd

from homesignal.features.lags import (
    drawdown_from_peak,
    forward_target,
    momentum_features,
    pct_change,
    volatility,
)


def _wide() -> pd.DataFrame:
    months = pd.date_range("2020-01-01", periods=6, freq="MS")
    return pd.DataFrame(
        [[100, 110, 121, 121, 110, 99]], index=["00001"], columns=months, dtype=float
    )


def test_pct_change_uses_past_only():
    w = _wide()
    g1 = pct_change(w, 1)
    assert np.isnan(g1.iloc[0, 0])
    assert np.isclose(g1.iloc[0, 1], 10.0)
    assert np.isclose(g1.iloc[0, 5], -10.0)


def test_forward_target_uses_future_column():
    w = _wide()
    t = forward_target(w, 2)
    assert np.isclose(t.iloc[0, 0], 21.0)
    assert np.isnan(t.iloc[0, 4]) and np.isnan(t.iloc[0, 5])


def test_drawdown_non_positive_and_zero_at_peak():
    w = _wide()
    d = drawdown_from_peak(w, 36)
    assert (d.fillna(0) <= 1e-9).all().all()
    assert d.iloc[0, 2] == 0.0
    assert np.isclose(d.iloc[0, 5], (99 / 121 - 1) * 100)


def test_volatility_needs_full_window():
    w = _wide()
    v = volatility(w, 3)
    assert v.iloc[0, :3].isna().all()
    assert v.iloc[0, 3] >= 0


def test_momentum_features_ignore_future_values():
    w = _wide()
    feats = momentum_features(w, [1, 2], 3, 4)
    w2 = w.copy()
    w2.iloc[0, 4:] = 1e6  # change the future
    feats2 = momentum_features(w2, [1, 2], 3, 4)
    for name in feats:
        pd.testing.assert_frame_equal(feats[name].iloc[:, :4], feats2[name].iloc[:, :4])
