import numpy as np

from homesignal.evaluation.metrics import (
    cross_sectional_spearman,
    regression_metrics,
    summarise_folds,
)


def test_perfect_prediction():
    y = np.array([1.0, -2.0, 3.0, 0.5])
    m = regression_metrics(y, y)
    assert (
        m["mae"] == 0
        and m["rmse"] == 0
        and m["r2"] == 1
        and m["directional_accuracy"] == 1
        and m["n"] == 4
    )
    assert np.isclose(m["spearman"], 1.0)


def test_known_values_and_nan_handling():
    y = np.array([0.0, 2.0, 4.0, np.nan])
    p = np.array([1.0, 1.0, 5.0, 1.0])
    m = regression_metrics(y, p)
    assert m["n"] == 3
    assert np.isclose(m["mae"], 1.0)
    assert np.isclose(m["rmse"], 1.0)
    assert np.isclose(m["r2"], 1 - 3 / 8)
    assert m["directional_accuracy"] == 1.0


def test_directional_accuracy_counts_sign():
    y = np.array([-1.0, 1.0, -1.0, 1.0])
    p = np.array([-1.0, -1.0, 1.0, 1.0])
    assert regression_metrics(y, p)["directional_accuracy"] == 0.5


def test_cross_sectional_spearman_ignores_level_shifts():
    # Perfect ranking within each origin but a large level offset between origins.
    y = np.array([1.0, 2.0, 3.0, 11.0, 12.0, 13.0])
    p = np.array([0.1, 0.2, 0.3, -5.0, -4.0, -3.0])
    g = np.array(["a", "a", "a", "b", "b", "b"])
    assert np.isclose(cross_sectional_spearman(y, p, g), 1.0)
    assert regression_metrics(y, p)["spearman"] < 1.0


def test_summarise_folds():
    keys = ("mae", "rmse", "r2", "spearman", "cs_spearman", "directional_accuracy", "n")
    s = summarise_folds([{k: 1.0 for k in keys}, {k: 3.0 for k in keys}])
    assert s["mae_mean"] == 2.0 and np.isclose(s["mae_std"], np.sqrt(2))
