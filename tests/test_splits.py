import pandas as pd
import pytest

from homesignal.config import EvaluationConfig
from homesignal.evaluation.splits import backtest_folds, holdout_fold, is_origin_month, split_label


def _cfg(**kw) -> EvaluationConfig:
    base = dict(
        origin_months=[3, 6, 9, 12],
        first_origin="2013-03",
        backtest_validation_years=[2016, 2017, 2018],
        gap_months=12,
        holdout_start="2020-01",
    )
    base.update(kw)
    return EvaluationConfig(**base)


def test_folds_have_gap_and_no_overlap():
    cfg = _cfg()
    months = pd.Series(pd.date_range("2013-03-01", "2021-12-01", freq="MS"))
    for fold in backtest_folds(cfg):
        tr, va = fold.train_mask(months), fold.val_mask(months)
        assert not (tr & va).any()
        # every training origin's 12-month target window ends on/before the first validation origin
        assert (months[tr] + pd.DateOffset(months=12) <= fold.val_start).all()
        assert fold.train_end + pd.DateOffset(months=cfg.gap_months) == fold.val_start
        assert (months[va] < pd.Timestamp(cfg.holdout_start)).all()


def test_folds_are_expanding():
    folds = backtest_folds(_cfg())
    ends = [f.train_end for f in folds]
    assert ends == sorted(ends) and len(set(ends)) == len(ends)


def test_validation_year_cannot_touch_holdout():
    with pytest.raises(ValueError):
        backtest_folds(_cfg(backtest_validation_years=[2019, 2020]))


def test_holdout_fold_training_stops_gap_before_holdout():
    cfg = _cfg()
    fold = holdout_fold(cfg, pd.Timestamp("2021-06-01"))
    assert fold.train_end == pd.Timestamp("2019-01-01")
    assert fold.val_start == pd.Timestamp("2020-01-01")
    assert fold.val_end == pd.Timestamp("2021-06-01")


def test_origin_month_and_split_label():
    cfg = _cfg()
    months = pd.Series(pd.to_datetime(["2012-12-01", "2013-03-01", "2013-04-01", "2020-03-01"]))
    assert is_origin_month(months, cfg).tolist() == [False, True, False, True]
    assert split_label(months, cfg).tolist() == ["backtest", "backtest", "backtest", "holdout"]
