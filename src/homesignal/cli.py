"""Command-line entry point: ``homesignal <command>``."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence

from homesignal.config import (
    load_data_config,
    load_evaluation_config,
    load_feature_config,
    load_model_config,
)


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def cmd_download(_: argparse.Namespace) -> None:
    """Download and cache all raw sources."""
    from homesignal.data import acs, bls, freddie, zillow

    cfg = load_data_config()
    zillow.load_zhvi(cfg)
    zillow.load_zori(cfg)
    freddie.load_mortgage_rates(cfg)
    bls.load_bls(cfg)
    acs.load_acs(cfg)


def cmd_build_panel(_: argparse.Namespace) -> None:
    """Build the monthly ZIP panel with target and features."""
    from homesignal.features.build import build_panel, save_panel

    data_cfg = load_data_config()
    feat_cfg = load_feature_config()
    panel, report = build_panel(data_cfg, feat_cfg)
    save_panel(panel, report, data_cfg.processed_dir)


def cmd_backtest(args: argparse.Namespace) -> None:
    """Run the expanding-window backtest for all model families."""
    from homesignal.evaluation.backtest import run_backtest

    run_backtest(
        load_data_config(),
        load_feature_config(),
        load_evaluation_config(),
        load_model_config(),
        models=args.models,
        tag=args.tag,
        exclude_features=args.exclude or [],
    )


def cmd_tune(_: argparse.Namespace) -> None:
    """Tune LightGBM hyper-parameters on recent backtest folds."""
    from homesignal.models.tune import tune

    tune(load_data_config(), load_feature_config(), load_evaluation_config(), load_model_config())


def cmd_train_final(_: argparse.Namespace) -> None:
    """Train the final model on pre-holdout data, evaluate once on the holdout, save artifacts."""
    from homesignal.models.train import train_final

    train_final(
        load_data_config(), load_feature_config(), load_evaluation_config(), load_model_config()
    )


def cmd_explain(_: argparse.Namespace) -> None:
    """SHAP analysis, error analysis and calibration figures for the final model."""
    from homesignal.evaluation.error_analysis import run_error_analysis
    from homesignal.explain.shap_analysis import run_shap

    data_cfg = load_data_config()
    run_shap(data_cfg, load_feature_config(), load_evaluation_config(), load_model_config())
    run_error_analysis(
        data_cfg, load_feature_config(), load_evaluation_config(), load_model_config()
    )


def cmd_report(_: argparse.Namespace) -> None:
    """Write reports/evaluation.md from the experiment log and holdout results."""
    from homesignal.evaluation.report import write_reports

    write_reports(
        load_data_config(), load_feature_config(), load_evaluation_config(), load_model_config()
    )


def cmd_predict(args: argparse.Namespace) -> None:
    """Predict 12-month growth for one ZIP at one origin month, with top contributing features."""
    from homesignal.models.predict import predict_cli

    predict_cli(args.zip, args.month, top=args.top)


def cmd_export(_: argparse.Namespace) -> None:
    """Export the benchmark dataset (features + target + split labels) as one Parquet file."""
    from homesignal.evaluation.export import export_benchmark

    export_benchmark(
        load_data_config(), load_feature_config(), load_evaluation_config(), load_model_config()
    )


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    p = argparse.ArgumentParser(
        prog="homesignal", description="HomeSignal: ZIP-level 12-month home value growth."
    )
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("download", help=cmd_download.__doc__).set_defaults(func=cmd_download)
    sub.add_parser("build-panel", help=cmd_build_panel.__doc__).set_defaults(func=cmd_build_panel)
    bt = sub.add_parser("backtest", help=cmd_backtest.__doc__)
    bt.add_argument("--models", nargs="*", default=None, help="subset of model names")
    bt.add_argument("--tag", default="main", help="experiment tag")
    bt.add_argument("--exclude", nargs="*", help="feature names to drop (ablation)")
    bt.set_defaults(func=cmd_backtest)
    sub.add_parser("tune", help=cmd_tune.__doc__).set_defaults(func=cmd_tune)
    sub.add_parser("train-final", help=cmd_train_final.__doc__).set_defaults(func=cmd_train_final)
    sub.add_parser("explain", help=cmd_explain.__doc__).set_defaults(func=cmd_explain)
    sub.add_parser("report", help=cmd_report.__doc__).set_defaults(func=cmd_report)
    sub.add_parser("export", help=cmd_export.__doc__).set_defaults(func=cmd_export)
    pr = sub.add_parser("predict", help=cmd_predict.__doc__)
    pr.add_argument("zip")
    pr.add_argument("month", help="origin month, e.g. 2025-08")
    pr.add_argument("--top", type=int, default=8)
    pr.set_defaults(func=cmd_predict)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point."""
    args = build_parser().parse_args(argv)
    _setup_logging(args.verbose)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
