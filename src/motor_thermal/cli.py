"""Command-line interface: ``uv run motor-thermal <command>``.

Commands: data, eda, train, evaluate, forecast, report, all.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from . import data as data_mod
from .config import ARTIFACTS_DIR, MODES, load_config


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def cmd_data(args: argparse.Namespace, cfg: dict) -> int:
    path = data_mod.locate_dataset(cfg, args.csv)
    df = data_mod.load_measurements(path, cfg)
    rep = data_mod.validate_schema(df, cfg)
    prov = data_mod.build_provenance(path, df, cfg)
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    (ARTIFACTS_DIR / "provenance.json").write_text(json.dumps(prov, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "provenance": prov,
                "facts": rep.facts,
                "warnings": rep.warnings,
                "errors": rep.errors,
            },
            indent=2,
        )
    )
    return 0 if rep.ok else 1


def cmd_eda(args: argparse.Namespace, cfg: dict) -> int:
    from .eda import run_eda

    run_eda(cfg, args.csv)
    return 0


def cmd_train(args: argparse.Namespace, cfg: dict) -> int:
    from .train import run_training

    run_training(cfg, args.mode, args.csv)
    return 0


def cmd_evaluate(args: argparse.Namespace, cfg: dict) -> int:
    from .evaluate import run_evaluation

    run_evaluation(cfg, args.mode, args.csv)
    return 0


def cmd_forecast(args: argparse.Namespace, cfg: dict) -> int:
    from .forecast import run_forecast

    run_forecast(cfg, args.mode, args.csv)
    return 0


def cmd_report(args: argparse.Namespace, cfg: dict) -> int:
    from .report import build_all

    build_all(cfg, args.mode)
    return 0


def cmd_all(args: argparse.Namespace, cfg: dict) -> int:
    for fn in (cmd_data, cmd_eda, cmd_train, cmd_evaluate, cmd_forecast, cmd_report):
        rc = fn(args, cfg)
        if rc:
            return rc
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="motor-thermal", description=__doc__)
    p.add_argument(
        "--config", default=None, help="path to YAML config (default configs/default.yaml)"
    )
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)
    for name, fn, help_ in (
        ("data", cmd_data, "download/locate the dataset, validate it, write provenance"),
        ("eda", cmd_eda, "exploratory data analysis figures + summary"),
        ("train", cmd_train, "compare candidates, ablate history, tune, select (never reads test)"),
        ("evaluate", cmd_evaluate, "final held-out test evaluation, plots, residual monitoring"),
        ("forecast", cmd_forecast, "optional 60 s ahead forecasting experiment"),
        ("report", cmd_report, "generate report, architecture diagram and PowerPoint"),
        ("all", cmd_all, "run the whole pipeline"),
    ):
        sp = sub.add_parser(name, help=help_)
        sp.add_argument("--csv", default=None, help="manual path to measures_v2.csv")
        if name not in ("data", "eda"):
            sp.add_argument("--mode", choices=MODES, default="full")
        sp.set_defaults(func=fn)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _setup_logging(args.verbose)
    cfg = load_config(args.config)
    try:
        return args.func(args, cfg)
    except data_mod.DatasetUnavailableError as exc:
        print(f"\nBLOCKED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
