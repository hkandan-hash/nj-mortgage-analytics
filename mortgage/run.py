"""Pipeline entry point: python -m mortgage.run [ingest|clean|eda|regression|classification|export|all]"""
from __future__ import annotations

import argparse
import logging

STEPS = ["ingest", "clean", "eda", "regression", "classification", "export"]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="New Jersey mortgage lending analytics")
    parser.add_argument("step", choices=STEPS + ["all"])
    parser.add_argument("--force", action="store_true", help="re-download raw files")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    steps = STEPS if args.step == "all" else [args.step]

    if "ingest" in steps:
        from .ingest import ingest
        ingest(force=args.force)
    if "clean" in steps:
        from .clean import clean
        clean()
    if "eda" in steps:
        from .eda import run_eda
        run_eda()
    if "regression" in steps:
        from .regression import run_regression
        run_regression()
    if "classification" in steps:
        from .classification import run_classification
        run_classification()
    if "export" in steps:
        from .export import run_export
        run_export()


if __name__ == "__main__":
    main()
