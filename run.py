#!/usr/bin/env python3
"""Run a medextract pipeline over an input dir -> per-file JSON (+ optional zip).

    # quick demo on the bundled samples
    python run.py --config configs/baseline.yaml --input data/sample_input --output out

    # full submission (zipped) with the improved pipeline
    python run.py --config configs/improved.yaml --input <test dir> --output sub --zip
"""
from __future__ import annotations

import argparse
import logging

from medextract import set_seed
from medextract.config import load_config


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Run a medextract pipeline over an input dir.")
    p.add_argument("--config", required=True,
                   help="config YAML (configs/baseline.yaml | configs/improved.yaml)")
    p.add_argument("--input", required=True, help="dir of *.txt inputs")
    p.add_argument("--output", required=True, help="dir to write per-file JSON")
    p.add_argument("--zip", action="store_true", help="also build submission.zip")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    set_seed(args.seed)

    # import after logging/seed so heavy modules initialise with the seed set
    from medextract.pipeline import build_pipeline

    config = load_config(args.config)
    config.setdefault("seed", args.seed)
    pipeline = build_pipeline(config)
    pipeline.run_dir(args.input, args.output, zip_it=args.zip)
    logging.getLogger("run").info("done: %s -> %s", args.config, args.output)


if __name__ == "__main__":
    main()
