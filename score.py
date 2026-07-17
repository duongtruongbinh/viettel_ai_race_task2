#!/usr/bin/env python3
"""Score predictions against gold on the dev set (local read of the host formula).

    python score.py --pred out/dev_baseline --gold data/dev
    python score.py --pred out/dev_improved --gold data/dev -v

``--gold`` is a dir containing a ``gold/`` subdir of ``{stem}.json`` files (the
20-note dev set is laid out this way). ``--pred`` is a dir of ``{stem}.json``
predictions. Prints text / assertion / candidate / FINAL (host formula:
final = 0.3*text + 0.3*assertions + 0.4*candidates, x100).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from medextract.scoring.scorer import score_corpus_host


def _load_dir(d: Path) -> dict:
    """Load every {stem}.json in a dir into {stem: [concepts]}."""
    out = {}
    for f in sorted(d.glob("*.json")):
        data = json.loads(f.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "concepts" in data:
            data = data["concepts"]
        out[f.stem] = data
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description="Score predictions against gold (dev set).")
    p.add_argument("--pred", required=True, help="dir of prediction {stem}.json")
    p.add_argument("--gold", required=True, help="dir containing a gold/ subdir")
    p.add_argument("-v", "--verbose", action="store_true", help="print per-record scores")
    args = p.parse_args(argv)

    gold_dir = Path(args.gold)
    gold_dir = gold_dir / "gold" if (gold_dir / "gold").is_dir() else gold_dir
    golds = _load_dir(gold_dir)
    preds = _load_dir(Path(args.pred))

    score = score_corpus_host(preds, golds)
    if args.verbose:
        for r in sorted(score.per_record, key=lambda r: r.stem):
            print(f"  {r.stem:>4}  text={r.text:.3f}  assert={r.assertions:.3f}"
                  f"  (n_pred={r.n_pred} n_gold={r.n_gold})")
    print(f"text       {score.text_score:.3f}")
    print(f"assertions {score.assertions_score:.3f}")
    print(f"candidates {score.candidates_score:.3f}")
    print(f"FINAL      {score.final_score * 100:.2f}")


if __name__ == "__main__":
    main()
