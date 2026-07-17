"""I/O helpers: read inputs, write/validate/zip submission, gold adapters.

The writer *guarantees* ``input[start:end] == concept["text"]`` (WER depends on
it), UTF-8 with ``ensure_ascii=False``, and stable ordering by position.
"""
from __future__ import annotations

import json
import os
import zipfile
from pathlib import Path
from typing import Dict, List

from .schema import Concept, validate_output


def read_text(path: os.PathLike | str) -> str:
    """Read one input record verbatim (offsets are into *this* exact string)."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def list_inputs(in_dir: os.PathLike | str) -> List[Path]:
    """Return input ``*.txt`` files sorted by numeric stem when possible."""
    paths = list(Path(in_dir).glob("*.txt"))

    def key(p: Path):
        stem = p.stem
        return (0, int(stem)) if stem.isdigit() else (1, stem)

    return sorted(paths, key=key)


def write_json(path: os.PathLike | str, concepts: List[Concept]) -> None:
    """Write one record's concepts as UTF-8 JSON (ensure_ascii=False)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(concepts, f, ensure_ascii=False, indent=2)


def read_json(path: os.PathLike | str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_record(out_dir: os.PathLike | str, stem: str, concepts: list, text: str) -> List[Concept]:
    """Validate + clean concepts for one record and write ``<stem>.json``.

    Returns the cleaned concepts.  Raises on schema violations so bad output
    never reaches a submission silently.
    """
    cleaned = validate_output(concepts, text)
    write_json(Path(out_dir) / f"{stem}.json", cleaned)
    return cleaned


def zip_submission(out_dir: os.PathLike | str, zip_name: str = "submission.zip") -> Path:
    """Zip all ``*.json`` in ``out_dir`` into ``out_dir/zip_name`` (flat)."""
    out_dir = Path(out_dir)
    zip_path = out_dir / zip_name
    jsons = sorted(out_dir.glob("*.json"))
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for jp in jsons:
            zf.write(jp, arcname=jp.name)
    return zip_path


def load_gold_dir(gold_dir: os.PathLike | str) -> Dict[str, list]:
    """Load gold annotations ``<stem>.json`` -> list[concept].

    The dev gold already follows the submission schema, so this is a passthrough
    keyed by file stem.  If a future gold format diverges, adapt it here.
    """
    gold: Dict[str, list] = {}
    for jp in sorted(Path(gold_dir).glob("*.json")):
        gold[jp.stem] = read_json(jp)
    return gold
