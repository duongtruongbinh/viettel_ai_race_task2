"""Build the ICD-10 term table.

Preferred source: the official Vietnamese **QĐ 4469/QĐ-BYT** Excel dropped into
``data/kb/raw/`` (any ``*.xlsx`` with columns Mã / Mô tả tiếng Việt / Mô tả tiếng
Anh / Ghi chú).  If present, its Vietnamese names become ``name_vi`` and dominate
retrieval of Vietnamese diagnosis mentions.

Fallback (no VN xlsx available): the public **WHO ICD-10 English** list — full
code coverage, English names; multilingual SapBERT links Vietnamese mentions to
them.  (The QĐ 4469 Excel is license-gated and must be downloaded manually.)

Output: ``data/kb/processed/icd_terms.parquet`` with columns
``code, name_vi, name_en, source`` (one row per (code, name); synonyms exploded).
Always adds COVID codes U07.1 / U07.2 (QĐ 98).
"""
from __future__ import annotations

import argparse
import io
import urllib.request
from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/kb/raw")
OUT = Path("data/kb/processed/icd_terms.parquet")

# Public WHO ICD-10 (English) — dotless codes + description. Cached locally.
WHO_EN_URL = "https://raw.githubusercontent.com/u4507075/icd_10/master/result/icd10.csv"
WHO_EN_CACHE = RAW_DIR / "icd10_who_en.csv"

COVID_CODES = [
    ("U07.1", "COVID-19, vi rút được xác định", "COVID-19, virus identified"),
    ("U07.2", "COVID-19, vi rút không được xác định", "COVID-19, virus not identified"),
]


def dot_code(code: str) -> str:
    """Normalize a dotless ICD-10 code to dotted form (A001 -> A00.1)."""
    code = str(code).strip().upper().replace(".", "")
    if len(code) > 3:
        return f"{code[:3]}.{code[3:]}"
    return code


def _find_vn_xlsx() -> Path | None:
    for pat in ("*.xlsx", "*.xls"):
        for p in sorted(RAW_DIR.glob(pat)):
            return p
    return None


def _load_vn(path: Path) -> pd.DataFrame:
    """Parse the QĐ4469 Excel (.xls/.xlsx) into (code, name_vi, name_en, source).

    The sheet has a multi-column layout (chapter / group / disease). We target the
    **disease-level** columns specifically: MÃ BỆNH (dotted code), TÊN BỆNH
    (Vietnamese), DISEASE NAME (English) — not the chapter/group MÃ/TÊN columns.
    The header row isn't row 0, so we locate it by finding "MÃ BỆNH".
    """
    engine = "xlrd" if path.suffix.lower() == ".xls" else None
    raw = pd.read_excel(path, sheet_name=0, header=None, dtype=str, engine=engine).fillna("")

    # locate the header row (the one containing a 'MÃ BỆNH' cell)
    header_row = None
    for i in range(min(15, len(raw))):
        cells = [str(x).strip().upper() for x in raw.iloc[i].tolist()]
        if any(c == "MÃ BỆNH" for c in cells):
            header_row = i
            break
    if header_row is None:
        raise ValueError(f"Could not find 'MÃ BỆNH' header in {path}")

    header = [str(x).strip() for x in raw.iloc[header_row].tolist()]
    body = raw.iloc[header_row + 1:].reset_index(drop=True)
    body.columns = header

    def col(name_upper: str):
        for c in header:
            if c.strip().upper() == name_upper:
                return c
        return None

    c_code = col("MÃ BỆNH")
    c_vi = col("TÊN BỆNH")
    c_en = col("DISEASE NAME")
    if c_code is None or c_vi is None:
        raise ValueError(f"Missing MÃ BỆNH / TÊN BỆNH columns in {path}: {header}")

    out = pd.DataFrame({
        "code": body[c_code].map(dot_code),
        "name_vi": body[c_vi].astype(str).str.strip(),
        "name_en": (body[c_en].astype(str).str.strip() if c_en else ""),
        "source": "QD4469",
    })
    return out[out["code"].str.len() >= 3]


def _load_who_en(path: Path = WHO_EN_CACHE) -> pd.DataFrame:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(WHO_EN_URL, headers={"User-Agent": "curl"})
        data = urllib.request.urlopen(req, timeout=60).read()
        path.write_bytes(data)
    raw = pd.read_csv(io.BytesIO(path.read_bytes()), dtype=str, keep_default_na=False)
    # columns: index, code, cdesc
    code_col = "code" if "code" in raw.columns else raw.columns[1]
    desc_col = "cdesc" if "cdesc" in raw.columns else raw.columns[-1]
    out = pd.DataFrame({
        "code": raw[code_col].map(dot_code),
        "name_vi": "",
        "name_en": raw[desc_col].str.strip(),
        "source": "WHO-ICD10-EN",
    })
    return out[out["code"].str.len() >= 3]


def build(out: Path = OUT) -> pd.DataFrame:
    vn = _find_vn_xlsx()
    if vn is not None:
        print(f"[icd] using Vietnamese source: {vn}")
        df = _load_vn(vn)
    else:
        print("[icd] no QĐ4469 xlsx found; falling back to WHO ICD-10 (English)")
        df = _load_who_en()

    covid = pd.DataFrame(COVID_CODES, columns=["code", "name_vi", "name_en"])
    covid["source"] = "QD98-COVID"
    df = pd.concat([df, covid], ignore_index=True)

    # explode: keep a 'name' column preferring VI, and drop empty names
    df["name"] = df["name_vi"].where(df["name_vi"].str.len() > 0, df["name_en"])
    df = df[df["name"].str.len() > 0]
    df = df.drop_duplicates(subset=["code", "name"]).reset_index(drop=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    return df


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)
    df = build(Path(args.out))
    print(f"icd_terms: {len(df):,} rows, {df['code'].nunique():,} codes -> {args.out}")
    print(df["source"].value_counts().to_string())
    for code in ["K21.0", "K21.9", "I10", "E11.9", "U07.1"]:
        names = df.loc[df["code"] == code, "name"].tolist()
        print(f"  {code}: {names[:1]}")


if __name__ == "__main__":
    main()
