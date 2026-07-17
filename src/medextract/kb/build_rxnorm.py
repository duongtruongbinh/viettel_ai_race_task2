"""Build the RxNorm term table from RXNCONSO (CSV or RRF).

Input:  data/kb/raw/rxnorm_rxnconso.csv   (RXNCONSO columns, comma-delimited)
        or a classic pipe-delimited RXNCONSO.RRF.
Output: data/kb/processed/rxnorm_terms.parquet  with columns rxcui, name, tty.

We keep English RXNORM-source atoms of term types IN / SCDC / SCD / SBD — the
ingredient / clinical-drug-component / clinical-drug / branded-drug granularities
that cover the drug mentions in the notes (mostly international ingredient names,
sometimes with strength).  All host-verified rxcuis (amlodipine 308135, …) have
clean SCD names under sab=RXNORM, so this filter preserves them.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

RRF_COLUMNS = [
    "rxcui", "lat", "ts", "lui", "stt", "sui", "ispref", "rxaui", "saui",
    "scui", "sdui", "sab", "tty", "code", "str", "srl", "suppress", "cvf",
]

KEEP_TTY = {"IN", "SCDC", "SCD", "SBD"}
# synonym term-types added as extra surface forms for the kept rxcuis (more names
# per code → better mention→code retrieval recall). Does not add new codes.
SYN_TTY = {"SY", "PSN", "TMSY"}
KEEP_SAB = {"RXNORM"}

RAW_RRF = Path("data/kb/raw/RXNCONSO.RRF")          # official NLM/UTS release
RAW_CSV = Path("data/kb/raw/rxnorm_rxnconso.csv")   # Kaggle CSV fallback
OUT = Path("data/kb/processed/rxnorm_terms.parquet")


def _default_raw() -> Path:
    return RAW_RRF if RAW_RRF.exists() else RAW_CSV


def _read_raw(path: Path) -> pd.DataFrame:
    import csv as _csv

    if path.suffix.lower() == ".rrf":
        # pipe-delimited, no header, trailing pipe -> extra empty col.
        # QUOTE_NONE: RRF strings can contain quote chars that aren't delimiters.
        df = pd.read_csv(
            path, sep="|", header=None, names=RRF_COLUMNS + ["_"],
            dtype=str, keep_default_na=False, encoding="utf-8",
            quoting=_csv.QUOTE_NONE, engine="c", on_bad_lines="skip",
        )
        return df[RRF_COLUMNS]
    # CSV form (has header, possibly a UTF-8 BOM on the first column name)
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")


def build(raw: Path = None, out: Path = OUT, synonyms: bool = False) -> pd.DataFrame:
    raw = raw or _default_raw()
    df = _read_raw(raw)
    df.columns = [c.strip().lower() for c in df.columns]

    base = (
        (df["lat"] == "ENG")
        & df["sab"].isin(KEEP_SAB)
        & (df["suppress"].str.upper() != "Y")
    )
    # Keep obsolete-string ('O') atoms: the host uses some (e.g. chlorpheniramine
    # 360047 is marked suppress='O' in this release yet is a gold answer).
    mask = base & df["tty"].isin(KEEP_TTY)
    kept = df.loc[mask, ["rxcui", "str", "tty"]].rename(columns={"str": "name"})
    if synonyms:
        # Add synonym surface forms (SY/PSN/TMSY) only for rxcuis already in our
        # code set — more names per code → better mention→code recall, no new
        # codes. RELABEL each synonym with its parent code's kept tty so the
        # query-time tty filter (e.g. keep only SCD) still surfaces it; otherwise
        # the retriever would drop SY rows and the expansion would be a no-op.
        rxcui2tty = dict(zip(kept["rxcui"], kept["tty"]))  # base tty per kept code
        syn_mask = base & df["tty"].isin(SYN_TTY) & df["rxcui"].isin(rxcui2tty)
        syn = df.loc[syn_mask, ["rxcui", "str"]].rename(columns={"str": "name"})
        syn["tty"] = syn["rxcui"].map(rxcui2tty)
        kept = pd.concat([kept, syn], ignore_index=True)
    kept["name"] = kept["name"].str.strip()
    kept = kept[kept["name"].str.len() > 0]
    kept = kept.drop_duplicates(subset=["rxcui", "name"]).reset_index(drop=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    kept.to_parquet(out, index=False)
    return kept


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", default=str(_default_raw()))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--synonyms", action="store_true",
                    help="add SY/PSN/TMSY surface forms for kept codes (recall)")
    args = ap.parse_args(argv)
    df = build(Path(args.raw), Path(args.out), synonyms=args.synonyms)
    print(f"rxnorm_terms: {len(df):,} rows, {df['rxcui'].nunique():,} rxcui -> {args.out}")
    print(df["tty"].value_counts().to_string())
    # quick host-code sanity
    for code in ["308135", "243670", "866436", "392085", "313782", "904475", "197527"]:
        names = df.loc[df["rxcui"] == code, "name"].tolist()
        print(f"  {code}: {names[:2]}")


if __name__ == "__main__":
    main()
