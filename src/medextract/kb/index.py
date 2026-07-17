"""SapBERT embedding + FAISS index per KB, with query().

Encoder: multilingual SapBERT (``cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR``),
[CLS] pooling (the SapBERT convention), L2-normalized → cosine == inner product.
One FAISS ``IndexFlatIP`` per KB.  Building is a one-time offline step (GPU if
free, else CPU); querying is CPU-friendly.

Artifacts written to ``data/kb/processed/<kb>.faiss`` + ``<kb>_meta.parquet``
(row-aligned ``id`` = ICD code / RxNorm rxcui, plus ``name``).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger("medextract.kb.index")

PROCESSED = Path("data/kb/processed")
DEFAULT_MODEL = "cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR"

# KB name -> (terms parquet, id column)
KB_SOURCES = {
    "ICD10": ("icd_terms.parquet", "code"),
    "RXNORM": ("rxnorm_terms.parquet", "rxcui"),
}


class SapBERTEncoder:
    def __init__(self, model_name: str = DEFAULT_MODEL, device: str = "auto",
                 max_length: int = 32, batch_size: int = 256):
        import torch
        from transformers import AutoModel, AutoTokenizer

        from ..utils.gpu import resolve_device

        self.device = resolve_device(device)
        self.max_length = max_length
        self.batch_size = batch_size
        log.info("loading SapBERT %s on %s", model_name, self.device)
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        self._torch = torch

    def encode(self, texts: Sequence[str], show_progress: bool = False) -> np.ndarray:
        torch = self._torch
        vecs = []
        n = len(texts)
        for i in range(0, n, self.batch_size):
            batch = list(texts[i:i + self.batch_size])
            enc = self.tok(batch, padding=True, truncation=True,
                           max_length=self.max_length, return_tensors="pt").to(self.device)
            with torch.no_grad():
                out = self.model(**enc)
            cls = out.last_hidden_state[:, 0, :]  # [CLS]
            cls = torch.nn.functional.normalize(cls, p=2, dim=1)
            vecs.append(cls.cpu().numpy().astype("float32"))
            if show_progress and (i // self.batch_size) % 20 == 0:
                log.info("  encoded %d/%d", min(i + self.batch_size, n), n)
        return np.vstack(vecs) if vecs else np.zeros((0, self.model.config.hidden_size), "float32")


class KBIndex:
    """A FAISS inner-product index over one KB's term names."""

    def __init__(self, kb: str, index, meta: pd.DataFrame, encoder: Optional[SapBERTEncoder] = None):
        self.kb = kb
        self.index = index
        self.meta = meta  # columns: id, name
        self.encoder = encoder

    # -- build / io -----------------------------------------------------------
    @staticmethod
    def build(kb: str, encoder: SapBERTEncoder, processed: Path = PROCESSED) -> "KBIndex":
        import faiss

        fname, id_col = KB_SOURCES[kb]
        df = pd.read_parquet(processed / fname)
        keep = ["id", "name"] + (["tty"] if "tty" in df.columns else [])
        meta = df.rename(columns={id_col: "id"})[keep].reset_index(drop=True)
        log.info("[%s] embedding %d names", kb, len(meta))
        emb = encoder.encode(meta["name"].tolist(), show_progress=True)
        index = faiss.IndexFlatIP(emb.shape[1])
        index.add(emb)
        return KBIndex(kb, index, meta, encoder)

    def save(self, processed: Path = PROCESSED) -> None:
        import faiss

        processed.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(processed / f"{self.kb}.faiss"))
        self.meta.to_parquet(processed / f"{self.kb}_meta.parquet", index=False)

    @staticmethod
    def load(kb: str, encoder: Optional[SapBERTEncoder] = None,
             processed: Path = PROCESSED) -> "KBIndex":
        import faiss

        index = faiss.read_index(str(processed / f"{kb}.faiss"))
        meta = pd.read_parquet(processed / f"{kb}_meta.parquet")
        return KBIndex(kb, index, meta, encoder)

    # -- query ----------------------------------------------------------------
    def query(self, mention: str, top_k: int = 10) -> List[Tuple[str, str, float, str]]:
        """Return ``[(id, name, score, tty), …]`` best matches for a mention.

        ``tty`` is ``""`` for KBs without a term-type column (e.g. ICD).
        """
        if self.encoder is None:
            raise RuntimeError("KBIndex has no encoder; construct/load with one")
        q = self.encoder.encode([mention])
        scores, idx = self.index.search(q, min(top_k, self.index.ntotal))
        has_tty = "tty" in self.meta.columns
        out = []
        for j, s in zip(idx[0], scores[0]):
            if j < 0:
                continue
            row = self.meta.iloc[int(j)]
            tty = str(row["tty"]) if has_tty else ""
            out.append((str(row["id"]), str(row["name"]), float(s), tty))
        return out


def build_all(model_name: str = DEFAULT_MODEL, device: str = "auto") -> None:
    enc = SapBERTEncoder(model_name=model_name, device=device)
    for kb in KB_SOURCES:
        ix = KBIndex.build(kb, enc)
        ix.save()
        log.info("[%s] index built: %d vectors", kb, ix.index.ntotal)


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="Build SapBERT+FAISS indexes for all KBs.")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default="auto", help="auto | wait | cpu | cuda:N")
    args = ap.parse_args()
    build_all(model_name=args.model, device=args.device)
