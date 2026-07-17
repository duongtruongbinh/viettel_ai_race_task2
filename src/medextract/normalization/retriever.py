"""Bi-encoder + FAISS retriever: a mention -> ranked ontology codes.

This is the linking (entity-normalization) step. A CHẨN_ĐOÁN (diagnosis) span is
queried against the ICD-10 index; a THUỐC (drug) span against the RxNorm index.
SapBERT embeds the mention, FAISS returns the nearest concept names, and we keep
the codes above a similarity cutoff.

Two small, general refinements matter for this task:
* Drugs: strip route/frequency tokens ("po", "bid") but keep the strength, since
  the gold RxNorm code is the strength-specific clinical drug (e.g. "amlodipine
  10 mg" -> "Amlodipine 10 MG Oral Tablet"). A term-type (tty) filter keeps only
  those clinical-drug (SCD) hits.
* Candidates are capped low per KB: the candidate score is a Jaccard, so one
  spurious extra code on a single-answer concept halves the score.
"""
from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional

from ..schema import CANDIDATE_TYPES, Span
from .base import Normalizer

log = logging.getLogger("medextract.retriever")

# route / frequency tokens to strip from drug mentions (the strength is KEPT)
_DRUG_ROUTE_FREQ = re.compile(
    r"\b(po|iv|im|sc|sl|pr|tid|bid|qd|qid|qhs|qam|qpm|prn|q\d+h(:prn)?|daily|"
    r"uống|tiêm|lần|ngày)\b",
    re.IGNORECASE,
)


class RetrieverNormalizer(Normalizer):
    def __init__(
        self,
        indexes: Dict[str, "object"],  # kb -> KBIndex
        top_k: int = 10,
        cutoffs: Optional[Dict[str, float]] = None,
        max_candidates: Optional[Dict[str, int]] = None,
        strip_drug_noise: bool = True,
        prefer_tty: Optional[Dict[str, List[str]]] = None,
        filter_tty: Optional[Dict[str, List[str]]] = None,
    ):
        self.indexes = indexes
        self.top_k = top_k
        self.cutoffs = cutoffs or {"ICD10": 0.70, "RXNORM": 0.80}
        self.max_candidates = max_candidates or {"ICD10": 2, "RXNORM": 1}
        self.strip_drug_noise = strip_drug_noise
        self.prefer_tty = prefer_tty or {}
        self.filter_tty = filter_tty or {}

    def _clean_mention(self, mention: str, kb: str) -> str:
        m = " ".join(mention.split())
        if kb == "RXNORM" and self.strip_drug_noise:
            m2 = " ".join(_DRUG_ROUTE_FREQ.sub(" ", m).split())
            if len(m2) >= 3:
                m = m2
        return m

    def retrieve(self, text: str, span: Span, top_k: Optional[int] = None):
        """Return ``(kb, mention, hits)`` where hits are tty-filtered
        ``(code, name, score, tty)`` tuples, before the cutoff/cap in ``predict``.
        Shared by :meth:`predict` and the LLM reranker.
        """
        start, end, typ = span
        kb = CANDIDATE_TYPES.get(typ)
        if kb is None or kb not in self.indexes:
            return None, "", []
        mention = self._clean_mention(text[start:end], kb)
        if not mention:
            return kb, "", []
        hits = self.indexes[kb].query(mention, top_k=top_k or self.top_k)
        keep_tty = self.filter_tty.get(kb)
        if keep_tty:
            hits = [h for h in hits if h[3] in keep_tty]
        return kb, mention, hits

    def predict(self, text: str, span: Span) -> List[str]:
        kb, mention, hits = self.retrieve(text, span)
        if kb is None or not hits:
            return []
        cutoff = self.cutoffs.get(kb, 0.0)
        hits = [h for h in hits if h[2] >= cutoff]
        if not hits:
            return []

        # prefer a specific term type (e.g. SCD for drugs) when a preferred hit is
        # within a small score margin of the top hit
        preferred = self.prefer_tty.get(kb)
        if preferred:
            top_score = hits[0][2]
            for h in hits:
                if h[3] in preferred and (top_score - h[2]) <= 0.05:
                    hits = [h] + [x for x in hits if x is not h]
                    break

        cap = self.max_candidates.get(kb, 2)
        codes: List[str] = []
        for code, _name, _score, _tty in hits:
            if code not in codes:
                codes.append(code)
            if len(codes) >= cap:
                break
        return codes


def from_config(cfg: dict) -> RetrieverNormalizer:
    """Build a retriever, loading FAISS indexes + the SapBERT encoder from config."""
    from ..kb.index import KBIndex, SapBERTEncoder

    n = (cfg or {}).get("normalization", {}) or {}
    kb_cfg = (cfg or {}).get("kb", {}) or {}
    model = kb_cfg.get("sapbert_model", "cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR")
    device = kb_cfg.get("device", "auto")

    encoder = SapBERTEncoder(model_name=model, device=device)
    indexes = {}
    for kb in CANDIDATE_TYPES.values():
        try:
            indexes[kb] = KBIndex.load(kb, encoder=encoder)
        except Exception as e:
            log.warning("could not load %s index (%s); that KB will return []", kb, e)

    return RetrieverNormalizer(
        indexes=indexes,
        top_k=n.get("top_k", 10),
        cutoffs=n.get("cutoffs", {"ICD10": 0.70, "RXNORM": 0.80}),
        max_candidates=n.get("max_candidates", {"ICD10": 2, "RXNORM": 1}),
        strip_drug_noise=n.get("strip_drug_noise", True),
        prefer_tty=n.get("prefer_tty", {}),
        filter_tty=n.get("filter_tty", {}),
    )
