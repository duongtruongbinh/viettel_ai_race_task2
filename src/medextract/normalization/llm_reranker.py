"""LLM candidate reranker: the LLM picks the exact code from the retriever top-k.

The bi-encoder retrieves by string similarity, which places the right code in the
top-k but not always at rank 1. The LLM reasons about which RxNorm/ICD variant
actually matches the mention in context. It is constrained to the codes the
retriever surfaced, so it can only return valid ontology codes — never a
hallucinated one.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Dict, List, Optional

_ICD_CATEGORY = re.compile(r"^[A-Z]\d\d$")  # bare 3-char category, e.g. E11

from ..llm.prompt import render
from ..schema import Span
from .base import Normalizer
from .retriever import RetrieverNormalizer

log = logging.getLogger("medextract.llm_reranker")

_JSON_ARRAY = re.compile(r"\[.*?\]", re.DOTALL)


def _parse_codes(text: str, valid: set) -> List[str]:
    """Extract a JSON array of codes from the LLM output, keep only valid ones."""
    m = _JSON_ARRAY.search(text)
    if not m:
        return []
    try:
        arr = json.loads(m.group(0))
    except Exception:
        # fall back: grab quoted tokens
        arr = re.findall(r'"([^"]+)"', m.group(0))
    out: List[str] = []
    for c in arr:
        c = str(c).strip()
        if c in valid and c not in out:
            out.append(c)
    return out


class LLMRerankNormalizer(Normalizer):
    def __init__(
        self,
        retriever: RetrieverNormalizer,
        engine,
        retrieve_k: int = 12,
        max_candidates: Optional[Dict[str, int]] = None,
        icd_codes: Optional[set] = None,
        max_new_tokens: int = 48,
    ):
        self.retriever = retriever
        self.engine = engine
        self.retrieve_k = retrieve_k
        self.max_candidates = max_candidates or {"ICD10": 2, "RXNORM": 1}
        # full ICD code set (for the category -> .9 leaf remap)
        self.icd_codes = icd_codes or set()
        self.max_new_tokens = max_new_tokens

    def _leaf_ify(self, codes: List[str]) -> List[str]:
        """Remap a bare 3-char ICD category to its ``.9`` (unspecified) leaf when
        that leaf exists in the KB. Such categories are non-terminal, so gold
        (always a leaf) is never the bare code — this can only turn a 0 into a 1."""
        out = []
        for c in codes:
            if _ICD_CATEGORY.match(c) and f"{c}.9" in self.icd_codes:
                out.append(f"{c}.9")
            else:
                out.append(c)
        return out

    @staticmethod
    def _sentence(text: str, span: Span) -> str:
        s, e, _ = span
        ls = text.rfind("\n", 0, s) + 1
        le = text.find("\n", e)
        le = le if le != -1 else len(text)
        return " ".join(text[ls:le].split())[:300]

    def predict(self, text: str, span: Span) -> List[str]:
        kb, mention, hits = self.retriever.retrieve(text, span, top_k=self.retrieve_k)
        if kb is None or not hits or not mention:
            return []
        # dedup candidates by code, keep first (best-retrieved) name
        seen, cands = set(), []
        for code, name, _s, _tty in hits:
            if code not in seen:
                seen.add(code)
                cands.append({"code": code, "name": name})
        # ICD: drop a bare 3-char category code when a sub-code of it is present,
        # so the LLM must choose a specific leaf (gold is virtually always a leaf).
        if kb == "ICD10":
            leaves = {c["code"].split(".")[0] for c in cands if "." in c["code"]}
            cands = [c for c in cands
                     if not (_ICD_CATEGORY.match(c["code"]) and c["code"] in leaves)]
        cap = self.max_candidates.get(kb, 2)
        prompt = render("candidate_rerank", mention=mention, kb=kb,
                        sentence=self._sentence(text, span), candidates=cands, max_codes=cap)
        try:
            out = self.engine.complete(prompt, max_new_tokens=self.max_new_tokens)
        except Exception as e:  # pragma: no cover - soft-fail to retriever top-1
            log.warning("LLM rerank failed (%s); falling back to retriever", e)
            return [cands[0]["code"]] if cands else []
        codes = _parse_codes(out, seen)
        # When the LLM returns nothing it means "no confident match" — we keep it
        # empty rather than forcing the retriever top-1 (empty is usually correct).
        codes = codes[:cap]
        if kb == "ICD10":
            codes = self._leaf_ify(codes)
        return codes


def from_config(cfg: dict, engine=None) -> Normalizer:
    from ..llm.engine import LLMEngine
    from .retriever import from_config as build_retriever

    retriever = build_retriever(cfg)
    n = (cfg or {}).get("normalization", {}) or {}
    lc = n.get("llm_rerank", {}) or {}
    # full ICD code set for the category -> .9 leaf remap
    icd_codes = set()
    try:
        import pandas as pd
        icd_codes = set(pd.read_parquet("data/kb/processed/icd_terms.parquet")["code"].astype(str))
    except Exception as e:  # pragma: no cover
        log.warning("could not load ICD code set for leaf remap (%s)", e)
    if engine is None:
        lcfg = (cfg or {}).get("llm", {}) or {}
        engine = LLMEngine(
            model_name=lcfg.get("model", "/mnt/pretrained_fm/Qwen_Qwen3-8B"),
            device=lcfg.get("device", "wait"),
            dtype=lcfg.get("dtype", "bfloat16"),
            min_free_gb=lcfg.get("min_free_gb", 18.0),
            enable_thinking=lcfg.get("enable_thinking", False),
            load_in_4bit=lcfg.get("load_in_4bit", False),
        )
    return LLMRerankNormalizer(
        retriever=retriever,
        engine=engine,
        retrieve_k=lc.get("retrieve_k", 12),
        max_candidates=lc.get("max_candidates", n.get("max_candidates", {"ICD10": 2, "RXNORM": 1})),
        icd_codes=icd_codes,
        max_new_tokens=lc.get("max_new_tokens", 48),
    )
