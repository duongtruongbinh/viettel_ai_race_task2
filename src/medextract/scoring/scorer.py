"""Local scorer for Task 2.

    final_score = 0.3·text_score + 0.3·assertions_score + 0.4·candidates_score

⚠️  THIS IS OUR BEST READING OF THE HOST FORMULA, reconstructed from the dev-set
README (data/dev_data/README.md), *not* the official grader.  It MUST be
reconciled with the organizers' script when released.  All the implementation
choices the host left unspecified are localized here and flagged inline so they
are easy to swap:

  * Concept matching:  a prediction matches a gold concept iff **same type** and
    **overlapping char span** (greedy by overlap).  So right-text/wrong-type is a
    brand-new concept that scores 0 everywhere (it can't match its gold twin).
  * "i" indexes **records** (files), not individual concepts:
      text_score       = mean_i (1 − WER(i))
      assertions_score = mean_i J_assertions(i)
      candidates_score = Σ_i J_cand(i)·w(i) / Σ_i w(i),  w(i)=Σ_gold_dx/drug (n_codes+1)
  * Within a record, assertion/candidate Jaccard is over whole-record sets
    {(concept_id, label)} / {(concept_id, code)} keyed by matched id — a spurious
    or wrong-type prediction gets a unique id, so it never intersects gold.
  * WER(i): word-level edit distance over the **concatenation** of concept texts
    (position order) ÷ reference word count; 1−WER clipped to [0,1].

Sanity guarantee: scoring gold against itself returns final_score == 1.0.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

try:
    import jiwer

    _HAS_JIWER = True
except Exception:  # pragma: no cover - jiwer optional at import
    _HAS_JIWER = False

from ..schema import ASSERTABLE_TYPES, CANDIDATE_TYPES

WEIGHTS = {"text": 0.3, "assertions": 0.3, "candidates": 0.4}


# ---- helpers ----------------------------------------------------------------
def _jaccard(pred: set, gold: set) -> float:
    """Jaccard with host edge cases: ∅,∅→1; ∅,X or X,∅→0; else |∩|/|∪|."""
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    return len(pred & gold) / len(pred | gold)


def _span(c: dict) -> Tuple[int, int]:
    s, e = c["position"]
    return int(s), int(e)


def _overlap(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def match_concepts(
    pred: Sequence[dict], gold: Sequence[dict]
) -> Tuple[Dict[int, int], List[int], List[int]]:
    """Greedily match pred→gold by (same type, max char overlap).

    Returns ``(pred_idx -> gold_idx, unmatched_pred_idx, unmatched_gold_idx)``.
    """
    pairs = []  # (overlap, pred_i, gold_j)
    for pi, p in enumerate(pred):
        ps, pt = _span(p), p.get("type")
        for gj, g in enumerate(gold):
            if g.get("type") != pt:
                continue
            ov = _overlap(ps, _span(g))
            if ov > 0:
                pairs.append((ov, pi, gj))
    pairs.sort(key=lambda x: (-x[0], x[1], x[2]))

    matched: Dict[int, int] = {}
    used_pred, used_gold = set(), set()
    for _, pi, gj in pairs:
        if pi in used_pred or gj in used_gold:
            continue
        matched[pi] = gj
        used_pred.add(pi)
        used_gold.add(gj)
    unmatched_pred = [i for i in range(len(pred)) if i not in used_pred]
    unmatched_gold = [j for j in range(len(gold)) if j not in used_gold]
    return matched, unmatched_pred, unmatched_gold


# ---- per-record scoring -----------------------------------------------------
def _wer_concat(pred: Sequence[dict], gold: Sequence[dict]) -> float:
    """1 − WER over concatenated concept texts (position order), clipped [0,1]."""
    def concat(cs):
        cs = sorted(cs, key=lambda c: (c["position"][0], c["position"][1]))
        return " ".join(c["text"] for c in cs).strip()

    ref, hyp = concat(gold), concat(pred)
    ref_words = ref.split()
    if not ref_words:
        # no gold text: perfect only if we also emitted nothing
        return 1.0 if not hyp.split() else 0.0
    if not _HAS_JIWER:
        raise RuntimeError("jiwer is required for text_score; pip install jiwer")
    wer = jiwer.wer(ref, hyp)
    return max(0.0, min(1.0, 1.0 - wer))


def _concept_ids(pred, gold, matched):
    """Assign a shared id to matched pairs; unique ids to unmatched concepts."""
    pred_id = {}
    for pi in range(len(pred)):
        pred_id[pi] = ("g", matched[pi]) if pi in matched else ("p", pi)
    gold_id = {gj: ("g", gj) for gj in range(len(gold))}
    return pred_id, gold_id


def _assertion_sets(pred, gold, pred_id, gold_id):
    ap = {
        (pred_id[pi], lab)
        for pi, c in enumerate(pred)
        if c.get("type") in ASSERTABLE_TYPES
        for lab in c.get("assertions", []) or []
    }
    ag = {
        (gold_id[gj], lab)
        for gj, c in enumerate(gold)
        if c.get("type") in ASSERTABLE_TYPES
        for lab in c.get("assertions", []) or []
    }
    return ap, ag


def _candidate_sets(pred, gold, pred_id, gold_id):
    cp = {
        (pred_id[pi], code)
        for pi, c in enumerate(pred)
        if c.get("type") in CANDIDATE_TYPES
        for code in c.get("candidates", []) or []
    }
    cg = {
        (gold_id[gj], code)
        for gj, c in enumerate(gold)
        if c.get("type") in CANDIDATE_TYPES
        for code in c.get("candidates", []) or []
    }
    return cp, cg


def _candidate_weight(gold: Sequence[dict]) -> int:
    """w = Σ over gold dx/drug concepts of (n_gold_codes + 1)."""
    return sum(
        len(c.get("candidates", []) or []) + 1
        for c in gold
        if c.get("type") in CANDIDATE_TYPES
    )


@dataclass
class RecordScore:
    stem: str
    text: float
    assertions: float
    candidates: float
    cand_weight: int
    n_pred: int
    n_gold: int


def score_record(stem: str, pred: Sequence[dict], gold: Sequence[dict]) -> RecordScore:
    matched, _, _ = match_concepts(pred, gold)
    pred_id, gold_id = _concept_ids(pred, gold, matched)

    text = _wer_concat(pred, gold)
    ap, ag = _assertion_sets(pred, gold, pred_id, gold_id)
    cp, cg = _candidate_sets(pred, gold, pred_id, gold_id)

    return RecordScore(
        stem=stem,
        text=text,
        assertions=_jaccard(ap, ag),
        candidates=_jaccard(cp, cg),
        cand_weight=_candidate_weight(gold),
        n_pred=len(pred),
        n_gold=len(gold),
    )


# ---- corpus aggregation -----------------------------------------------------
@dataclass
class Score:
    text_score: float
    assertions_score: float
    candidates_score: float
    final_score: float
    per_record: List[RecordScore] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "text_score": self.text_score,
            "assertions_score": self.assertions_score,
            "candidates_score": self.candidates_score,
            "final_score": self.final_score,
            "per_record": [r.__dict__ for r in self.per_record],
        }


def _wer_one(gold_text: str, pred_text: str) -> float:
    if not _HAS_JIWER:
        raise RuntimeError("jiwer required; pip install jiwer")
    ref = gold_text.strip()
    if not ref.split():
        return 0.0 if not pred_text.strip().split() else 1.0
    return jiwer.wer(ref, pred_text.strip())


def score_corpus_host(
    preds: Dict[str, list], golds: Dict[str, list], stems: Optional[Sequence[str]] = None
) -> Score:
    """Host-aligned scorer: per-concept WER/Jaccard with the spurious
    double-count penalty, aggregated globally over concepts.

    This is a local read of the official formula: a spurious (unmatched)
    prediction is penalised twice, so over-emission lowers every term. Absolute
    numbers won't match the grader exactly, but the ranking of two runs does.

      text  = Σ_matched (1−WER) / (n_gold + 2·n_spurious)
      assn  = Σ_matched J_assert / (n_gold_assertable + 2·n_spurious_assertable)
      cand  = Σ_matched J_cand·w / (Σ_gold w + 2·Σ_spurious w),  w = n_codes+1
    """
    if stems is None:
        stems = sorted(golds, key=lambda s: (0, int(s)) if s.isdigit() else (1, s))

    t_num = t_den = a_num = a_den = c_num = c_den = 0.0
    per_record: List[RecordScore] = []
    for s in stems:
        gold, pred = golds[s], preds.get(s, [])
        matched, up, ug = match_concepts(pred, gold)

        rt_num = 0.0
        for pi, gj in matched.items():
            rt_num += max(0.0, 1.0 - _wer_one(gold[gj]["text"], pred[pi]["text"]))
        rt_den = len(gold) + 2 * len(up)
        t_num += rt_num
        t_den += rt_den

        # assertions (assertable types)
        ga = [j for j in range(len(gold)) if gold[j].get("type") in ASSERTABLE_TYPES]
        spa = [i for i in up if pred[i].get("type") in ASSERTABLE_TYPES]
        ra_num = 0.0
        for pi, gj in matched.items():
            if gold[gj].get("type") in ASSERTABLE_TYPES:
                ra_num += _jaccard(set(pred[pi].get("assertions", []) or []),
                                   set(gold[gj].get("assertions", []) or []))
        a_num += ra_num
        a_den += len(ga) + 2 * len(spa)

        # candidates (candidate types), weighted by n_gold_codes+1
        for pi, gj in matched.items():
            if gold[gj].get("type") in CANDIDATE_TYPES:
                w = len(gold[gj].get("candidates", []) or []) + 1
                c_num += _jaccard(set(pred[pi].get("candidates", []) or []),
                                  set(gold[gj].get("candidates", []) or [])) * w
                c_den += w
        for gj in ug:
            if gold[gj].get("type") in CANDIDATE_TYPES:
                c_den += len(gold[gj].get("candidates", []) or []) + 1
        for pi in up:
            if pred[pi].get("type") in CANDIDATE_TYPES:
                c_den += 2 * (len(pred[pi].get("candidates", []) or []) + 1)

        per_record.append(RecordScore(
            stem=s, text=(rt_num / rt_den if rt_den else 1.0),
            assertions=(ra_num / (len(ga) + 2 * len(spa)) if (len(ga) + 2 * len(spa)) else 1.0),
            candidates=0.0, cand_weight=0, n_pred=len(pred), n_gold=len(gold)))

    text = t_num / t_den if t_den else 1.0
    assn = a_num / a_den if a_den else 1.0
    cand = c_num / c_den if c_den else 1.0
    final = WEIGHTS["text"] * text + WEIGHTS["assertions"] * assn + WEIGHTS["candidates"] * cand
    return Score(text, assn, cand, final, per_record)


def score_corpus(
    preds: Dict[str, list], golds: Dict[str, list], stems: Optional[Sequence[str]] = None
) -> Score:
    """Aggregate per-record scores. Missing prediction for a gold stem == []."""
    if stems is None:
        stems = sorted(golds, key=lambda s: (0, int(s)) if s.isdigit() else (1, s))

    records = [score_record(s, preds.get(s, []), golds[s]) for s in stems]

    n = len(records) or 1
    text_score = sum(r.text for r in records) / n
    assertions_score = sum(r.assertions for r in records) / n

    wsum = sum(r.cand_weight for r in records)
    candidates_score = (
        sum(r.candidates * r.cand_weight for r in records) / wsum if wsum else 1.0
    )

    final = (
        WEIGHTS["text"] * text_score
        + WEIGHTS["assertions"] * assertions_score
        + WEIGHTS["candidates"] * candidates_score
    )
    return Score(text_score, assertions_score, candidates_score, final, records)
