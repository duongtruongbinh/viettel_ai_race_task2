"""Scorer tests, incl. the required gold-vs-gold == 1.0 sanity check."""
from pathlib import Path

from medextract.io_utils import load_gold_dir
from medextract.scoring.scorer import (
    _jaccard,
    match_concepts,
    score_corpus,
    score_record,
)

GOLD_DIR = Path(__file__).resolve().parents[1] / "data" / "dev" / "gold"


def test_jaccard_edge_cases():
    assert _jaccard(set(), set()) == 1.0
    assert _jaccard(set(), {1}) == 0.0
    assert _jaccard({1}, set()) == 0.0
    assert _jaccard({1, 2}, {2, 3}) == 1 / 3


def test_gold_vs_gold_is_one():
    golds = load_gold_dir(GOLD_DIR)
    score = score_corpus(golds, golds)
    assert abs(score.final_score - 1.0) < 1e-9, score.as_dict()
    assert abs(score.text_score - 1.0) < 1e-9
    assert abs(score.assertions_score - 1.0) < 1e-9
    assert abs(score.candidates_score - 1.0) < 1e-9


def test_empty_pred_scores_low():
    golds = load_gold_dir(GOLD_DIR)
    empty = {k: [] for k in golds}
    score = score_corpus(empty, golds)
    assert score.final_score < 0.5


def test_wrong_type_does_not_match():
    gold = [{"text": "sốt", "type": "TRIỆU_CHỨNG", "position": [0, 3], "assertions": ["isNegated"]}]
    pred = [{"text": "sốt", "type": "CHẨN_ĐOÁN", "position": [0, 3], "assertions": ["isNegated"], "candidates": []}]
    matched, up, ug = match_concepts(pred, gold)
    assert matched == {}  # different type -> no match


def test_span_overlap_matches():
    gold = [{"text": "sốt cao", "type": "TRIỆU_CHỨNG", "position": [0, 7], "assertions": []}]
    pred = [{"text": "sốt", "type": "TRIỆU_CHỨNG", "position": [0, 3], "assertions": []}]
    matched, _, _ = match_concepts(pred, gold)
    assert matched == {0: 0}


def test_spurious_concept_hurts_assertions():
    gold = [{"text": "sốt", "type": "TRIỆU_CHỨNG", "position": [0, 3], "assertions": ["isNegated"]}]
    # correct concept + one spurious symptom with an assertion
    pred = [
        {"text": "sốt", "type": "TRIỆU_CHỨNG", "position": [0, 3], "assertions": ["isNegated"]},
        {"text": "ho", "type": "TRIỆU_CHỨNG", "position": [10, 12], "assertions": ["isNegated"]},
    ]
    r = score_record("x", pred, gold)
    assert r.assertions < 1.0  # spurious (id ("p",1)) never intersects gold


def test_host_scorer_penalizes_spurious():
    from medextract.scoring.scorer import score_corpus_host
    gold = {"1": [{"text": "sốt", "type": "TRIỆU_CHỨNG", "position": [0, 3], "assertions": []}]}
    text = "sốt cao ho khan"
    clean = {"1": [{"text": "sốt", "type": "TRIỆU_CHỨNG", "position": [0, 3], "assertions": []}]}
    noisy = {"1": clean["1"] + [
        {"text": "cao", "type": "TRIỆU_CHỨNG", "position": [4, 7], "assertions": []},
        {"text": "ho", "type": "TRIỆU_CHỨNG", "position": [8, 10], "assertions": []},
    ]}
    s_clean = score_corpus_host(clean, gold).text_score
    s_noisy = score_corpus_host(noisy, gold).text_score
    assert s_clean == 1.0 and s_noisy < s_clean  # spurious concepts drag text down
