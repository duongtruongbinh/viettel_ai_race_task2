"""Tests for EDA-driven span post-processing."""
from medextract.ner.postprocess import clean_spans


def _clean(text, spans):
    return [(text[s:e], t) for s, e, t in clean_spans(text, spans)]


def test_strip_leading_negation():
    t = "Không sốt cao"
    assert _clean(t, [(0, len(t), "TRIỆU_CHỨNG")]) == [("sốt cao", "TRIỆU_CHỨNG")]


def test_strip_trailing_punct():
    t = "omeprazole 20 mg po daily."
    assert _clean(t, [(0, len(t), "THUỐC")]) == [("omeprazole 20 mg po daily", "THUỐC")]


def test_drop_vietnamese_route_from_drug():
    t = "prednisolone 40 mg uống"
    assert _clean(t, [(0, len(t), "THUỐC")]) == [("prednisolone 40 mg", "THUỐC")]


def test_drop_junk_header_word():
    t = "thuốc"
    assert _clean(t, [(0, 5, "THUỐC")]) == []


def test_drop_too_short():
    t = "a b c"
    assert _clean(t, [(0, 1, "TRIỆU_CHỨNG")]) == []


def test_dedup_repeats():
    t = "ho khan, sau đó ho khan lại, và ho khan nữa"
    spans = [(0, 7, "TRIỆU_CHỨNG"), (16, 23, "TRIỆU_CHỨNG"), (32, 39, "TRIỆU_CHỨNG")]
    out = clean_spans(t, spans, dedup_repeats=True, max_repeats=1)
    assert len(out) == 1  # 3 copies of "ho khan" collapsed to 1
    out2 = clean_spans(t, spans, dedup_repeats=False)
    assert len(out2) == 3
