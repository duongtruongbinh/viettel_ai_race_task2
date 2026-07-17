"""Retriever core: tty filter, similarity cutoff, and candidate cap."""
from medextract.normalization.retriever import RetrieverNormalizer


class _FakeIndex:
    """Returns a fixed hit list of (code, name, score, tty) tuples."""

    def __init__(self, hits):
        self._hits = hits

    def query(self, mention, top_k=10):
        return self._hits[:top_k]


def _norm(hits, **kw):
    return RetrieverNormalizer(indexes={"RXNORM": _FakeIndex(hits)},
                               cutoffs={"RXNORM": 0.5}, **kw)


def test_filter_tty_keeps_only_requested_type():
    hits = [("IN1", "amlodipine", 0.9, "IN"),
            ("SCD1", "Amlodipine 10 MG Oral Tablet", 0.88, "SCD")]
    n = _norm(hits, filter_tty={"RXNORM": ["SCD"]}, max_candidates={"RXNORM": 2})
    codes = n.predict("amlodipine 10 mg", (0, 16, "THUỐC"))
    assert codes == ["SCD1"]


def test_cutoff_drops_low_scoring_hits():
    hits = [("SCD1", "x", 0.4, "SCD")]           # below the 0.5 cutoff
    n = _norm(hits, filter_tty={"RXNORM": ["SCD"]})
    assert n.predict("aspirin 81 mg", (0, 13, "THUỐC")) == []


def test_cap_limits_candidate_count():
    hits = [("A", "a", 0.9, "SCD"), ("B", "b", 0.85, "SCD"), ("C", "c", 0.8, "SCD")]
    n = _norm(hits, max_candidates={"RXNORM": 1})
    assert n.predict("metoprolol 50 mg", (0, 16, "THUỐC")) == ["A"]
