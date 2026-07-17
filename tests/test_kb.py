"""KB build + retriever unit tests (no model download required)."""
from medextract.kb.build_icd import dot_code
from medextract.normalization.retriever import RetrieverNormalizer


def test_dot_code():
    assert dot_code("A001") == "A00.1"
    assert dot_code("I10") == "I10"
    assert dot_code("E1165") == "E11.65"
    assert dot_code("k21.0") == "K21.0"


class _FakeIndex:
    """Returns canned (id, name, score, tty) hits, score-sorted, for any mention."""
    def __init__(self, hits):
        self.hits = hits

    def query(self, mention, top_k=10):
        return self.hits[:top_k]


def test_retriever_cutoff_and_cap():
    idx = _FakeIndex([("308135", "Amlodipine 10 MG Oral Tablet", 0.99, "SCD"),
                      ("111", "junk", 0.10, "IN")])
    r = RetrieverNormalizer({"RXNORM": idx}, top_k=10,
                            cutoffs={"RXNORM": 0.5}, max_candidates={"RXNORM": 2})
    out = r.predict("amlodipine 10 mg po daily", (0, 25, "THUỐC"))
    assert out == ["308135"]  # 0.10 below cutoff dropped


def test_retriever_filter_tty_scd_only():
    # ingredient scores highest but SCD-only filter keeps just the SCD tablet
    idx = _FakeIndex([("17767", "Amlodipine", 0.97, "IN"),
                      ("308135", "Amlodipine 10 MG Oral Tablet", 0.95, "SCD")])
    r = RetrieverNormalizer({"RXNORM": idx}, cutoffs={"RXNORM": 0.5},
                            max_candidates={"RXNORM": 1}, filter_tty={"RXNORM": ["SCD"]})
    out = r.predict("amlodipine 10 mg", (0, 16, "THUỐC"))
    assert out == ["308135"]


def test_retriever_prefer_tty_within_margin():
    # prefer_tty promotes an SCD hit within score margin of the top hit
    idx = _FakeIndex([("17767", "Amlodipine", 0.97, "IN"),
                      ("308135", "Amlodipine 10 MG Oral Tablet", 0.95, "SCD")])
    r = RetrieverNormalizer({"RXNORM": idx}, cutoffs={"RXNORM": 0.5},
                            max_candidates={"RXNORM": 1}, prefer_tty={"RXNORM": ["SCD"]})
    out = r.predict("amlodipine 10 mg", (0, 16, "THUỐC"))
    assert out == ["308135"]


def test_retriever_wrong_type_returns_empty():
    r = RetrieverNormalizer({"RXNORM": _FakeIndex([("1", "x", 0.9, "SCD")])})
    assert r.predict("sốt", (0, 3, "TRIỆU_CHỨNG")) == []


def test_drug_keeps_strength_strips_route():
    captured = {}

    class Cap:
        def query(self, mention, top_k=10):
            captured["m"] = mention
            return [("1", "x", 0.99, "SCD")]

    r = RetrieverNormalizer({"RXNORM": Cap()}, strip_drug_noise=True)
    r.predict("amlodipine 10 mg po daily", (0, 25, "THUỐC"))
    # strength kept, route/freq stripped
    assert "10" in captured["m"] and "mg" in captured["m"]
    assert "po" not in captured["m"].split() and "daily" not in captured["m"].split()
