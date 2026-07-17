"""build_pipeline dispatch: baseline -> retriever, improved -> LLM reranker."""
from medextract.pipeline import build_pipeline, Pipeline


def test_baseline_config_builds_retriever_pipeline(monkeypatch):
    import medextract.ner.gliner_ner as g
    import medextract.assertions.context_rules as a
    import medextract.normalization.retriever as r
    monkeypatch.setattr(g, "from_config", lambda c: "NER")
    monkeypatch.setattr(a, "from_config", lambda c: "ASSERT")
    monkeypatch.setattr(r, "from_config", lambda c: "RETR")
    p = build_pipeline({"pipeline": {}})
    assert isinstance(p, Pipeline)
    assert p.ner == "NER" and p.assertion == "ASSERT" and p.normalizer == "RETR"


def test_improved_config_builds_llm_reranker(monkeypatch):
    import medextract.ner.gliner_ner as g
    import medextract.assertions.context_rules as a
    import medextract.normalization.llm_reranker as lr
    monkeypatch.setattr(g, "from_config", lambda c: "NER")
    monkeypatch.setattr(a, "from_config", lambda c: "ASSERT")
    monkeypatch.setattr(lr, "from_config", lambda c, engine=None: "LLMRR")
    p = build_pipeline({"normalization": {"llm_rerank": {"retrieve_k": 20}}}, engine=object())
    assert p.normalizer == "LLMRR"
