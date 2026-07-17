"""Pipeline: compose NER + assertions + normalization into per-record concepts.

The same ``Pipeline`` object is used by ``main.py`` (batch/zip) and ``serve.py``
(API). Stages are pluggable; any may be ``None`` (e.g. dummy solution).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

from . import io_utils
from .assertions.base import AssertionModel
from .ner.base import NERModel
from .normalization.base import Normalizer
from .schema import (
    ASSERTABLE_TYPES,
    CANDIDATE_TYPES,
    Concept,
    Span,
    validate_output,
)

log = logging.getLogger("medextract.pipeline")


def build_pipeline(config: dict, engine=None) -> "Pipeline":
    """Build the full pipeline from a config dict.

    NER is always GLiNER; assertions are always the ConText rule engine. The
    normalizer is the plain SapBERT retriever unless the config declares a
    ``normalization.llm_rerank`` block, in which case a local LLM reranks the
    retrieved candidates (pass an existing ``engine`` to reuse one).
    """
    from .ner.gliner_ner import from_config as build_ner
    from .assertions.context_rules import from_config as build_assertions

    norm_cfg = (config or {}).get("normalization", {}) or {}
    if norm_cfg.get("llm_rerank"):
        from .normalization.llm_reranker import from_config as build_norm
        if engine is None:
            from .llm.engine import LLMEngine
            lc = (config or {}).get("llm", {}) or {}
            engine = LLMEngine(
                model_name=lc.get("model", "/mnt/pretrained_fm/Qwen_Qwen3-8B"),
                device=lc.get("device", "wait"),
                dtype=lc.get("dtype", "bfloat16"),
                min_free_gb=lc.get("min_free_gb", 18.0),
                enable_thinking=lc.get("enable_thinking", False),
                load_in_4bit=lc.get("load_in_4bit", False),
            )
        normalizer = build_norm(config, engine=engine)
    else:
        from .normalization.retriever import from_config as build_norm
        normalizer = build_norm(config)

    return Pipeline(
        ner=build_ner(config),
        assertion=build_assertions(config),
        normalizer=normalizer,
        name=(config or {}).get("solution", "medextract"),
        **pipeline_opts(config),
    )


def pipeline_opts(config: dict) -> dict:
    """Extract pipeline-level options (span cleanup / dedup) from config."""
    p = (config or {}).get("pipeline", {}) or {}
    return dict(
        clean_spans=p.get("clean_spans", True),
        dedup_repeats=p.get("dedup_repeats", False),
        max_repeats=p.get("max_repeats", 1),
    )


class Pipeline:
    def __init__(
        self,
        ner: Optional[NERModel] = None,
        assertion: Optional[AssertionModel] = None,
        normalizer: Optional[Normalizer] = None,
        name: str = "pipeline",
        clean_spans: bool = True,
        dedup_repeats: bool = False,
        max_repeats: int = 1,
    ):
        self.ner = ner
        self.assertion = assertion
        self.normalizer = normalizer
        self.name = name
        self.clean_spans = clean_spans
        self.dedup_repeats = dedup_repeats
        self.max_repeats = max_repeats

    # -- core -----------------------------------------------------------------
    def run_text(self, text: str) -> List[Concept]:
        """Extract concepts from one raw record string."""
        if self.ner is None:
            return []
        spans: List[Span] = self.ner.predict(text)
        if self.clean_spans:
            from .ner.postprocess import clean_spans
            spans = clean_spans(text, spans, dedup_repeats=self.dedup_repeats,
                                max_repeats=self.max_repeats)

        # assertions (batch over all spans; applied only to assertable types)
        if self.assertion is not None and spans:
            labels_per_span = self.assertion.predict(text, spans)
        else:
            labels_per_span = [[] for _ in spans]

        concepts: List[dict] = []
        for span, labels in zip(spans, labels_per_span):
            start, end, typ = span
            c: dict = {"text": text[start:end], "position": [start, end], "type": typ}
            c["assertions"] = list(labels) if typ in ASSERTABLE_TYPES else []
            if typ in CANDIDATE_TYPES and self.normalizer is not None:
                c["candidates"] = self.normalizer.predict(text, span)
            else:
                c["candidates"] = []
            concepts.append(c)

        # validate + clean + stable-order
        return validate_output(concepts, text)

    def run_file(self, path) -> List[Concept]:
        return self.run_text(io_utils.read_text(path))

    def run_dir(self, in_dir, out_dir, zip_it: bool = True) -> None:
        """Run over all ``*.txt`` in ``in_dir``, write validated JSON, zip."""
        in_dir, out_dir = Path(in_dir), Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        inputs = io_utils.list_inputs(in_dir)
        log.info("[%s] running over %d files -> %s", self.name, len(inputs), out_dir)
        for i, path in enumerate(inputs, 1):
            text = io_utils.read_text(path)
            concepts = self.run_text(text)
            io_utils.write_record(out_dir, path.stem, concepts, text)
            if i % 10 == 0 or i == len(inputs):
                log.info("[%s]  %d/%d", self.name, i, len(inputs))
        if zip_it:
            zp = io_utils.zip_submission(out_dir)
            log.info("[%s] wrote %s", self.name, zp)
