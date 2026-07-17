"""Zero-shot NER with GLiNER (multilingual) for the 5 concept types.

GLiNER returns character offsets into the string it was given, so we chunk long
notes on line boundaries (verbatim slices, no text mutation) and shift offsets by
the chunk start — guaranteeing ``text[start:end]`` equals the mention (WER-safe).

Label strings are prompt-like and configurable; each maps to one canonical type.
Overlapping spans are resolved by score, then by length, keeping clean boundaries.
"""
from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Sequence, Tuple

from ..schema import CONCEPT_TYPE_SET, Span
from .base import NERModel

log = logging.getLogger("medextract.ner.gliner")

# a dosage/route/frequency token that should be pulled into a drug span
_DOSE_TOKEN = re.compile(
    r"^(\d+([.,\-/]\d+)*"                                  # 10, 0.5, 325-650
    r"|mg|mcg|ug|g|ml|iu|x|%"                              # units
    r"|po|iv|im|sc|sl|pr|tab"                              # routes
    r"|(q\d+h|qd|qid|qod|bid|tid|qhs|qam|qpm|prn|daily)(:prn)?"  # freq (+opt :prn)
    r")$",
    re.IGNORECASE,
)
# Vietnamese indication markers that END a drug span (don't absorb them)
_INDICATION = ("điều trị", "cho", "để", "khi", "nếu")

# label string -> canonical type. Descriptive labels generalize best in GLiNER.
DEFAULT_LABEL_MAP: Dict[str, str] = {
    "symptom": "TRIỆU_CHỨNG",
    "disease or diagnosis": "CHẨN_ĐOÁN",
    "medication or drug": "THUỐC",
    "medical test or lab name": "TÊN_XÉT_NGHIỆM",
    "test result or measurement value": "KẾT_QUẢ_XÉT_NGHIỆM",
}


class GLiNERNER(NERModel):
    def __init__(
        self,
        model_name: str = "urchade/gliner_multi-v2.1",
        label_map: Optional[Dict[str, str]] = None,
        threshold: float = 0.5,
        max_chunk_chars: int = 1200,
        max_chunk_tokens: Optional[int] = None,
        device: str = "auto",
    ):
        from gliner import GLiNER

        from ..utils.gpu import resolve_device

        self.label_map = label_map or DEFAULT_LABEL_MAP
        self.labels = list(self.label_map.keys())
        self.threshold = threshold
        self.max_chunk_chars = max_chunk_chars
        # When set, chunk by TOKEN budget (GLiNER truncates at 384 tokens; char-based
        # 800 exceeds that on ~8% of dense chunks, silently dropping their tails).
        self.max_chunk_tokens = max_chunk_tokens
        self._counter = None
        self.device = resolve_device(device)
        log.info("loading GLiNER %s on %s", model_name, self.device)
        self.model = GLiNER.from_pretrained(model_name)
        try:
            self.model = self.model.to(self.device)
        except Exception:  # some GLiNER versions manage device internally
            pass

    # -- chunking -------------------------------------------------------------
    def _tok_count(self, s: str) -> int:
        if self._counter is None:
            from transformers import AutoTokenizer
            self._counter = AutoTokenizer.from_pretrained("microsoft/mdeberta-v3-base")
        return len(self._counter(s, add_special_tokens=False)["input_ids"])

    def _chunks(self, text: str) -> List[Tuple[int, str]]:
        """Yield (global_start, chunk_text) verbatim slices on line boundaries.

        Token-budget mode (``max_chunk_tokens``) packs whole lines up to the budget
        so no chunk ever exceeds GLiNER's 384-token cap (no truncated tails)."""
        if self.max_chunk_tokens:
            return self._token_chunks(text)
        chunks: List[Tuple[int, str]] = []
        pos = 0
        n = len(text)
        while pos < n:
            end = min(pos + self.max_chunk_chars, n)
            if end < n:
                nl = text.rfind("\n", pos, end)
                if nl > pos:
                    end = nl + 1
            chunks.append((pos, text[pos:end]))
            pos = end
        return chunks or [(0, "")]

    def _token_chunks(self, text: str) -> List[Tuple[int, str]]:
        budget = self.max_chunk_tokens
        # split keeping line-ending chars so offsets stay exact
        lines, pos = [], 0
        for line in text.splitlines(keepends=True):
            lines.append((pos, line)); pos += len(line)
        chunks: List[Tuple[int, str]] = []
        buf, buf_start, buf_tok = "", 0, 0
        for lstart, line in lines:
            lt = self._tok_count(line)
            if buf and buf_tok + lt > budget:
                chunks.append((buf_start, buf)); buf, buf_tok = "", 0
            if not buf:
                buf_start = lstart
            buf += line; buf_tok += lt
        if buf:
            chunks.append((buf_start, buf))
        return chunks or [(0, "")]

    @staticmethod
    def _extend_drug_span(text: str, start: int, end: int) -> int:
        """Extend a THUỐC span rightward over trailing dosage/route/freq tokens.

        Gold drug spans include strength/route ("amlodipine 10 mg po daily") which
        GLiNER usually clips to the name. We absorb following ASCII dosage tokens up
        to end-of-line, stopping at a Vietnamese indication marker.
        """
        line_end = text.find("\n", end)
        if line_end == -1:
            line_end = len(text)
        rest = text[end:line_end]
        low = rest.lower()
        for marker in _INDICATION:
            mpos = low.find(marker)
            if mpos != -1:
                rest = rest[:mpos]
                break
        new_end = end
        for m in re.finditer(r"\S+", rest):
            tok = m.group(0).strip(".,;:")
            if _DOSE_TOKEN.match(tok):
                new_end = end + m.end()
            else:
                break
        # trim trailing whitespace already excluded (we advanced to token end)
        return new_end

    @staticmethod
    def _resolve_overlaps(spans: List[Tuple[int, int, str, float]]) -> List[Span]:
        """Drop overlapping spans, preferring higher score then shorter length."""
        spans = sorted(spans, key=lambda s: (-s[3], s[0] - s[1]))
        kept: List[Tuple[int, int, str, float]] = []
        for s in spans:
            if any(not (s[1] <= k[0] or s[0] >= k[1]) for k in kept):
                continue  # overlaps an already-kept span
            kept.append(s)
        kept.sort(key=lambda s: (s[0], s[1]))
        return [(s[0], s[1], s[2]) for s in kept]

    # -- API ------------------------------------------------------------------
    def predict(self, text: str) -> List[Span]:
        raw: List[Tuple[int, int, str, float]] = []
        for gstart, chunk in self._chunks(text):
            if not chunk.strip():
                continue
            ents = self.model.predict_entities(chunk, self.labels, threshold=self.threshold)
            for e in ents:
                typ = self.label_map.get(e["label"])
                if typ not in CONCEPT_TYPE_SET:
                    continue
                s, en = gstart + e["start"], gstart + e["end"]
                # trim whitespace at span edges (keeps text == input[s:en])
                while s < en and text[s].isspace():
                    s += 1
                while en > s and text[en - 1].isspace():
                    en -= 1
                if typ == "THUỐC":
                    en = self._extend_drug_span(text, s, en)
                if s < en:
                    raw.append((s, en, typ, float(e.get("score", 1.0))))
        return self._resolve_overlaps(raw)


def from_config(cfg: dict) -> GLiNERNER:
    n = (cfg or {}).get("ner", {}) or {}
    return GLiNERNER(
        model_name=n.get("model", "urchade/gliner_multi-v2.1"),
        label_map=n.get("label_map", DEFAULT_LABEL_MAP),
        threshold=n.get("threshold", 0.5),
        max_chunk_chars=n.get("max_chunk_chars", 1200),
        max_chunk_tokens=n.get("max_chunk_tokens"),
        device=(cfg or {}).get("kb", {}).get("device", "auto"),
    )
