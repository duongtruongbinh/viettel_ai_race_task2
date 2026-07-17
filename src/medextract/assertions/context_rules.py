"""Vietnamese ConText assertion detector (negation / family / history).

Self-contained rule engine (no medspaCy/spaCy). For each span we inspect a
context window and fire assertion labels:

* **isNegated**   — a negation trigger precedes the mention in its clause, not cut
  off by a terminator (``nhưng``, ``tuy nhiên``, ``;`` …). Scope: same line.
* **isFamily**    — a family trigger ("mẹ", "chị gái", "gia đình" …) on the
  mention's **line**. Family history is expressed inline in these notes.
* **isHistorical** — a history trigger ("tiền sử", "trước đó", "đã dùng" …) on the
  line, or a section-header history cue governing the mention's block (e.g. a
  pre-admission "thuốc … trước nhập viện" list makes every item historical).

Matching is **word-boundary** based (regex), so short triggers like "anh" no
longer fire inside "nhanh". Trigger lists are overridable via config.
"""
from __future__ import annotations

import re
import unicodedata
from typing import List, Optional, Sequence, Tuple

from ..schema import ASSERTABLE_TYPES, Span
from .base import AssertionModel

# ---- default Vietnamese trigger lexicons ------------------------------------
NEGATION = [
    "không có", "không thấy", "không ghi nhận", "không còn", "chưa ghi nhận",
    "chưa có", "loại trừ", "phủ định", "phủ nhận", "âm tính", "không bị",
    "không do", "không", "chưa", "ko",
]
# curated family triggers (word-boundary matched; same-line scope)
FAMILY = [
    "gia đình", "người nhà", "họ hàng", "di truyền",
    "bố", "mẹ", "cha", "ba", "má",
    "ông", "bà", "ông nội", "bà nội", "ông ngoại", "bà ngoại",
    "anh", "chị", "em", "anh trai", "chị gái", "em trai", "em gái",
    "con", "cậu", "dì", "chú", "bác", "cô",
]
# same-line history cues, applied DIRECTIONALLY (must precede the concept in its
# clause), like negation — so "Tiền sử X, vào vì Y" marks X but not Y.
HISTORY_LINE = [
    "tiền sử", "tiền căn", "trước đây", "trước đó", "đã từng", "đã dùng",
    "đã được chẩn đoán", "cách đây", "cách nhập viện", "trong quá khứ",
    "trước khi nhập viện", "trước nhập viện", "trước lúc nhập viện",
]
# section-header cues that make a following block historical. Bare "tiền sử" is
# handled specially (excluded when the section is "tiền sử gia đình" = family).
# NOTE: excludes bare "bệnh sử" — ambiguous with "bệnh sử hiện tại" (= current).
HISTORY_SECTION = [
    "tiền sử bệnh", "tiền sử nội khoa", "tiền căn",
    "bệnh lý nội khoa mạn", "bệnh lý mạn tính", "các bệnh lý nội khoa",
    "bệnh mạn tính", "tập kinh lâm sàng trước",
    "trước nhập viện", "trước khi nhập viện", "trước lúc nhập viện",
    "thuốc trước", "thuốc đang dùng", "tại nhà", "đang dùng tại nhà",
    "các sự kiện trước khi nhập viện",
]
# section-header cues that make a following block a family-history section
FAMILY_SECTION = ["tiền sử gia đình", "gia đình", "bệnh sử gia đình"]
# a line that opens a new top-level section (resets the governing context)
SECTION_HEADER = None  # compiled below
# markers after which a concept is an indication (don't inherit history)
INDICATION = ["điều trị", "chỉ định", "dự phòng", "để", "cho"]
# clause boundaries that reset the history scope inside a line
HISTORY_STOPS = [".", "hiện", "vào vì", "vào viện", "hiện dùng", "hiện tại"]
TERMINATORS = ["nhưng", "tuy nhiên", "mặc dù", ";", "song", "ngoại trừ"]


def _norm(s: str) -> str:
    return unicodedata.normalize("NFC", s).lower()


def _compile(triggers: Sequence[str]) -> List[re.Pattern]:
    """Word-boundary regexes for each trigger (handles Vietnamese unicode)."""
    return [re.compile(r"(?<!\w)" + re.escape(_norm(t)) + r"(?!\w)") for t in triggers]


# a top-level numbered section header line ("1. Tiền sử bệnh", "2. Bệnh sử …")
_TOP_SECTION = re.compile(r"^\s*\d+[.)]\s+\S")


class ContextRules(AssertionModel):
    def __init__(
        self,
        negation: Optional[Sequence[str]] = None,
        family: Optional[Sequence[str]] = None,
        history_line: Optional[Sequence[str]] = None,
        history_section: Optional[Sequence[str]] = None,
        indication: Optional[Sequence[str]] = None,
        terminators: Optional[Sequence[str]] = None,
        neg_window_chars: int = 80,
        block_lookback_lines: int = 15,
    ):
        self.negation = _compile(negation or NEGATION)
        self.family = _compile(family or FAMILY)
        self.history_line = _compile(history_line or HISTORY_LINE)
        self.history_section = _compile(history_section or HISTORY_SECTION)
        self.family_section = _compile(FAMILY_SECTION)
        self.indication = _compile(indication or INDICATION)
        self.history_stops = _compile(HISTORY_STOPS)
        self.terminators = _compile(terminators or TERMINATORS)
        self.neg_window_chars = neg_window_chars
        self.block_lookback_lines = block_lookback_lines

    # -- context helpers ------------------------------------------------------
    @staticmethod
    def _line_bounds(text: str, pos: int) -> Tuple[int, int]:
        start = text.rfind("\n", 0, pos) + 1
        end = text.find("\n", pos)
        if end == -1:
            end = len(text)
        return start, end

    def _block_before(self, text: str, line_start: int) -> str:
        """Preceding lines up to a blank line (section block), for history headers."""
        lines_seen, cur, chunks = 0, line_start, []
        while cur > 0 and lines_seen < self.block_lookback_lines:
            prev_end = cur - 1
            prev_start = text.rfind("\n", 0, prev_end) + 1
            line = text[prev_start:prev_end]
            if line.strip() == "":
                break
            chunks.append(line)
            cur = prev_start
            lines_seen += 1
        return "\n".join(reversed(chunks))

    def _governing_context(self, text: str, line_start: int, max_lines: int = 20) -> str:
        """Preceding lines back to (and including) the nearest top-level numbered
        section header — the section that governs this concept in a *structured*
        note. Returns "" if no real section header is found, so section-scoped
        assertions only fire on structured notes (not short unstructured ones)."""
        cur, chunks, seen, found = line_start, [], 0, False
        while cur > 0 and seen < max_lines:
            prev_end = cur - 1
            prev_start = text.rfind("\n", 0, prev_end) + 1
            line = text[prev_start:prev_end]
            if line.strip() == "":
                break
            chunks.append(line)
            cur = prev_start
            seen += 1
            if _TOP_SECTION.match(line):  # reached the section header
                found = True
                break
        return "\n".join(reversed(chunks)) if found else ""

    @staticmethod
    def _any(patterns: List[re.Pattern], haystack: str) -> bool:
        h = _norm(haystack)
        return any(p.search(h) for p in patterns)

    def _negated(self, text: str, start: int, end: int, line_start: int) -> bool:
        # (a) negation trigger at the START of the span itself ("Không sốt")
        span = _norm(text[start:end])
        if any(p.match(span) for p in self.negation):
            return True
        # (b) negation trigger in the pre-context, not cut off by a terminator
        pre = _norm(text[max(line_start, start - self.neg_window_chars):start])
        last_neg = -1
        for p in self.negation:
            for m in p.finditer(pre):
                last_neg = max(last_neg, m.end())
        if last_neg < 0:
            return False
        tail = pre[last_neg:]
        return not any(t.search(tail) for t in self.terminators)

    def _historical(self, text: str, start: int, line_start: int, section: str) -> bool:
        pre_line = _norm(text[line_start:start])
        # scope = text after the last clause boundary before the concept
        cut = 0
        for p in self.history_stops:
            for m in p.finditer(pre_line):
                cut = max(cut, m.end())
        scope = pre_line[cut:]
        # 1) directional same-line history cue in the current clause
        if any(p.search(scope) for p in self.history_line):
            return True
        # 2) governing section is a patient-history / pre-admission-med section,
        #    unless it is a *family*-history section (→ family, not blanket
        #    historical) or the concept is a drug indication on its line.
        if self._any(self.history_section, section) and not self._any(self.family_section, section):
            if not any(p.search(pre_line) for p in self.indication):
                return True
        return False

    # -- API ------------------------------------------------------------------
    def predict(self, text: str, spans: List[Span]) -> List[List[str]]:
        out: List[List[str]] = []
        for (start, end, typ) in spans:
            if typ not in ASSERTABLE_TYPES:
                out.append([])
                continue
            line_start, line_end = self._line_bounds(text, start)
            line = text[line_start:line_end]
            # section block = preceding lines up to a blank line (sections in these
            # notes are blank-line separated); governs section-scoped assertions.
            section = self._block_before(text, line_start)

            labels: List[str] = []
            if self._negated(text, start, end, line_start):
                labels.append("isNegated")
            # family: same-line cue OR a family-history section governs the concept
            if self._any(self.family, line) or self._any(self.family_section, section):
                labels.append("isFamily")
            if self._historical(text, start, line_start, section):
                labels.append("isHistorical")
            out.append(labels[:3])
        return out


def from_config(cfg: dict) -> ContextRules:
    a = (cfg or {}).get("assertions", {}) or {}
    trig = a.get("triggers", {}) or {}
    return ContextRules(
        negation=trig.get("negation"),
        family=trig.get("family"),
        history_line=trig.get("history_line"),
        history_section=trig.get("history_section"),
        indication=trig.get("indication"),
        terminators=trig.get("terminators"),
        neg_window_chars=a.get("neg_window_chars", 80),
        block_lookback_lines=a.get("block_lookback_lines", 8),
    )
