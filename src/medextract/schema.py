"""Core data contracts for Task 2 concepts and output validation.

Everything downstream depends on these definitions. Keep them stable.

A *concept* is one extracted medical mention, serialised as a JSON dict with
fields ``text``, ``position`` (``[start, end]`` char offsets, end-exclusive),
``type``, ``assertions`` and ``candidates``.  Field rules (from the host spec):

* ``type`` is one of :data:`CONCEPT_TYPES`.
* ``assertions`` only carry meaning for :data:`ASSERTABLE_TYPES`; empty otherwise.
* ``candidates`` (ontology codes) only for :data:`CANDIDATE_TYPES`; empty otherwise.
* offsets must satisfy ``text == input[start:end]`` exactly (WER depends on it).
"""
from __future__ import annotations

from typing import List, Tuple, TypedDict

# ---- label constants --------------------------------------------------------
CONCEPT_TYPES: List[str] = [
    "TRIỆU_CHỨNG",        # symptom
    "TÊN_XÉT_NGHIỆM",     # test name
    "KẾT_QUẢ_XÉT_NGHIỆM", # test result
    "CHẨN_ĐOÁN",          # diagnosis  -> ICD-10 candidates
    "THUỐC",              # drug       -> RxNorm candidates
]
CONCEPT_TYPE_SET = set(CONCEPT_TYPES)

ASSERTION_LABELS: List[str] = ["isNegated", "isFamily", "isHistorical"]
ASSERTION_LABEL_SET = set(ASSERTION_LABELS)

# Assertions are only meaningful for these types.
ASSERTABLE_TYPES = {"TRIỆU_CHỨNG", "CHẨN_ĐOÁN", "THUỐC"}

# Candidate ontology codes only for these types.
CANDIDATE_TYPES = {"CHẨN_ĐOÁN": "ICD10", "THUỐC": "RXNORM"}

# A raw NER span before assertions/candidates are attached.
Span = Tuple[int, int, str]  # (start, end, type)


class Concept(TypedDict, total=False):
    text: str
    position: List[int]  # [start, end]
    type: str
    assertions: List[str]
    candidates: List[str]


class SchemaError(ValueError):
    """Raised when a concept violates the output contract."""


def validate_concept(c: dict, text: str) -> None:
    """Validate a single concept against ``text``; raise :class:`SchemaError`."""
    if not isinstance(c, dict):
        raise SchemaError(f"concept is not a dict: {c!r}")

    typ = c.get("type")
    if typ not in CONCEPT_TYPE_SET:
        raise SchemaError(f"unknown type {typ!r}; must be one of {CONCEPT_TYPES}")

    pos = c.get("position")
    if (
        not isinstance(pos, (list, tuple))
        or len(pos) != 2
        or not all(isinstance(x, int) for x in pos)
    ):
        raise SchemaError(f"position must be [start, end] ints, got {pos!r}")
    start, end = pos
    if not (0 <= start <= end <= len(text)):
        raise SchemaError(
            f"offsets out of range: [{start}, {end}] for text of len {len(text)}"
        )

    span_text = c.get("text")
    if span_text != text[start:end]:
        raise SchemaError(
            "text != input[start:end]: "
            f"{span_text!r} != {text[start:end]!r} at [{start}, {end}]"
        )

    assertions = c.get("assertions", []) or []
    if not isinstance(assertions, list):
        raise SchemaError(f"assertions must be a list, got {assertions!r}")
    if len(assertions) > 3:
        raise SchemaError(f"at most 3 assertions, got {len(assertions)}")
    for a in assertions:
        if a not in ASSERTION_LABEL_SET:
            raise SchemaError(f"unknown assertion label {a!r}")
    if len(set(assertions)) != len(assertions):
        raise SchemaError(f"duplicate assertions: {assertions!r}")
    if assertions and typ not in ASSERTABLE_TYPES:
        raise SchemaError(f"assertions not allowed on type {typ!r}")

    candidates = c.get("candidates", []) or []
    if not isinstance(candidates, list):
        raise SchemaError(f"candidates must be a list, got {candidates!r}")
    if candidates and typ not in CANDIDATE_TYPES:
        raise SchemaError(f"candidates not allowed on type {typ!r}")
    if not all(isinstance(x, str) for x in candidates):
        raise SchemaError(f"candidate codes must be strings, got {candidates!r}")


def clean_concept(c: dict, text: str) -> Concept:
    """Return a canonical concept dict with only the fields the type allows.

    Drops ``assertions``/``candidates`` for types that must not carry them, so
    the writer always emits schema-consistent objects.
    """
    typ = c["type"]
    start, end = c["position"]
    out: Concept = {
        "text": text[start:end],
        "position": [int(start), int(end)],
        "type": typ,
    }
    if typ in ASSERTABLE_TYPES:
        out["assertions"] = list(dict.fromkeys(c.get("assertions", []) or []))
    else:
        out["assertions"] = []
    if typ in CANDIDATE_TYPES:
        out["candidates"] = list(dict.fromkeys(c.get("candidates", []) or []))
    else:
        out["candidates"] = []
    return out


def validate_output(concepts: list, text: str) -> List[Concept]:
    """Validate + clean a full list of concepts for one record.

    Returns a cleaned, stably-ordered list (by ``position[0]`` then ``[1]``).
    Raises :class:`SchemaError` on the first invalid concept.
    """
    if not isinstance(concepts, list):
        raise SchemaError(f"output must be a list, got {type(concepts)}")
    cleaned: List[Concept] = []
    for c in concepts:
        validate_concept(c, text)
        cleaned.append(clean_concept(c, text))
    cleaned.sort(key=lambda c: (c["position"][0], c["position"][1]))
    return cleaned
