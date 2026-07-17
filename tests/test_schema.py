"""Tests for the core output contract (schema.validate_*)."""
import pytest

from medextract.schema import (
    SchemaError,
    clean_concept,
    validate_concept,
    validate_output,
)

TEXT = "Bệnh nhân sốt cao, không ho. amlodipine 10 mg."


def _c(text, typ, start, end, **kw):
    return {"text": text, "type": typ, "position": [start, end], **kw}


def test_valid_symptom():
    validate_concept(_c("sốt cao", "TRIỆU_CHỨNG", 10, 17, assertions=["isNegated"]), TEXT)


def test_offset_must_match_text():
    with pytest.raises(SchemaError):
        validate_concept(_c("sốt", "TRIỆU_CHỨNG", 0, 3), TEXT)  # TEXT[0:3] != 'sốt'


def test_offsets_out_of_range():
    with pytest.raises(SchemaError):
        validate_concept(_c("x", "TRIỆU_CHỨNG", 0, 999), TEXT)


def test_unknown_type():
    with pytest.raises(SchemaError):
        validate_concept(_c("sốt cao", "SYMPTOM", 10, 17), TEXT)


def test_assertions_on_non_assertable_type():
    with pytest.raises(SchemaError):
        validate_concept(
            _c("amlodipine", "TÊN_XÉT_NGHIỆM", 28, 38, assertions=["isNegated"]), TEXT
        )


def test_candidates_on_non_candidate_type():
    with pytest.raises(SchemaError):
        validate_concept(
            _c("sốt cao", "TRIỆU_CHỨNG", 10, 17, candidates=["K21.0"]), TEXT
        )


def test_too_many_assertions():
    with pytest.raises(SchemaError):
        validate_concept(
            _c("sốt cao", "TRIỆU_CHỨNG", 10, 17,
               assertions=["isNegated", "isFamily", "isHistorical", "isNegated"]),
            TEXT,
        )


def test_unknown_assertion_label():
    with pytest.raises(SchemaError):
        validate_concept(_c("sốt cao", "TRIỆU_CHỨNG", 10, 17, assertions=["isFoo"]), TEXT)


def test_clean_strips_disallowed_fields():
    c = clean_concept(
        _c("amlodipine", "TÊN_XÉT_NGHIỆM", 28, 38, assertions=["isNegated"],
           candidates=["X"]),
        TEXT,
    )
    assert c["assertions"] == [] and c["candidates"] == []


def test_validate_output_sorts_and_cleans():
    out = validate_output(
        [
            _c("10 mg", "TRIỆU_CHỨNG", 40, 45),
            _c("sốt cao", "TRIỆU_CHỨNG", 10, 17),
        ],
        TEXT,
    )
    assert [c["position"][0] for c in out] == [10, 40]  # sorted by start


def test_candidate_dedup():
    c = clean_concept(
        _c("amlodipine", "THUỐC", 28, 38, candidates=["308135", "308135"]), TEXT
    )
    assert c["candidates"] == ["308135"]
