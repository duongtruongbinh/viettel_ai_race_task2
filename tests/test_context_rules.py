"""Tests for the Vietnamese ConText assertion rules."""
from medextract.assertions.context_rules import ContextRules


def _labels(text, span):
    return ContextRules().predict(text, [span])[0]


def test_negation_simple():
    text = "Bệnh nhân không sốt, không ho."
    # span over "sốt"
    s = text.index("sốt")
    assert "isNegated" in _labels(text, (s, s + 3, "TRIỆU_CHỨNG"))


def test_negation_inside_span():
    # gold spans often include the negation word itself: "Không sốt"
    text = "Khám: Không sốt, không ho."
    s = text.index("Không sốt")
    assert "isNegated" in _labels(text, (s, s + len("Không sốt"), "TRIỆU_CHỨNG"))


def test_no_negation_when_absent():
    text = "Bệnh nhân sốt cao 39 độ."
    s = text.index("sốt cao")
    assert _labels(text, (s, s + len("sốt cao"), "TRIỆU_CHỨNG")) == []


def test_family_history():
    text = "Tiền sử gia đình: mẹ bị đái tháo đường."
    s = text.index("đái tháo đường")
    labels = _labels(text, (s, s + len("đái tháo đường"), "CHẨN_ĐOÁN"))
    assert "isFamily" in labels and "isHistorical" in labels


def test_history_section_header():
    text = "Danh sách thuốc trước nhập viện\n- amlodipine 10 mg po daily"
    s = text.index("amlodipine 10 mg po daily")
    labels = _labels(text, (s, s + len("amlodipine 10 mg po daily"), "THUỐC"))
    assert "isHistorical" in labels


def test_terminator_blocks_negation():
    text = "Không sốt nhưng ho nhiều."
    s = text.index("ho")
    # 'nhưng' terminates the negation scope before 'ho'
    assert "isNegated" not in _labels(text, (s, s + 2, "TRIỆU_CHỨNG"))


def test_non_assertable_type_gets_empty():
    text = "Xét nghiệm công thức máu."
    s = text.index("công thức máu")
    assert _labels(text, (s, s + len("công thức máu"), "TÊN_XÉT_NGHIỆM")) == []


def test_section_scoped_history_structured():
    text = ("1. Tiền sử bệnh\nCác bệnh lý nội khoa mạn tính\n"
            "- tăng huyết áp\n\n2. Bệnh sử hiện tại\n- đau ngực")
    s = text.index("tăng huyết áp")
    labs = ContextRules().predict(text, [(s, s + len("tăng huyết áp"), "CHẨN_ĐOÁN")])[0]
    assert "isHistorical" in labs
    # current symptom in the present-illness section is NOT historical
    s2 = text.index("đau ngực")
    labs2 = ContextRules().predict(text, [(s2, s2 + len("đau ngực"), "TRIỆU_CHỨNG")])[0]
    assert "isHistorical" not in labs2
