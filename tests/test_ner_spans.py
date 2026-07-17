"""Tests for GLiNER drug-span extension (no model load — static method)."""
from medextract.ner.gliner_ner import GLiNERNER

ext = GLiNERNER._extend_drug_span


def test_extend_over_dosage():
    text = "- amlodipine 10 mg po daily"
    s = text.index("amlodipine")
    e = s + len("amlodipine")
    new_e = ext(text, s, e)
    assert text[s:new_e] == "amlodipine 10 mg po daily"


def test_stops_at_indication_marker():
    text = "guaifenesin ml po q6h:prn điều trị ho"
    s = 0
    e = len("guaifenesin")
    new_e = ext(text, s, e)
    assert text[s:new_e] == "guaifenesin ml po q6h:prn"


def test_stops_at_end_of_line():
    text = "clonazepam 0.5 mg po qam:prn\nnext line"
    s = 0
    e = len("clonazepam")
    new_e = ext(text, s, e)
    assert text[s:new_e] == "clonazepam 0.5 mg po qam:prn"


def test_no_dosage_leaves_span():
    text = "atenolol (uống hôm nay)"
    s = 0
    e = len("atenolol")
    assert ext(text, s, e) == e  # '(' is not a dose token
