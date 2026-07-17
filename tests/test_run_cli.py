"""run.py argument parsing."""
import run


def test_parse_args_defaults():
    a = run.parse_args(["--config", "configs/baseline.yaml", "--input", "in", "--output", "out"])
    assert a.config == "configs/baseline.yaml"
    assert a.zip is False
    assert a.seed == 42


def test_parse_args_zip_flag():
    a = run.parse_args(["--config", "c", "--input", "i", "--output", "o", "--zip"])
    assert a.zip is True
