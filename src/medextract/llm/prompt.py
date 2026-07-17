"""Jinja prompt loader — all LLM prompts live as .jinja in /prompts."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

PROMPTS_DIR = Path(__file__).resolve().parents[3] / "prompts"


@lru_cache(maxsize=1)
def _env() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(PROMPTS_DIR)),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )


def render(template_name: str, **kwargs) -> str:
    if not template_name.endswith(".jinja"):
        template_name += ".jinja"
    return _env().get_template(template_name).render(**kwargs)
