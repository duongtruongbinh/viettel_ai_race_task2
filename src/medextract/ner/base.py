"""Abstract NER interface. Implementations return raw spans, no side effects."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from ..schema import Span


class NERModel(ABC):
    @abstractmethod
    def predict(self, text: str) -> List[Span]:
        """Return ``(start, end, type)`` spans; ``text[start:end]`` is the mention."""
        raise NotImplementedError
