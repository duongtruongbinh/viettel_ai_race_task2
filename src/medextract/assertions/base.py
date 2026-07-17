"""Abstract assertion interface."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from ..schema import Span


class AssertionModel(ABC):
    @abstractmethod
    def predict(self, text: str, spans: List[Span]) -> List[List[str]]:
        """Return a list of assertion-label lists, one per input span.

        Labels are a subset of {isNegated, isFamily, isHistorical}. The caller
        applies them only to assertable types.
        """
        raise NotImplementedError
