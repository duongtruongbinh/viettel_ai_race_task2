"""Abstract normalizer interface (mention -> ranked ontology codes)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from ..schema import Span


class Normalizer(ABC):
    @abstractmethod
    def predict(self, text: str, span: Span) -> List[str]:
        """Return ranked candidate codes for one CHẨN_ĐOÁN / THUỐC span.

        Empty list means "no confident candidate". The KB used (ICD10 vs
        RXNORM) is chosen from the span's type by the implementation.
        """
        raise NotImplementedError
