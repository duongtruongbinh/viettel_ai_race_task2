"""medextract — shared library for AI Race 2026 Task 2.

Medical concept extraction from Vietnamese clinical text: NER, assertions,
ICD-10 / RxNorm normalization, producing the host submission JSON.
"""
from __future__ import annotations

__version__ = "0.1.0"


def set_seed(seed: int = 42) -> None:
    """Fix seeds across random / numpy / torch for reproducibility."""
    import os
    import random

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
