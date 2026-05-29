"""Stage 2 — Arabic linguistic preprocessing.

Public API:
    ArabicTextPipeline   — main entry point, composes the steps below
    PipelineConfig       — config dataclass
    PreprocessedDocument — return type of ``ArabicTextPipeline.process``
    DateHit              — single extracted date occurrence
"""

from arnlp.preprocessing.pipeline import ArabicTextPipeline, PipelineConfig
from arnlp.preprocessing.types import DateHit, PreprocessedDocument

__all__ = [
    "ArabicTextPipeline",
    "PipelineConfig",
    "PreprocessedDocument",
    "DateHit",
]
