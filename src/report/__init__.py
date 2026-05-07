"""Output-driven reporting layer.

Users declare desired outputs ("accuracy", "qsnr", "histogram", ...).
The system derives needed observers and evaluation.
"""

from src.report._session_report import SessionReport
from src.report._spec import PRESETS, _OUTPUT_SPEC, resolve_outputs
from src.report._study_report import StudyReport

__all__ = [
    "SessionReport",
    "StudyReport",
    "resolve_outputs",
    "PRESETS",
    "_OUTPUT_SPEC",
]
