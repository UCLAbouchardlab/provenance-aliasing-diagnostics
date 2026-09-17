"""Public configuration, validation, and metadata diagnostic API.

Declare the scientific roles explicitly with :class:`AnalysisConfig`, then use
``validate(metadata, config=config)`` to inspect the table or
``diagnose(metadata, config=config)`` to run the selected diagnostics.
"""

from .config import AnalysisConfig, ConfigurationError
from .validation import (
    MetadataValidationError,
    ValidationIssue,
    ValidationReport,
    validate,
)
from .diagnostics import diagnose
from .results import DiagnosticComputationError, DiagnosticResult
from .run_record import (
    RecordedRun,
    RunRecord,
    RunRecordError,
    VerificationReport,
    run_analysis,
    verify_run,
)

__all__ = [
    "AnalysisConfig",
    "ConfigurationError",
    "MetadataValidationError",
    "ValidationIssue",
    "ValidationReport",
    "validate",
    "diagnose",
    "DiagnosticResult",
    "DiagnosticComputationError",
    "RecordedRun",
    "RunRecord",
    "RunRecordError",
    "VerificationReport",
    "run_analysis",
    "verify_run",
]
