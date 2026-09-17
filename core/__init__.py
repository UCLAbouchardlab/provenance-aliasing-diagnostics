"""Development APIs for metadata-based provenance-aliasing diagnostics.

Exports retain their reference names during migration. Imports are lazy so package
information does not load numerical or optional dependencies. Each name has one
explicit owner; an installation failure is never hidden by searching other modules.
"""

from __future__ import annotations

import importlib
from importlib.metadata import version
from typing import Any

__version__ = version("provenance-aliasing-diagnostics")

__all__ = [
    "Corpus",
    "DuplicateObservationWarning",
    "label_entropy",
    "leave_one_source_out",
    "bootstrap_ratio",
    "balanced_accuracy_ceiling",
    "pair_sharing_by_group",
    "structural_leave_one_source_out",
    "spanning_source_table",
    "score_grains",
    "guard_report",
    "cramers_v",
    "arbitrary_split_null",
    "k_diagnostic",
    "dispersion_gap",
    "METRICS",
    "design_curve",
    "simulated_ceiling_envelope",
    "diagnose",
]

_ORIGIN: dict[str, str] = {
    "Corpus": "incidence",
    "DuplicateObservationWarning": "incidence",
    "label_entropy": "entropy",
    "leave_one_source_out": "entropy",
    "bootstrap_ratio": "entropy",
    "EntropyResult": "entropy",
    "BootstrapResult": "entropy",
    "balanced_accuracy_ceiling": "ceiling",
    "CeilingResult": "ceiling",
    "simulated_ceiling_envelope": "design",
    "pair_sharing_by_group": "structure",
    "structural_leave_one_source_out": "structure",
    "spanning_source_table": "structure",
    "score_grains": "grain",
    "guard_report": "guards",
    "cramers_v": "guards",
    "arbitrary_split_null": "calibration",
    "k_diagnostic": "calibration",
    "dispersion_gap": "calibration",
    "METRICS": "metrics",
    "design_curve": "design",
    "diagnose": "report",
    "Diagnosis": "report",
}
_MODULES: tuple[str, ...] = (
    "incidence",
    "entropy",
    "ceiling",
    "structure",
    "grain",
    "guards",
    "calibration",
    "metrics",
    "design",
    "report",
)

_SUBMODULES: frozenset[str] = frozenset(_MODULES) | {"adapters"}


def __getattr__(name: str) -> Any:
    if name in _SUBMODULES and name not in _ORIGIN:
        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    if name not in _ORIGIN:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = importlib.import_module(f".{_ORIGIN[name]}", __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(_ORIGIN) | set(_SUBMODULES) | {"__version__"})
