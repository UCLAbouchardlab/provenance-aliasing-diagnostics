"""Detached metadata-diagnostic results and strict, lossless JSON exports.

This module presents completed calculations. It does not run diagnostics, write
files, or change the reference core's numerical or reporting behavior.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from importlib.metadata import version
import json
import math
from numbers import Real
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd

from .config import AnalysisConfig
from .validation import ValidationReport


_PACKAGE_NAME = "provenance-aliasing-diagnostics"
_METRIC_COLUMNS = ("metric", "family", "value", "status", "reason", "weighting")
_METRIC_STATUSES = frozenset({
    "ok", "undefined", "not_applicable", "not_requested", "resource_limited",
})


class DiagnosticComputationError(RuntimeError):
    """An unexpected stage failure, retaining its metadata assessment."""

    def __init__(
        self, stage: str, message: str, *, validation: ValidationReport | None = None,
    ) -> None:
        if not isinstance(stage, str) or not stage.strip():
            raise ValueError("stage must be a nonblank string")
        if not isinstance(message, str) or not message.strip():
            raise ValueError("message must be a nonblank string")
        if validation is not None and not isinstance(validation, ValidationReport):
            raise TypeError("validation must be a ValidationReport or None")
        self.stage = stage
        self.validation = validation
        super().__init__(f"{stage}: {message}")


def _copy_index(index: pd.Index) -> pd.Index:
    """Copy axis buffers and object labels, preserving tuple-key boundaries."""
    if isinstance(index, pd.MultiIndex):
        return pd.MultiIndex(
            levels=[_copy_index(level) for level in index.levels],
            codes=[code.copy() for code in index.codes],
            names=deepcopy(list(index.names)),
            verify_integrity=False,
        )
    if isinstance(index, pd.CategoricalIndex):
        return pd.CategoricalIndex(pd.Categorical.from_codes(
            index.codes.copy(), categories=_copy_index(index.categories), ordered=index.ordered,
        ), name=deepcopy(index.name))
    if pd.api.types.is_object_dtype(index.dtype):
        return pd.Index(
            [deepcopy(value) for value in index], dtype=object,
            name=deepcopy(index.name), tupleize_cols=False,
        )
    copied = index.copy(deep=True)
    copied.name = deepcopy(index.name)
    return copied


def _copy_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Detach mutable object cells as well as pandas' ordinary array buffers."""
    copied = frame.copy(deep=True)
    for column in range(len(frame.columns)):
        dtype = frame.dtypes.iloc[column]
        if pd.api.types.is_object_dtype(dtype):
            for row in range(len(frame)):
                copied.iat[row, column] = deepcopy(frame.iat[row, column])
        elif isinstance(dtype, pd.CategoricalDtype):
            values = frame.iloc[:, column].array
            copied.isetitem(column, pd.Categorical.from_codes(
                values.codes.copy(), categories=_copy_index(values.categories), ordered=values.ordered,
            ))
    copied.index = _copy_index(frame.index)
    copied.columns = _copy_index(frame.columns)
    copied.attrs = deepcopy(frame.attrs)
    return copied


def _missing(value: Any) -> bool:
    if value is None or value is pd.NA or value is pd.NaT:
        return True
    if isinstance(value, (float, np.floating)):
        return bool(np.isnan(value))
    if isinstance(value, (np.datetime64, np.timedelta64)):
        return bool(np.isnat(value))
    return False


def _json_value(value: Any, *, active: set[int] | None = None) -> Any:
    """Convert supported scalar/container values without stringifying identity."""
    if _missing(value):
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("result exports cannot contain infinite numbers")
        return number
    if isinstance(value, (Mapping, list, tuple, np.ndarray)):
        active = set() if active is None else active
        identity = id(value)
        if identity in active:
            raise ValueError("result exports cannot contain circular containers")
        active.add(identity)
        try:
            if isinstance(value, Mapping):
                result: dict[str, Any] = {}
                for key, item in value.items():
                    if not isinstance(key, str):
                        raise TypeError("result mapping keys must be strings; use a table for structured keys")
                    result[key] = _json_value(item, active=active)
                return result
            if isinstance(value, np.ndarray):
                return _json_value(value.tolist(), active=active)
            return [_json_value(item, active=active) for item in value]
        finally:
            active.remove(identity)
    raise TypeError(f"unsupported result value type: {type(value).__name__}")


def _require_frame(value: Any, name: str) -> pd.DataFrame:
    if not isinstance(value, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame")
    return value


def _require_record_columns(frame: pd.DataFrame, name: str) -> None:
    if not frame.columns.is_unique or any(not isinstance(label, str) for label in frame.columns):
        raise ValueError(f"{name} requires unique string column names for record export")


def _validate_metrics(metrics: pd.DataFrame) -> None:
    _require_record_columns(metrics, "metrics")
    if set(metrics.columns) != set(_METRIC_COLUMNS):
        raise ValueError(f"metrics columns must be exactly {_METRIC_COLUMNS!r}")
    seen: set[str] = set()
    for row in metrics.loc[:, list(_METRIC_COLUMNS)].itertuples(index=False, name=None):
        metric, family, value, status, reason, weighting = row
        for name, text in (("metric", metric), ("family", family), ("weighting", weighting)):
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"metric {name} must be a nonblank string")
        if metric in seen:
            raise ValueError(f"metric identifiers must be unique: {metric!r}")
        seen.add(metric)
        if not isinstance(status, str) or status not in _METRIC_STATUSES:
            raise ValueError(f"invalid status for metric {metric!r}: expected one of {sorted(_METRIC_STATUSES)!r}")
        if status == "ok":
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
                raise ValueError(f"successful metric {metric!r} requires a finite real numeric value")
            try:
                finite = math.isfinite(value)
            except (OverflowError, TypeError, ValueError):
                finite = False
            if not finite:
                raise ValueError(f"successful metric {metric!r} requires a finite real numeric value")
            if not _missing(reason) and not isinstance(reason, str):
                raise ValueError(f"reason for metric {metric!r} must be text or missing")
        else:
            if not _missing(value):
                raise ValueError(f"unavailable metric {metric!r} must have a missing value")
            if not isinstance(reason, str) or not reason.strip():
                raise ValueError(f"unavailable metric {metric!r} requires a nonblank reason")


def _frame_rows(frame: pd.DataFrame) -> list[list[Any]]:
    if not len(frame.columns):
        # pandas' itertuples yields no rows when both columns and index output
        # are absent. Preserve the row dimension of an n-by-zero table.
        return [[] for _ in range(len(frame))]
    return [list(row) for row in frame.itertuples(index=False, name=None)]


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [dict(zip(frame.columns, values)) for values in _frame_rows(frame)]


def _split_table(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "columns": list(frame.columns),
        "index": list(frame.index),
        "index_names": list(frame.index.names),
        "data": _frame_rows(frame),
        "attrs": deepcopy(frame.attrs),
    }


@dataclass(frozen=True, slots=True, init=False)
class DiagnosticResult:
    """An immutable public snapshot of a completed metadata diagnosis.

    Returned DataFrames are independent copies, including nested object cells.
    Unsupported cell/attribute types can be held as detached Python objects, but
    exporting them raises TypeError rather than substituting a display string.
    """

    _validation: ValidationReport
    _metrics: pd.DataFrame = field(repr=False, compare=False)
    _tables: Mapping[str, pd.DataFrame] = field(repr=False, compare=False)
    _audit: pd.DataFrame = field(repr=False, compare=False)
    package_version: str

    def __init__(
        self, *, validation: ValidationReport, metrics: pd.DataFrame,
        tables: Mapping[str, pd.DataFrame], audit: pd.DataFrame,
    ) -> None:
        if not isinstance(validation, ValidationReport):
            raise TypeError("validation must be a ValidationReport")
        validation.raise_for_errors()
        _validate_metrics(_require_frame(metrics, "metrics"))
        _require_record_columns(_require_frame(audit, "audit"), "audit")
        if not isinstance(tables, Mapping):
            raise TypeError("tables must map names to pandas DataFrames")
        copied_tables: dict[str, pd.DataFrame] = {}
        for name, table in tables.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("table names must be nonblank strings")
            copied_tables[name] = _copy_frame(_require_frame(table, f"table {name!r}"))
        object.__setattr__(self, "_validation", validation)
        object.__setattr__(self, "_metrics", _copy_frame(metrics.loc[:, list(_METRIC_COLUMNS)]))
        object.__setattr__(self, "_tables", MappingProxyType(copied_tables))
        object.__setattr__(self, "_audit", _copy_frame(audit))
        object.__setattr__(self, "package_version", version(_PACKAGE_NAME))

    @property
    def validation(self) -> ValidationReport:
        return self._validation

    @property
    def config(self) -> AnalysisConfig:
        return self._validation.config

    @property
    def metrics(self) -> pd.DataFrame:
        return _copy_frame(self._metrics)

    @property
    def audit(self) -> pd.DataFrame:
        return _copy_frame(self._audit)

    @property
    def tables(self) -> Mapping[str, pd.DataFrame]:
        return MappingProxyType({name: _copy_frame(table) for name, table in self._tables.items()})

    def to_dict(self) -> dict[str, Any]:
        """Return strict-JSON-compatible records and tables with structured keys."""
        return _json_value({
            "schema_version": self.config.schema_version,
            "scope": "metadata_diagnostics",
            "package": {"name": _PACKAGE_NAME, "version": self.package_version},
            "config": self.config.to_dict(),
            "validation": self.validation.to_dict(),
            "metrics": _records(self._metrics),
            "tables": {name: _split_table(table) for name, table in self._tables.items()},
            "audit": _records(self._audit),
        })

    def to_json(self) -> str:
        """Return strict JSON text without writing a file."""
        return json.dumps(self.to_dict(), ensure_ascii=False, allow_nan=False, indent=2)
