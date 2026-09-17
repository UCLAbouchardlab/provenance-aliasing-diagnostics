"""Assay-independent checks of explicitly mapped observation metadata.

The API assesses a proposed metadata projection. It never edits the input or
returns a cleaned table. In particular, a valid report with drop/collapse or
missing-token policies does not mean the original table can be passed unchanged
to the legacy core. Diagnostic selections and execution budgets are preserved
in the report, but are not executed by this metadata-only check.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import json
import math
from numbers import Complex, Real
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .config import AnalysisConfig


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    """One issue, using zero-based row positions rather than index labels."""

    code: str
    severity: str
    message: str
    row_positions: tuple[int, ...] = ()
    columns: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.severity not in {"error", "warning", "info"}:
            raise ValueError("severity must be error, warning, or info")
        object.__setattr__(self, "row_positions", tuple(self.row_positions))
        object.__setattr__(self, "columns", tuple(self.columns))

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "row_positions": list(self.row_positions),
            "columns": list(self.columns),
        }


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """Immutable metadata assessment and its resolved configuration.

    ``n_key_complete`` counts rows with usable source, unit, and group keys.
    ``n_retained`` counts those rows after any requested duplicate collapse;
    they may still have errors. ``n_excluded`` counts only explicit missing-key
    drops and duplicate collapses. ``n_error_records`` counts distinct positions
    cited by errors and can overlap retained rows. Counts are not a partition
    when errors exist. Counts not assessed after a schema failure are ``None``.
    Source/unit/group counts describe retained keys, including zero-weight rows.
    ``total_weight`` is ``None`` when retained weights cannot be interpreted.
    """

    config: AnalysisConfig
    issues: tuple[ValidationIssue, ...]
    counts: Mapping[str, int | float | None]

    def __post_init__(self) -> None:
        object.__setattr__(self, "issues", tuple(self.issues))
        object.__setattr__(self, "counts", MappingProxyType(dict(self.counts)))

    @property
    def valid(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.config.schema_version,
            "scope": "metadata_validation",
            "valid": self.valid,
            "config": self.config.to_dict(),
            "counts": dict(self.counts),
            "issues": [issue.to_dict() for issue in self.issues],
        }

    def to_json(self) -> str:
        """Return strict JSON, with unassessed quantities represented as null."""
        return json.dumps(self.to_dict(), ensure_ascii=False, allow_nan=False, indent=2)

    def raise_for_errors(self) -> None:
        """Raise with this report attached if any error prevents validation."""
        if not self.valid:
            raise MetadataValidationError(self)


class MetadataValidationError(ValueError):
    """Failed validation; the complete structured assessment is in ``report``."""

    def __init__(self, report: ValidationReport) -> None:
        self.report = report
        errors = [issue for issue in report.issues if issue.severity == "error"]
        super().__init__("Metadata validation failed: " + "; ".join(i.message for i in errors))


@dataclass(frozen=True, slots=True)
class _Assessment:
    """Private products of one metadata assessment, owned by the caller.

    Projection and audit tables are supplied only for valid metadata. The public
    validation report remains independent of these mutable preparation products.
    """

    report: ValidationReport
    projection: pd.DataFrame
    audit: pd.DataFrame


_AUDIT_COLUMNS = (
    "row_position", "status", "reason", "representative_row_position",
    "projection_row_position", "normalized_columns",
)


def _empty_assessment(report: ValidationReport) -> _Assessment:
    projection = pd.DataFrame({
        **{role: pd.Series(dtype=object) for role in ("source", "unit", "group")},
        "weight": pd.Series(dtype=float),
        **{f"hierarchy::{level}": pd.Series(dtype=object) for level in report.config.hierarchy},
    })
    audit = pd.DataFrame({
        column: pd.Series(dtype="int64" if column == "row_position" else object)
        for column in _AUDIT_COLUMNS
    })
    return _Assessment(report=report, projection=projection, audit=audit)


def _is_missing(value: Any) -> bool:
    if not pd.api.types.is_scalar(value):
        return False
    if isinstance(value, Decimal):
        # Signaling NaNs must be rejected by value validation, not evaluated by
        # pandas' missingness routines, which differ across supported versions.
        return value.is_qnan()
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError, InvalidOperation):
        return False


def _scalar_key(value: Any, missing_tokens: frozenset[str]) -> tuple[Any, str | None, bool]:
    if not pd.api.types.is_scalar(value):
        return None, "invalid_key", False
    if _is_missing(value):
        return None, None, False
    if isinstance(value, Decimal) and not value.is_finite():
        return None, "invalid_key", False
    if isinstance(value, Complex) and not isinstance(value, Real):
        return None, "invalid_key", False
    if isinstance(value, (float, np.floating)) and not math.isfinite(value):
        return None, "invalid_key", False
    key = str(value).strip()
    changed = not isinstance(value, str) or key != value
    if not key or key in missing_tokens:
        return None, None, changed
    if key.lower() in {"none", "nan", "<na>"}:
        return None, "core_reserved_key", changed
    return key, None, changed


def _key(value: Any, tokens: frozenset[str]) -> tuple[Any, str | None, bool]:
    if not isinstance(value, tuple):
        return _scalar_key(value, tokens)
    if not value:
        return None, "invalid_key", False
    parts = [_scalar_key(part, tokens) for part in value]
    error = next((error for _, error, _ in parts if error), None)
    complete = all(part is not None for part, _, _ in parts)
    return (tuple(part for part, _, _ in parts) if complete else None,
            error, any(changed for _, _, changed in parts))


def _weight(value: Any) -> float | None:
    # NumPy represents some duration units as numbers under float conversion.
    if isinstance(value, (np.datetime64, np.timedelta64)):
        return None
    if not pd.api.types.is_scalar(value) or _is_missing(value):
        return None
    if isinstance(value, (bool, np.bool_)):
        return None
    if isinstance(value, Complex) and not isinstance(value, Real):
        return None
    # Numeric strings are accepted, but timestamps and arbitrary float-like
    # objects are not a declared numerical mass.
    if not isinstance(value, (str, Real, np.number, Decimal)):
        return None
    try:
        number = float(value)
    except (ValueError, TypeError, OverflowError, InvalidOperation):
        return None
    return number if math.isfinite(number) and number >= 0 else None


def _same_cell(left: Any, right: Any) -> bool:
    """Conservative whole-record equality for optional, unmapped metadata."""
    if _is_missing(left) or _is_missing(right):
        return _is_missing(left) and _is_missing(right)
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(_same_cell(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(_same_cell(a, b) for a, b in zip(left, right))
    try:
        equal = left == right
        return isinstance(equal, (bool, np.bool_)) and bool(equal)
    except (TypeError, ValueError, InvalidOperation):
        return False


def validate(frame: pd.DataFrame, *, config: AnalysisConfig) -> ValidationReport:
    """Check mapped metadata without modifying it or running diagnostics.

    Keys are categorical identifiers: scalar values are stringified and stripped;
    tuple keys preserve their components. Composite column mappings preserve
    boundaries, so identifiers containing delimiters do not collide. Declared
    missing tokens match case-sensitively after stripping. Literal NA is retained.
    A unit must have one selected group and at most one nonmissing value at each
    named parent level, across all sources. Parent levels are checked separately;
    the config does not declare a chain of relationships between parent levels.

    Malformed metadata yields issues. Wrong Python argument types raise TypeError.
    No memory-size or scientific estimability guarantee follows from ``valid``.
    """
    return _assess(frame, config=config).report


def _assess(frame: pd.DataFrame, *, config: AnalysisConfig) -> _Assessment:
    """Validate and prepare the exact same declared projection in a single pass."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    if not isinstance(config, AnalysisConfig):
        raise TypeError("config must be an AnalysisConfig")

    issues: list[ValidationIssue] = []
    counts: dict[str, int | float | None] = dict.fromkeys((
        "n_key_complete", "n_retained", "n_excluded", "n_duplicate_keys",
        "n_duplicate_rows", "n_sources", "n_units", "n_groups", "total_weight",
    ))
    counts.update(n_records=len(frame), n_error_records=0)

    def add(code: str, severity: str, message: str, rows=(), columns=()) -> None:
        issues.append(ValidationIssue(code, severity, message,
                                      tuple(sorted(set(rows))), tuple(dict.fromkeys(columns))))

    def finish() -> _Assessment:
        counts["n_error_records"] = len({row for issue in issues if issue.severity == "error"
                                        for row in issue.row_positions})
        return _empty_assessment(ValidationReport(config=config, issues=tuple(issues), counts=counts))

    if not frame.columns.is_unique:
        duplicates = tuple(str(c) for c in frame.columns[frame.columns.duplicated()])
        add("duplicate_columns", "error", "Column names must be unique.", columns=duplicates)
    if any(not isinstance(c, str) or not c.strip() for c in frame.columns):
        add("invalid_columns", "error", "Column names must be nonblank strings.")
    if issues:
        return finish()

    roles = {"source": config.columns.source, "unit": config.columns.unit,
             "group": config.columns.group}
    key_columns = tuple(dict.fromkeys(c for cols in (*roles.values(), *config.hierarchy.values()) for c in cols))
    required = list(key_columns)
    if config.weighting.mode == "column":
        required.append(config.weighting.column)
    absent = [column for column in dict.fromkeys(required) if column not in frame.columns]
    if absent:
        add("missing_columns", "error", "Configured columns are absent from the table.", columns=absent)
        return finish()
    if frame.empty:
        counts.update({k: 0 for k in counts})
        counts["total_weight"] = 0.0
        add("empty_input", "error", "The metadata table has no observations.")
        return finish()

    tokens = frozenset(config.policies.missing_tokens)
    normalized: dict[str, list[Any]] = {}
    errors_by_column: dict[str, dict[str, list[int]]] = {}
    changed_by_column: dict[str, list[int]] = {}
    audit_changes: dict[int, set[str]] = defaultdict(set)

    def normalize_column(column: str, positions) -> None:
        if column in normalized:
            return
        values: list[Any] = [None] * len(frame)
        errors: dict[str, list[int]] = defaultdict(list)
        changed_rows: list[int] = []
        series = frame[column]
        for pos in positions:
            raw = series.iloc[pos]
            values[pos], error, changed = _key(raw, tokens)
            if error:
                errors[error].append(pos)
            if changed:
                changed_rows.append(pos)
            # Retain the public issue behavior while auditing transformations to
            # missing keys, including explicitly declared literal tokens.
            if changed or (error is None and values[pos] is None and not _is_missing(raw)):
                audit_changes[pos].add(column)
        normalized[column] = values
        errors_by_column[column] = errors
        changed_by_column[column] = changed_rows

    def project(columns: tuple[str, ...], positions) -> list[Any]:
        for column in columns:
            normalize_column(column, positions)
        values: list[Any] = [None] * len(frame)
        for pos in positions:
            parts = tuple(normalized[c][pos] for c in columns)
            if all(part is not None for part in parts):
                # A tuple-valued physical column is already a complete key.
                # Nesting it inside a multi-column key would not match the core.
                if len(parts) > 1 and any(isinstance(part, tuple) for part in parts):
                    add("invalid_key", "error", "Composite columns must contain scalar key components.",
                        (pos,), columns)
                else:
                    values[pos] = parts[0] if len(parts) == 1 else parts
        scalar_rows = [pos for pos in positions if values[pos] is not None and not isinstance(values[pos], tuple)]
        tuple_rows = [pos for pos in positions if isinstance(values[pos], tuple)]
        if scalar_rows and tuple_rows:
            add("mixed_key_shapes", "error", "A key cannot mix scalar and tuple identifiers.",
                scalar_rows + tuple_rows, columns)
        return values

    positions = range(len(frame))
    keys = {role: project(columns, positions) for role, columns in roles.items()}
    eligible = [pos for pos in positions if all(keys[role][pos] is not None for role in roles)]
    ineligible = set(positions).difference(eligible)
    key_errors = {pos for column in normalized for rows in errors_by_column[column].values() for pos in rows}
    key_errors.update(pos for issue in issues if issue.code == "invalid_key" for pos in issue.row_positions)
    missing = sorted(ineligible.difference(key_errors))
    excluded: set[int] = set()
    if missing:
        drop = config.policies.missing_required == "drop"
        add("missing_required_key", "warning" if drop else "error",
            "Rows with missing source, unit, or group keys will be excluded." if drop else
            "Source, unit, and group keys must be complete.", missing,
            tuple(dict.fromkeys(c for cols in roles.values() for c in cols)))
        if drop:
            excluded.update(missing)

    parents = {level: project(columns, eligible) for level, columns in config.hierarchy.items()}
    for column, errors in errors_by_column.items():
        for code, rows in errors.items():
            message = ("Literal none/nan/<na> identifiers conflict with core normalization; rename them or explicitly declare them missing."
                       if code == "core_reserved_key" else
                       "Keys must contain finite scalar identifiers or tuples of scalar identifiers.")
            add(code, "error", message, rows, (column,))
    for column, rows in changed_by_column.items():
        if rows:
            add("key_normalized", "info", "Key values are compared after string conversion and whitespace stripping.", rows, (column,))

    by_unit: dict[Any, list[int]] = defaultdict(list)
    for pos in eligible:
        by_unit[keys["unit"][pos]].append(pos)
    group_conflicts = [pos for rows in by_unit.values() if len({keys["group"][p] for p in rows}) > 1 for pos in rows]
    if group_conflicts:
        add("conflicting_group", "error", "Each unit must have exactly one selected group across sources.",
            group_conflicts, (*roles["unit"], *roles["group"]))
    for level, parent_keys in parents.items():
        missing_parent = [pos for pos in eligible if parent_keys[pos] is None]
        if missing_parent:
            add("missing_parent", "error" if config.policies.missing_parent == "error" else "warning",
                f"Parent level {level!r} has missing keys; parent-level analyses would be incomplete.",
                missing_parent, config.hierarchy[level])
        conflicts = [pos for rows in by_unit.values()
                     if len({parent_keys[p] for p in rows if parent_keys[p] is not None}) > 1 for pos in rows]
        if conflicts:
            add("conflicting_parent", "error", f"A unit maps to multiple keys at parent level {level!r}.",
                conflicts, (*roles["unit"], *config.hierarchy[level]))

    weights: dict[int, float | None] = {pos: 1.0 for pos in eligible}
    if config.weighting.mode == "column":
        weights = {}
        for pos in eligible:
            raw = frame[config.weighting.column].iloc[pos]
            weights[pos] = _weight(raw)
            if weights[pos] is not None and not isinstance(raw, float):
                audit_changes[pos].add(config.weighting.column)
        invalid = [pos for pos, value in weights.items() if value is None]
        if invalid:
            add("invalid_weight", "error", "Weights must be finite real, nonnegative numbers; booleans are not weights.",
                invalid, (config.weighting.column,))

    by_observation: dict[Any, list[int]] = defaultdict(list)
    for pos in eligible:
        by_observation[tuple(keys[role][pos] for role in roles)].append(pos)
    duplicates = [rows for rows in by_observation.values() if len(rows) > 1]
    counts["n_duplicate_keys"] = len(duplicates)
    counts["n_duplicate_rows"] = sum(len(rows) - 1 for rows in duplicates)

    def same_record(left: int, right: int) -> bool:
        for column in frame.columns:
            if config.weighting.mode == "column" and column == config.weighting.column:
                if weights[left] is None or weights[left] != weights[right]:
                    return False
            elif column in normalized:
                if normalized[column][left] != normalized[column][right]:
                    return False
            elif not _same_cell(frame[column].iloc[left], frame[column].iloc[right]):
                return False
        return True

    collapsed_to: dict[int, int] = {}
    if duplicates:
        rows = [pos for group in duplicates for pos in group]
        columns = tuple(dict.fromkeys(c for cols in roles.values() for c in cols))
        policy = config.policies.duplicates
        if policy == "error":
            add("duplicate_observation", "error", "Repeated source/unit/group keys require an explicit duplicate policy.", rows, columns)
        elif policy == "keep_records":
            add("duplicate_records_retained", "warning", "Repeated observations are retained and each contributes its weight.", rows, columns)
        else:
            for group in duplicates:
                if all(same_record(group[0], pos) for pos in group[1:]):
                    excluded.update(group[1:])
                    collapsed_to.update({pos: group[0] for pos in group[1:]})
                    add("identical_duplicates_collapsed", "warning", "Identical repeated records will be collapsed to their first occurrence.",
                        group[1:], columns)
                else:
                    add("nonidentical_duplicate", "error", "Repeated keys have differing metadata or weights and cannot be collapsed as identical records.",
                        group, columns)

    retained = [pos for pos in eligible if pos not in excluded]
    counts.update(n_key_complete=len(eligible), n_retained=len(retained), n_excluded=len(excluded))
    for role, label in (("source", "n_sources"), ("unit", "n_units"), ("group", "n_groups")):
        counts[label] = len({keys[role][pos] for pos in retained})
    if not retained:
        counts["total_weight"] = 0.0
        add("no_retained_records", "error", "No observations with usable required keys remain.")
    elif all(weights[pos] is not None for pos in retained):
        try:
            total = math.fsum(weights[pos] for pos in retained)
        except OverflowError:
            total = math.inf
        if not math.isfinite(total):
            add("nonfinite_total_weight", "error", "The total observation weight exceeds finite numerical range.", retained,
                (config.weighting.column,) if config.weighting.mode == "column" else ())
        else:
            counts["total_weight"] = total
            if total == 0:
                add("zero_total_weight", "error", "At least one retained observation must have positive weight.", retained)
            else:
                positive_groups = {keys["group"][pos] for pos in retained if weights[pos] > 0}
                if len(positive_groups) < 2:
                    add("no_group_contrast", "warning", "Only one group has positive weight; group contrasts are not available.",
                        retained, roles["group"])
        zero_rows = [pos for pos in retained if weights[pos] == 0]
        if zero_rows:
            add("zero_weight_rows", "warning", "Zero-weight rows count as metadata records but contribute no analytical mass.", zero_rows)
    assessment = finish()
    if not assessment.report.valid:
        return assessment

    # These values are the same objects used for the validation checks above;
    # preparation never reconstructs a projection from issue messages or raw data.
    projection = pd.DataFrame({
        **{role: pd.Series([keys[role][pos] for pos in retained], dtype=object) for role in roles},
        "weight": pd.Series([weights[pos] for pos in retained], dtype=float),
        **{f"hierarchy::{level}": pd.Series([values[pos] for pos in retained], dtype=object)
           for level, values in parents.items()},
    })
    projection_positions = {pos: ordinal for ordinal, pos in enumerate(retained)}
    statuses: list[str] = []
    reasons: list[str] = []
    representatives: list[int | None] = []
    projected_positions: list[int | None] = []
    normalized_columns: list[tuple[str, ...]] = []
    for pos in positions:
        if pos in collapsed_to:
            representative = collapsed_to[pos]
            statuses.append("collapsed")
            reasons.append("identical_duplicate")
        elif pos in projection_positions:
            representative = pos
            statuses.append("retained")
            reasons.append("")
        else:
            representative = None
            statuses.append("excluded")
            reasons.append("missing_required_key")
        representatives.append(representative)
        projected_positions.append(projection_positions.get(representative))
        normalized_columns.append(tuple(column for column in frame.columns if column in audit_changes[pos]))
    audit = pd.DataFrame({
        "row_position": pd.Series(positions, dtype="int64"),
        "status": statuses,
        "reason": reasons,
        # Object dtype preserves absent positions as None rather than floating
        # NaNs, and keeps original positions distinct from projection offsets.
        "representative_row_position": pd.Series(representatives, dtype=object),
        "projection_row_position": pd.Series(projected_positions, dtype=object),
        "normalized_columns": pd.Series(normalized_columns, dtype=object),
    })
    return _Assessment(report=assessment.report, projection=projection, audit=audit)
