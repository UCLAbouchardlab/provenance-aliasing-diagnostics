"""Immutable, validated configuration for the metadata API.

This module has no numerical dependencies and does not import or modify the
reference core. JSON configuration and direct Python construction share validation.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import MISSING, dataclass, field, fields
import hashlib
import json
import math
from os import PathLike
from pathlib import Path
from types import MappingProxyType
from typing import Any, TypeVar


SCHEMA_VERSION = "0.1-draft"
_DEFAULT_DIAGNOSTICS = ("design", "entropy", "ceiling", "structure", "guards")
_DIAGNOSTICS = (*_DEFAULT_DIAGNOSTICS, "entropy_loso")


class ConfigurationError(ValueError):
    """An invalid configuration value, identified by its field path."""

    def __init__(self, message: str, *, path: str = "$") -> None:
        self.path = path
        super().__init__(f"{path}: {message}")


def _child(path: str, key: str) -> str:
    return key if path == "$" else f"{path}.{key}"


def _text(value: Any, path: str, *, trimmed: bool = False) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError("must be a nonblank string", path=path)
    if trimmed and value != value.strip():
        raise ConfigurationError("must not have surrounding whitespace", path=path)
    return value


def _choice(value: Any, choices: tuple[str, ...], path: str) -> str:
    if not isinstance(value, str) or value not in choices:
        raise ConfigurationError(f"must be one of {choices!r}", path=path)
    return value


def _integer(value: Any, path: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ConfigurationError(f"must be an integer >= {minimum}", path=path)
    return value


def _strings(
    value: Any, path: str, *, allow_empty: bool = False,
    choices: tuple[str, ...] | None = None, trimmed: bool = False,
) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ConfigurationError("must be an array of strings", path=path)
    if not allow_empty and not value:
        raise ConfigurationError("must contain at least one entry", path=path)
    result: list[str] = []
    for index, item in enumerate(value):
        item_path = f"{path}[{index}]"
        item = _text(item, item_path, trimmed=trimmed)
        if choices is not None:
            _choice(item, choices, item_path)
        if item in result:
            raise ConfigurationError(f"duplicate entry {item!r}", path=item_path)
        result.append(item)
    return tuple(result)


T = TypeVar("T")


def _construct(cls: type[T], value: Any, path: str) -> T:
    if isinstance(value, cls):
        return value
    if not isinstance(value, Mapping):
        raise ConfigurationError("must be an object", path=path)
    declared = {item.name: item for item in fields(cls)}
    for key in value:
        if not isinstance(key, str):
            raise ConfigurationError("object keys must be strings", path=path)
        if key not in declared:
            raise ConfigurationError("unknown field", path=_child(path, key))
    for name, item in declared.items():
        if item.default is MISSING and item.default_factory is MISSING and name not in value:
            raise ConfigurationError("required field is missing", path=_child(path, name))
    # An explicit null column is also an undeclared field in incidence weighting.
    if cls is WeightingConfig and value.get("mode", "incidence") == "incidence" and "column" in value:
        raise ConfigurationError("incidence weighting does not accept a column", path="weighting.column")
    return cls(**dict(value))


@dataclass(frozen=True, slots=True)
class ColumnMapping:
    source: tuple[str, ...]
    unit: tuple[str, ...]
    group: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("source", "unit", "group"):
            object.__setattr__(self, name, _strings(getattr(self, name), f"columns.{name}"))


@dataclass(frozen=True, slots=True)
class ProvenanceGrain:
    name: str
    description: str

    def __post_init__(self) -> None:
        _text(self.name, "grain.name")
        _text(self.description, "grain.description")


@dataclass(frozen=True, slots=True)
class WeightingConfig:
    mode: str = "incidence"
    column: str | None = None
    description: str | None = None

    def __post_init__(self) -> None:
        _choice(self.mode, ("incidence", "column"), "weighting.mode")
        if self.mode == "column":
            _text(self.column, "weighting.column")
            _text(self.description, "weighting.description")
        else:
            if self.column is not None:
                raise ConfigurationError("incidence weighting does not accept a column", path="weighting.column")
            if self.description is not None:
                _text(self.description, "weighting.description")


@dataclass(frozen=True, slots=True)
class ValidationPolicies:
    missing_required: str = "error"
    duplicates: str = "error"
    missing_parent: str = "report"
    missing_tokens: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _choice(self.missing_required, ("error", "drop"), "policies.missing_required")
        _choice(self.duplicates, ("error", "collapse_identical", "keep_records"), "policies.duplicates")
        _choice(self.missing_parent, ("report", "error"), "policies.missing_parent")
        object.__setattr__(self, "missing_tokens", _strings(
            self.missing_tokens, "policies.missing_tokens", allow_empty=True, trimmed=True,
        ))


@dataclass(frozen=True, slots=True)
class DiagnosticSelection:
    include: tuple[str, ...] = _DEFAULT_DIAGNOSTICS
    bootstrap_replicates: int = 0
    seed: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "include", _strings(
            self.include, "diagnostics.include", allow_empty=True, choices=_DIAGNOSTICS,
        ))
        _integer(self.bootstrap_replicates, "diagnostics.bootstrap_replicates")
        _integer(self.seed, "diagnostics.seed")


@dataclass(frozen=True, slots=True)
class ExecutionConfig:
    max_working_memory_mb: int = 512

    def __post_init__(self) -> None:
        _integer(self.max_working_memory_mb, "execution.max_working_memory_mb", minimum=1)


class _ObjectPairs(list):
    """Distinguish JSON object pairs from arrays until duplicate checks finish."""


def _json_value(value: Any, path: str = "$") -> Any:
    if isinstance(value, _ObjectPairs):
        result: dict[str, Any] = {}
        for key, item in value:
            item_path = _child(path, key)
            if key in result:
                raise ConfigurationError("duplicate JSON key", path=item_path)
            result[key] = _json_value(item, item_path)
        return result
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ConfigurationError("object keys must be strings", path=path)
            result[key] = _json_value(item, _child(path, key))
        return result
    if isinstance(value, (list, tuple)):
        return [_json_value(item, f"{path}[{index}]") for index, item in enumerate(value)]
    if isinstance(value, float) and not math.isfinite(value):
        raise ConfigurationError("nonfinite numbers are not permitted", path=path)
    return value


@dataclass(frozen=True, slots=True)
class AnalysisConfig:
    """Resolved draft configuration shared by Python and CLI workflows.

    Nested mappings supplied to direct construction are validated and copied.
    Diagnostic/execution settings describe the requested analysis; parsing them
    does not execute diagnostics or impose a process-wide memory limit.
    """

    schema_version: str
    name: str
    columns: ColumnMapping
    grain: ProvenanceGrain
    unit_description: str
    weighting: WeightingConfig = field(default_factory=WeightingConfig)
    hierarchy: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    policies: ValidationPolicies = field(default_factory=ValidationPolicies)
    diagnostics: DiagnosticSelection = field(default_factory=DiagnosticSelection)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)

    def __post_init__(self) -> None:
        _choice(self.schema_version, (SCHEMA_VERSION,), "schema_version")
        _text(self.name, "name")
        _text(self.unit_description, "unit_description")
        for name, cls in (
            ("columns", ColumnMapping), ("grain", ProvenanceGrain),
            ("weighting", WeightingConfig), ("policies", ValidationPolicies),
            ("diagnostics", DiagnosticSelection), ("execution", ExecutionConfig),
        ):
            object.__setattr__(self, name, _construct(cls, getattr(self, name), name))
        if not isinstance(self.hierarchy, Mapping):
            raise ConfigurationError("must be an object", path="hierarchy")
        hierarchy: dict[str, tuple[str, ...]] = {}
        for level, columns in self.hierarchy.items():
            _text(level, "hierarchy", trimmed=True)
            hierarchy[level] = _strings(columns, f"hierarchy.{level}")
        object.__setattr__(self, "hierarchy", MappingProxyType(hierarchy))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> AnalysisConfig:
        if not isinstance(value, Mapping):
            raise ConfigurationError("must be an object")
        try:
            return _construct(cls, _json_value(value), "$")
        except RecursionError as exc:
            raise ConfigurationError("configuration nesting is too deep or contains recursive values") from exc

    @classmethod
    def from_json(cls, path: str | PathLike[str]) -> AnalysisConfig:
        """Read UTF-8 JSON, allowing a BOM and rejecting duplicate/nonfinite data.

        File-system errors remain OSError subclasses so callers can distinguish
        an unavailable file from an invalid configuration.
        """
        return _read_config_with_fingerprint(path, cls)[0]

    def to_dict(self) -> dict[str, Any]:
        """Return a fresh, JSON-compatible copy including resolved defaults."""
        weighting = {"mode": self.weighting.mode}
        if self.weighting.column is not None:
            weighting["column"] = self.weighting.column
        if self.weighting.description is not None:
            weighting["description"] = self.weighting.description
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "columns": {name: list(getattr(self.columns, name)) for name in ("source", "unit", "group")},
            "grain": {"name": self.grain.name, "description": self.grain.description},
            "unit_description": self.unit_description,
            "weighting": weighting,
            "hierarchy": {level: list(columns) for level, columns in self.hierarchy.items()},
            "policies": {
                "missing_required": self.policies.missing_required,
                "duplicates": self.policies.duplicates,
                "missing_parent": self.policies.missing_parent,
                "missing_tokens": list(self.policies.missing_tokens),
            },
            "diagnostics": {
                "include": list(self.diagnostics.include),
                "bootstrap_replicates": self.diagnostics.bootstrap_replicates,
                "seed": self.diagnostics.seed,
            },
            "execution": {"max_working_memory_mb": self.execution.max_working_memory_mb},
        }

    def to_json(self) -> str:
        """Return strict JSON text; this method does not write files."""
        return json.dumps(self.to_dict(), ensure_ascii=False, allow_nan=False, indent=2) + "\n"


def _read_config_with_fingerprint(
    path: str | PathLike[str], config_type: type[AnalysisConfig],
) -> tuple[AnalysisConfig, dict[str, str | int]]:
    source = Path(path)
    content = source.read_bytes()
    try:
        # Keep the former text reader's universal-newline behavior while the
        # fingerprint records the unmodified bytes from the same read.
        text = content.decode("utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")
        value = json.loads(text, object_pairs_hook=_ObjectPairs)
        config = config_type.from_dict(_json_value(value))
    except UnicodeError as exc:
        raise ConfigurationError("configuration must use UTF-8 encoding") from exc
    except json.JSONDecodeError as exc:
        raise ConfigurationError(f"invalid JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}") from exc
    except RecursionError as exc:
        raise ConfigurationError("configuration nesting is too deep or contains recursive values") from exc
    return config, {"name": source.name, "sha256": hashlib.sha256(content).hexdigest(), "size_bytes": len(content)}


def read_config_with_fingerprint(path: str | PathLike[str]) -> tuple[AnalysisConfig, dict[str, str | int]]:
    """Read validated configuration and hash the exact bytes used for parsing.

    The file is read once. Its BOM and original line endings are included in
    the fingerprint, and only its basename is recorded as the file identity.
    """
    return _read_config_with_fingerprint(path, AnalysisConfig)
