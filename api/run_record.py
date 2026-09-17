"""Versioned execution records and explicit content verification.

Scientific reports retain their existing format. File fingerprints identify the
bytes read, not scientific validity or authenticity. DataFrames have no raw-file
fingerprint. Verification reads only paths explicitly supplied by the caller;
it does not rerun an analysis or recreate an environment.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from os import PathLike
from pathlib import Path
import platform
from typing import Any

import numpy as np
import pandas as pd

from .. import __version__
from ._tabular import read_metadata_with_fingerprint
from .config import AnalysisConfig, read_config_with_fingerprint


_SCHEMA = "1"
_INPUT_KEYS = {"kind", "name", "sha256", "size_bytes", "format", "encoding", "delimiter"}


class RunRecordError(ValueError):
    """An unsupported, malformed, or internally inconsistent run record."""


def _canonical(value: Any) -> str:
    """Deterministic JSON: sorted object keys, preserved array order, ASCII escapes."""
    def check(item: Any) -> None:
        if isinstance(item, dict):
            if any(not isinstance(key, str) for key in item):
                raise RunRecordError("JSON object keys must be strings")
            for child in item.values():
                check(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                check(child)
    try:
        check(value)
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    except (TypeError, ValueError, RecursionError) as exc:
        raise RunRecordError(f"Cannot represent record content as strict JSON: {exc}") from exc


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in items:
        if key in result:
            raise RunRecordError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _constant(value: str) -> None:
    raise RunRecordError(f"Nonfinite JSON number: {value}")


def _read_json(path: str | PathLike[str]) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8-sig"), object_pairs_hook=_pairs, parse_constant=_constant)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise RunRecordError(f"Cannot read strict UTF-8 JSON from {path}: {exc}") from exc


def _object(value: Any, keys: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise RunRecordError(f"{name} must be an object with exactly these fields: {', '.join(sorted(keys))}")
    return value


def _text(value: Any, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise RunRecordError(f"{name} must be a nonblank string")


def _sha(value: Any, name: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise RunRecordError(f"{name} must be a lowercase SHA-256 hexadecimal digest")


def _file_fields(value: dict[str, Any], name: str) -> None:
    _text(value["name"], f"{name}.name")
    _sha(value["sha256"], f"{name}.sha256")
    if type(value["size_bytes"]) is not int or value["size_bytes"] < 0:
        raise RunRecordError(f"{name}.size_bytes must be a nonnegative integer")


def _environment() -> dict[str, str]:
    # Module versions describe the code loaded into this process. No optional
    # numerical packages are imported just to collect environment information.
    return {"python": platform.python_version(), "package": __version__,
            "numpy": np.__version__, "pandas": pd.__version__}


def _validate_record(value: dict[str, Any]) -> None:
    _object(value, {"schema_version", "scope", "command", "exit_code", "input", "configuration", "environment", "report"}, "run record")
    if value["schema_version"] != _SCHEMA or value["scope"] != "run_record":
        raise RunRecordError("Unsupported run-record schema_version or scope")
    command, code = value["command"], value["exit_code"]
    if command not in ("validate", "diagnose") or type(code) is not int or code not in (0, 2, 3, 4):
        raise RunRecordError("Invalid command or analysis exit_code")
    if command == "validate" and code == 4:
        raise RunRecordError("Validation cannot have a resource-limited diagnosis exit code")
    input_info = _object(value["input"], _INPUT_KEYS, "input")
    if input_info["kind"] == "file":
        _file_fields(input_info, "input")
        if input_info["format"] not in ("csv", "tsv") or input_info["encoding"] != "utf-8-sig":
            raise RunRecordError("Unsupported input format or encoding")
        if input_info["delimiter"] != ("," if input_info["format"] == "csv" else "\t"):
            raise RunRecordError("Input delimiter disagrees with its format")
    elif input_info["kind"] == "dataframe":
        if any(input_info[key] is not None for key in _INPUT_KEYS - {"kind"}):
            raise RunRecordError("DataFrame inputs cannot claim raw-file fingerprints or parsing settings")
    else:
        raise RunRecordError("Input kind must be file or dataframe")
    configuration = _object(value["configuration"], {"file", "resolved", "resolved_sha256"}, "configuration")
    if configuration["file"] is not None:
        _file_fields(_object(configuration["file"], {"name", "sha256", "size_bytes"}, "configuration.file"), "configuration.file")
    try:
        resolved = AnalysisConfig.from_dict(configuration["resolved"]).to_dict()
    except (TypeError, ValueError) as exc:
        raise RunRecordError(f"Invalid resolved configuration: {exc}") from exc
    if resolved != configuration["resolved"]:
        raise RunRecordError("Resolved configuration must include all defaults")
    _sha(configuration["resolved_sha256"], "configuration.resolved_sha256")
    if _digest(resolved) != configuration["resolved_sha256"]:
        raise RunRecordError("Resolved configuration fingerprint does not match its content")
    for key, item in _object(value["environment"], {"python", "package", "numpy", "pandas"}, "environment").items():
        _text(item, f"environment.{key}")
    report = _object(value["report"], {"scope", "sha256"}, "report")
    _sha(report["sha256"], "report.sha256")
    expected_scope = ("diagnostic_error" if code == 3 else "metadata_validation"
                      if command == "validate" or code == 2 else "metadata_diagnostics")
    if report["scope"] != expected_scope:
        raise RunRecordError("Report scope disagrees with command and analysis exit_code")


@dataclass(frozen=True, slots=True, init=False)
class RunRecord:
    """Immutable JSON record; loading preserves the originally recorded versions."""

    _json: str

    def __init__(self, value: Mapping[str, Any]) -> None:
        if not isinstance(value, Mapping):
            raise RunRecordError("Run record must be an object")
        encoded = _canonical(dict(value))
        _validate_record(json.loads(encoded))
        object.__setattr__(self, "_json", encoded)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> RunRecord:
        return cls(value)

    @classmethod
    def from_json(cls, path: str | PathLike[str]) -> RunRecord:
        return cls(_read_json(path))

    def to_dict(self) -> dict[str, Any]:
        return json.loads(self._json)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=True, allow_nan=False) + "\n"


@dataclass(frozen=True, slots=True, init=False)
class RecordedRun:
    """Detached scientific report and its companion execution record.

    ``exit_code`` describes analysis completion, before any caller writes files.
    Invalid metadata and computation failures are returned as reports. Invalid
    configuration and unreadable input files raise their existing exceptions.
    """

    _report_json: str
    record: RunRecord

    def __init__(self, report: Mapping[str, Any], record: RunRecord) -> None:
        if not isinstance(report, Mapping) or not isinstance(record, RunRecord):
            raise TypeError("RecordedRun requires a report mapping and RunRecord")
        encoded = _canonical(dict(report))
        payload, provenance = json.loads(encoded), record.to_dict()
        if payload.get("scope") != provenance["report"]["scope"] or _digest(payload) != provenance["report"]["sha256"]:
            raise RunRecordError("Scientific report does not match its run record")
        if payload.get("config") != provenance["configuration"]["resolved"]:
            raise RunRecordError("Scientific report configuration does not match its run record")
        code = provenance["exit_code"]
        if payload["scope"] == "metadata_validation":
            valid = payload.get("valid")
            if type(valid) is not bool or code != (0 if valid else 2):
                raise RunRecordError("Validation outcome disagrees with the recorded exit code")
        elif payload["scope"] == "metadata_diagnostics":
            metrics = payload.get("metrics")
            if not isinstance(metrics, list) or any(not isinstance(item, dict) for item in metrics):
                raise RunRecordError("Diagnostic report metrics must be a list of records")
            limited = any(item.get("status") == "resource_limited" for item in metrics)
            validation = payload.get("validation")
            if (code != (4 if limited else 0) or not isinstance(validation, dict)
                    or validation.get("valid") is not True):
                raise RunRecordError("Diagnostic outcome disagrees with validation or recorded exit code")
        if "package" in payload:
            package = payload["package"]
            if not isinstance(package, dict) or package.get("version") != provenance["environment"]["package"]:
                raise RunRecordError("Report package version disagrees with the recorded environment")
        object.__setattr__(self, "_report_json", encoded)
        object.__setattr__(self, "record", record)

    @property
    def report(self) -> dict[str, Any]:
        return json.loads(self._report_json)

    @property
    def exit_code(self) -> int:
        return self.record.to_dict()["exit_code"]

    def to_dict(self) -> dict[str, Any]:
        return {"report": self.report, "run_record": self.record.to_dict()}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=True, allow_nan=False) + "\n"


def _failure_report(exc: Exception, config: AnalysisConfig, command: str, validation) -> dict[str, Any]:
    from .results import DiagnosticComputationError
    if isinstance(exc, DiagnosticComputationError) and exc.validation is not None:
        validation = exc.validation
    payload = {
        "schema_version": config.schema_version, "scope": "diagnostic_error",
        "stage": exc.stage if isinstance(exc, DiagnosticComputationError) else command,
        "error": {"type": type(exc).__name__, "message": str(exc)},
        "validation": validation.to_dict() if validation is not None else None,
        "config": config.to_dict(),
        "package": {"name": "provenance-aliasing-diagnostics", "version": __version__},
    }
    if exc.__cause__ is not None:
        payload["cause"] = {"type": type(exc.__cause__).__name__, "message": str(exc.__cause__)}
    return payload


def run_analysis(
    metadata: pd.DataFrame | str | PathLike[str], *,
    config: AnalysisConfig | str | PathLike[str], command: str = "diagnose", format: str | None = None,
) -> RecordedRun:
    """Run the public API while capturing a separate versioned execution record.

    Passing file paths captures their exact bytes during parsing. A DataFrame
    records its input kind without inventing an original-file fingerprint. The
    strict reader preserves text identifiers. Existing validate/diagnose calls
    and their scientific report formats are unchanged.
    """
    import provenance_aliasing.api as public_api
    if command not in ("validate", "diagnose"):
        raise ValueError("command must be validate or diagnose")
    if isinstance(metadata, pd.DataFrame) and format is not None:
        raise ValueError("format applies only to metadata files, not DataFrames")
    if isinstance(config, AnalysisConfig):
        configuration, config_file = config, None
    else:
        configuration, config_file = read_config_with_fingerprint(config)
    if isinstance(metadata, pd.DataFrame):
        frame = metadata
        input_info = {key: None for key in _INPUT_KEYS}
        input_info["kind"] = "dataframe"
    else:
        frame, input_info = read_metadata_with_fingerprint(metadata, format=format)
    environment = _environment()
    validation = None
    try:
        if command == "validate":
            validation = public_api.validate(frame, config=configuration)
            payload = validation.to_dict()
            code = 0 if validation.valid else 2
        else:
            result = public_api.diagnose(frame, config=configuration)
            validation = result.validation
            payload = result.to_dict()
            code = 4 if any(metric["status"] == "resource_limited" for metric in payload["metrics"]) else 0
    except public_api.MetadataValidationError as exc:
        payload, code = exc.report.to_dict(), 2
    except Exception as exc:
        payload, code = _failure_report(exc, configuration, command, validation), 3
    resolved = configuration.to_dict()
    record = RunRecord({
        "schema_version": _SCHEMA, "scope": "run_record", "command": command, "exit_code": code,
        "input": input_info,
        "configuration": {"file": config_file, "resolved": resolved, "resolved_sha256": _digest(resolved)},
        "environment": environment, "report": {"scope": payload["scope"], "sha256": _digest(payload)},
    })
    return RecordedRun(payload, record)


@dataclass(frozen=True, slots=True, init=False)
class VerificationReport:
    """Content checks, with environment differences reported separately.

    ``verified`` requires matching metadata and configuration, plus matching
    report content if supplied. It does not establish scientific correctness,
    authenticity, or reproduction of calculations. Omitted report content is
    explicitly marked not_checked. DataFrames have no verifiable raw-file hash.
    """

    _json: str

    def __init__(self, value: Mapping[str, Any]) -> None:
        object.__setattr__(self, "_json", _canonical(dict(value)))

    @property
    def verified(self) -> bool:
        return self.to_dict()["verified"]

    def to_dict(self) -> dict[str, Any]:
        return json.loads(self._json)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=True, allow_nan=False) + "\n"


def _file_digest(path: str | PathLike[str]) -> tuple[str, int]:
    digest, size = hashlib.sha256(), 0
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def verify_run(
    record: RunRecord | str | PathLike[str], *, metadata: str | PathLike[str] | None = None,
    config: AnalysisConfig | str | PathLike[str] | None = None,
    report: Mapping[str, Any] | str | PathLike[str] | None = None,
) -> VerificationReport:
    """Compare explicitly supplied inputs and optional report against a record.

    File identity includes exact bytes; cosmetic configuration-file changes can
    fail that check while preserving the resolved-configuration match. Recorded
    names are informational and are never followed as paths. Candidate-file
    problems become failed checks; malformed reference records raise errors.
    """
    saved = (record if isinstance(record, RunRecord) else RunRecord.from_json(record)).to_dict()
    checks = []

    def add(name: str, status: str, message: str, *, required: bool = True) -> None:
        checks.append({"name": name, "status": status, "required": required, "message": message})

    if saved["input"]["kind"] != "file":
        add("metadata", "unavailable", "This DataFrame run has no original-file fingerprint to verify.")
    elif metadata is None:
        add("metadata", "not_checked", "No metadata file was supplied.")
    else:
        try:
            digest, size = _file_digest(metadata)
            matches = digest == saved["input"]["sha256"] and size == saved["input"]["size_bytes"]
            add("metadata", "match" if matches else "mismatch", "Metadata bytes match." if matches else "Metadata bytes differ.")
        except (OSError, ValueError, TypeError) as exc:
            add("metadata", "error", str(exc))

    original_file = saved["configuration"]["file"]
    needs_config_file = original_file is not None
    current_config, current_file = None, None
    config_error = None
    if config is not None:
        try:
            if isinstance(config, AnalysisConfig):
                current_config = config
            else:
                current_config, current_file = read_config_with_fingerprint(config)
        except (OSError, ValueError, TypeError) as exc:
            config_error = str(exc)
    if not needs_config_file:
        add("configuration_file", "unavailable", "Configuration was supplied as an object; no raw-file fingerprint was recorded.", required=False)
    elif config_error is not None:
        add("configuration_file", "error", config_error)
    elif current_file is None:
        add("configuration_file", "not_checked", "The original configuration-file fingerprint requires a supplied file.")
    else:
        matches = all(current_file[key] == original_file[key] for key in ("sha256", "size_bytes"))
        add("configuration_file", "match" if matches else "mismatch", "Configuration bytes match." if matches else "Configuration bytes differ.")
    if config_error is not None:
        add("configuration", "error", config_error)
    elif current_config is None:
        add("configuration", "not_checked", "No configuration was supplied.")
    else:
        matches = _digest(current_config.to_dict()) == saved["configuration"]["resolved_sha256"]
        add("configuration", "match" if matches else "mismatch", "Resolved settings match." if matches else "Resolved settings differ.")

    if report is None:
        add("report", "not_checked", "Report contents were not supplied and have not been verified.", required=False)
    else:
        try:
            payload = dict(report) if isinstance(report, Mapping) else _read_json(report)
            matches = (isinstance(payload, dict) and payload.get("scope") == saved["report"]["scope"]
                       and _digest(payload) == saved["report"]["sha256"])
            if matches:
                RecordedRun(payload, RunRecord(saved))
            add("report", "match" if matches else "mismatch", "Report content matches." if matches else "Report content differs.")
        except (OSError, ValueError, TypeError) as exc:
            add("report", "error", str(exc))
    current_environment = _environment()
    return VerificationReport({
        "schema_version": _SCHEMA, "scope": "run_verification",
        "verified": all(check["status"] == "match" for check in checks if check["required"]),
        "checks": checks,
        "environment": {"matches": current_environment == saved["environment"],
                        "recorded": saved["environment"], "current": current_environment},
    })
