"""Content fingerprints and verification of recorded synthetic analyses."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

import provenance_aliasing.api as api


def config_dict(**overrides) -> dict:
    value = {
        "schema_version": "0.1-draft", "name": "recorded-synthetic-atlas",
        "columns": {"source": ["source"], "unit": ["unit"], "group": ["group"]},
        "grain": {"name": "study", "description": "Synthetic originating study"},
        "unit_description": "One synthetic sample",
    }
    value.update(overrides)
    return value


def sample_frame() -> pd.DataFrame:
    return pd.DataFrame({"source": ["S1", "S1", "S2", "S2"],
                         "unit": ["001", "002", "003", "004"],
                         "group": ["A", "B", "A", "B"]})


def write_inputs(tmp_path: Path):
    metadata = tmp_path / "synthetic metadata.csv"
    metadata.write_bytes(b"source,unit,group\nS1,001,A\nS1,002,B\nS2,003,A\nS2,004,B\n")
    configuration = tmp_path / "analysis config.json"
    configuration.write_bytes(json.dumps(config_dict(), indent=2).encode("utf-8"))
    return metadata, configuration


def canonical_digest(payload: dict) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def check(verification, name: str) -> dict:
    matches = [item for item in verification.to_dict()["checks"] if item["name"] == name]
    assert len(matches) == 1
    return matches[0]


def strict_json(text: str):
    def reject(value):
        raise AssertionError(f"Nonstandard JSON constant: {value}")
    return json.loads(text, parse_constant=reject)


def test_file_run_records_exact_input_bytes_resolved_configuration_and_scientific_report(tmp_path) -> None:
    command = "diagnose"
    metadata, configuration = write_inputs(tmp_path)
    recorded = api.run_analysis(metadata, config=configuration, command=command)
    resolved = api.AnalysisConfig.from_json(configuration)
    expected_report = getattr(api, command)(sample_frame(), config=resolved).to_dict()
    assert isinstance(recorded, api.RecordedRun)
    assert recorded.exit_code == 0
    assert recorded.report == expected_report
    record = recorded.record.to_dict()
    assert set(record) == {"schema_version", "scope", "command", "exit_code", "input", "configuration", "environment", "report"}
    assert record["schema_version"] == "1"
    assert record["scope"] == "run_record"
    assert record["command"] == command
    assert record["exit_code"] == 0
    assert record["input"] == {
        "kind": "file", "name": metadata.name, "sha256": hashlib.sha256(metadata.read_bytes()).hexdigest(),
        "size_bytes": len(metadata.read_bytes()), "format": "csv", "encoding": "utf-8-sig", "delimiter": ",",
    }
    assert record["configuration"]["file"] == {
        "name": configuration.name, "sha256": hashlib.sha256(configuration.read_bytes()).hexdigest(),
        "size_bytes": len(configuration.read_bytes()),
    }
    assert record["configuration"]["resolved"] == resolved.to_dict()
    assert record["configuration"]["resolved_sha256"] == canonical_digest(resolved.to_dict())
    assert record["report"] == {"scope": expected_report["scope"], "sha256": canonical_digest(expected_report)}
    assert set(record["environment"]) == {"python", "package", "numpy", "pandas"}
    assert all(isinstance(value, str) and value for value in record["environment"].values())
    assert recorded.to_dict() == {"report": expected_report, "run_record": record}
    assert strict_json(recorded.to_json()) == recorded.to_dict()
    assert strict_json(recorded.record.to_json()) == record


def test_dataframe_input_explicitly_has_no_raw_file_fingerprint() -> None:
    configuration = api.AnalysisConfig.from_dict(config_dict())
    recorded = api.run_analysis(sample_frame(), config=configuration)
    record = recorded.record.to_dict()
    assert record["input"] == {"kind": "dataframe", "name": None, "sha256": None,
                               "size_bytes": None, "format": None, "encoding": None, "delimiter": None}
    assert record["configuration"]["file"] is None
    verification = api.verify_run(recorded.record, config=configuration)
    assert verification.verified is False
    assert check(verification, "metadata")["status"] == "unavailable"
    with pytest.raises(ValueError):
        api.run_analysis(sample_frame(), config=configuration, format="csv")


def test_repeated_runs_have_stable_records_and_content_digests(tmp_path) -> None:
    metadata, configuration = write_inputs(tmp_path)
    first = api.run_analysis(metadata, config=configuration)
    second = api.run_analysis(metadata, config=configuration)
    assert first.report == second.report
    assert first.record.to_dict() == second.record.to_dict()
    assert first.to_json() == second.to_json()


def test_bom_and_line_ending_changes_affect_raw_fingerprint_but_not_scientific_payload(tmp_path) -> None:
    metadata, configuration = write_inputs(tmp_path)
    original_bytes = metadata.read_bytes()
    first = api.run_analysis(metadata, config=configuration)
    changed_bytes = b"\xef\xbb\xbf" + original_bytes.replace(b"\n", b"\r\n")
    metadata.write_bytes(changed_bytes)
    second = api.run_analysis(metadata, config=configuration)
    assert first.report == second.report
    assert first.record.to_dict()["report"] == second.record.to_dict()["report"]
    assert first.record.to_dict()["input"]["sha256"] != second.record.to_dict()["input"]["sha256"]
    assert second.record.to_dict()["input"]["sha256"] == hashlib.sha256(changed_bytes).hexdigest()
    verification = api.verify_run(first.record, metadata=metadata, config=configuration)
    assert not verification.verified
    assert check(verification, "metadata")["status"] == "mismatch"


def test_matching_explicit_candidates_verify_and_optional_report_is_not_required(tmp_path) -> None:
    metadata, configuration = write_inputs(tmp_path)
    recorded = api.run_analysis(metadata, config=configuration)
    verification = api.verify_run(recorded.record, metadata=metadata, config=configuration)
    assert isinstance(verification, api.VerificationReport)
    assert verification.verified is True
    for name in ("metadata", "configuration_file", "configuration"):
        assert check(verification, name)["status"] == "match"
        assert check(verification, name)["required"] is True
    assert check(verification, "report")["status"] == "not_checked"
    assert check(verification, "report")["required"] is False
    payload = verification.to_dict()
    assert payload["scope"] == "run_verification"
    assert payload["schema_version"] == "1"
    assert payload["environment"]["matches"] is True
    assert strict_json(verification.to_json()) == payload


def test_verification_never_infers_recorded_names_as_candidate_paths_or_executes_analysis(tmp_path, monkeypatch) -> None:
    metadata, configuration = write_inputs(tmp_path)
    recorded = api.run_analysis(metadata, config=configuration)
    record_path = tmp_path / "run.json"
    record_path.write_text(recorded.record.to_json(), encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    def forbidden(*args, **kwargs):
        raise AssertionError("Verification must not execute an analysis")

    monkeypatch.setattr(api, "diagnose", forbidden)
    monkeypatch.setattr(api, "validate", forbidden)
    missing = api.verify_run(record_path)
    assert not missing.verified
    assert check(missing, "metadata")["status"] == "not_checked"
    assert check(missing, "configuration")["status"] == "not_checked"
    matched = api.verify_run(record_path, metadata=metadata, config=configuration, report=recorded.report)
    assert matched.verified
    assert check(matched, "report")["status"] == "match"


@pytest.mark.parametrize("semantic", [False, True])
def test_config_file_and_resolved_config_are_verified_separately(tmp_path, semantic) -> None:
    metadata, configuration = write_inputs(tmp_path)
    recorded = api.run_analysis(metadata, config=configuration)
    changed = config_dict()
    if semantic:
        changed["unit_description"] = "A different declared analysis unit"
    configuration.write_bytes(json.dumps(changed, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    verification = api.verify_run(recorded.record, metadata=metadata, config=configuration)
    assert not verification.verified
    assert check(verification, "configuration_file")["status"] == "mismatch"
    assert check(verification, "configuration")["status"] == ("mismatch" if semantic else "match")


def test_config_object_can_verify_only_when_original_record_did_not_require_config_file_bytes(tmp_path) -> None:
    metadata, configuration = write_inputs(tmp_path)
    resolved = api.AnalysisConfig.from_json(configuration)
    with_file = api.run_analysis(metadata, config=configuration)
    without_file = api.run_analysis(metadata, config=resolved)
    unavailable = api.verify_run(with_file.record, metadata=metadata, config=resolved)
    assert not unavailable.verified
    assert check(unavailable, "configuration")["status"] == "match"
    assert check(unavailable, "configuration_file")["required"] is True
    assert check(unavailable, "configuration_file")["status"] in {"not_checked", "unavailable"}
    matched = api.verify_run(without_file.record, metadata=metadata, config=resolved)
    assert matched.verified
    assert check(matched, "configuration_file")["required"] is False


def test_report_verification_uses_canonical_payload_and_detects_tampering(tmp_path) -> None:
    metadata, configuration = write_inputs(tmp_path)
    recorded = api.run_analysis(metadata, config=configuration, command="validate")
    report_path = tmp_path / "scientific report.json"
    report_path.write_text(json.dumps(recorded.report, indent=4, sort_keys=True), encoding="utf-8")
    matched = api.verify_run(recorded.record, metadata=metadata, config=configuration, report=report_path)
    assert matched.verified
    assert check(matched, "report")["status"] == "match"
    assert check(matched, "report")["required"] is True
    tampered = recorded.report
    tampered["counts"]["n_records"] = 999
    mismatch = api.verify_run(recorded.record, metadata=metadata, config=configuration, report=tampered)
    assert not mismatch.verified
    assert check(mismatch, "report")["status"] == "mismatch"


def test_malformed_candidate_report_becomes_a_failed_check(tmp_path) -> None:
    metadata, configuration = write_inputs(tmp_path)
    recorded = api.run_analysis(metadata, config=configuration)
    bad = tmp_path / "malformed.json"
    bad.write_text("not JSON", encoding="utf-8")
    verification = api.verify_run(recorded.record, metadata=metadata, config=configuration, report=bad)
    assert not verification.verified
    assert check(verification, "report")["status"] == "error"
    assert strict_json(verification.to_json())["verified"] is False


def test_environment_versions_are_preserved_on_load_and_differences_are_informational(tmp_path) -> None:
    metadata, configuration = write_inputs(tmp_path)
    recorded = api.run_analysis(metadata, config=configuration)
    payload = recorded.record.to_dict()
    old_environment = {key: f"recorded-{value}" for key, value in payload["environment"].items()}
    payload["environment"] = old_environment.copy()
    loaded = api.RunRecord.from_dict(payload)
    path = tmp_path / "historical run.json"
    path.write_text(loaded.to_json(), encoding="utf-8")
    reloaded = api.RunRecord.from_json(path)
    assert reloaded.to_dict()["environment"] == old_environment
    verification = api.verify_run(reloaded, metadata=metadata, config=configuration)
    assert verification.verified
    environment = verification.to_dict()["environment"]
    assert environment["matches"] is False
    assert environment["recorded"] == old_environment
    assert environment["current"] != old_environment


def test_run_records_recorded_reports_and_verification_exports_are_detached(tmp_path) -> None:
    metadata, configuration = write_inputs(tmp_path)
    recorded = api.run_analysis(metadata, config=configuration, command="validate")
    source_payload = recorded.record.to_dict()
    record = api.RunRecord(source_payload)
    original_record = record.to_dict()
    source_payload["configuration"]["resolved"]["name"] = "changed caller dictionary"
    exported_record = record.to_dict()
    exported_record["environment"]["numpy"] = "changed export"
    assert record.to_dict() == original_record
    report_copy = recorded.report
    report_copy["counts"]["n_records"] = 999
    assert recorded.report["counts"]["n_records"] == 4
    verification = api.verify_run(record, metadata=metadata, config=configuration)
    verification_copy = verification.to_dict()
    verification_copy["checks"][0]["status"] = "mismatch"
    assert verification.verified
    assert verification.to_dict()["checks"][0]["status"] != "mismatch"
    with pytest.raises(api.RunRecordError):
        api.RecordedRun(report_copy, record)


@pytest.mark.parametrize("mutation", ["version", "bad_digest"])
def test_invalid_run_record_schema_is_rejected(tmp_path, mutation) -> None:
    metadata, configuration = write_inputs(tmp_path)
    payload = api.run_analysis(metadata, config=configuration).record.to_dict()
    if mutation == "version":
        payload["schema_version"] = "999"
    else:
        payload["input"]["sha256"] = "not-a-sha256"
    with pytest.raises(api.RunRecordError):
        api.RunRecord.from_dict(payload)
    with pytest.raises(api.RunRecordError):
        api.RunRecord(payload)


def test_duplicate_json_keys_and_malformed_reference_files_are_rejected(tmp_path) -> None:
    metadata, configuration = write_inputs(tmp_path)
    payload = api.run_analysis(metadata, config=configuration).record.to_dict()
    duplicated = json.dumps(payload).replace('"schema_version": "1"', '"schema_version": "1", "schema_version": "1"', 1)
    path = tmp_path / "duplicate record.json"
    path.write_text(duplicated, encoding="utf-8")
    with pytest.raises(api.RunRecordError):
        api.RunRecord.from_json(path)
    with pytest.raises(api.RunRecordError):
        api.verify_run(path, metadata=metadata, config=configuration)
    path.write_text("not JSON", encoding="utf-8")
    with pytest.raises(api.RunRecordError):
        api.verify_run(path, metadata=metadata, config=configuration)
    with pytest.raises(OSError):
        api.RunRecord.from_json(tmp_path / "absent record.json")


def test_invalid_metadata_is_recorded_as_validation_failure_not_success() -> None:
    command = "diagnose"
    frame = sample_frame()
    frame.at[0, "unit"] = None
    recorded = api.run_analysis(frame, config=api.AnalysisConfig.from_dict(config_dict()), command=command)
    assert recorded.exit_code == 2
    assert recorded.report["scope"] == "metadata_validation"
    assert recorded.report["valid"] is False
    record = recorded.record.to_dict()
    assert record["command"] == command
    assert record["exit_code"] == 2
    assert record["report"]["sha256"] == canonical_digest(recorded.report)


def test_computation_failures_are_recorded_with_their_error_payload(monkeypatch) -> None:
    def fail(frame, *, config):
        validation = api.validate(frame, config=config)
        try:
            raise ValueError("synthetic calculation failure")
        except ValueError as exc:
            raise api.DiagnosticComputationError("entropy", "synthetic calculation failure", validation=validation) from exc

    monkeypatch.setattr(api, "diagnose", fail)
    recorded = api.run_analysis(sample_frame(), config=api.AnalysisConfig.from_dict(config_dict()))
    assert recorded.exit_code == 3
    assert recorded.report["scope"] == "diagnostic_error"
    assert recorded.report["stage"] == "entropy"
    assert recorded.report["validation"]["valid"] is True
    assert recorded.record.to_dict()["report"]["sha256"] == canonical_digest(recorded.report)


def test_configuration_and_input_parse_errors_propagate_without_creating_a_record(tmp_path) -> None:
    metadata, configuration = write_inputs(tmp_path)
    configuration.write_text("not JSON", encoding="utf-8")
    with pytest.raises(api.ConfigurationError):
        api.run_analysis(metadata, config=configuration)
    config = api.AnalysisConfig.from_dict(config_dict())
    metadata.write_text("source,unit,group\nS1,too,few,fields\n", encoding="utf-8")
    with pytest.raises(ValueError):
        api.run_analysis(metadata, config=config)
    with pytest.raises(OSError):
        api.run_analysis(tmp_path / "absent.csv", config=config)


def test_excessively_nested_reference_json_raises_run_record_error(tmp_path) -> None:
    path = tmp_path / "deeply nested record.json"
    path.write_text("[" * 2500 + "]" * 2500, encoding="utf-8")
    with pytest.raises(api.RunRecordError):
        api.RunRecord.from_json(path)


@pytest.mark.parametrize("forgery", ["validation_exit", "diagnosis_exit", "package_version"])
def test_matching_report_hash_cannot_hide_inconsistent_exit_code_or_package_version(tmp_path, forgery) -> None:
    metadata, configuration = write_inputs(tmp_path)
    command = "validate" if forgery == "validation_exit" else "diagnose"
    recorded = api.run_analysis(metadata, config=configuration, command=command)
    payload = recorded.record.to_dict()
    if forgery == "validation_exit":
        payload["exit_code"] = 2
    elif forgery == "diagnosis_exit":
        payload["exit_code"] = 4
    else:
        payload["environment"]["package"] = "0.0.changed"
    forged_record = api.RunRecord.from_dict(payload)
    assert forged_record.to_dict()["report"]["sha256"] == canonical_digest(recorded.report)
    with pytest.raises(api.RunRecordError):
        api.RecordedRun(recorded.report, forged_record)
    verification = api.verify_run(forged_record, metadata=metadata, config=configuration, report=recorded.report)
    assert not verification.verified
    assert check(verification, "metadata")["status"] == "match"
    assert check(verification, "configuration")["status"] == "match"
    assert check(verification, "report")["status"] in {"error", "mismatch"}
