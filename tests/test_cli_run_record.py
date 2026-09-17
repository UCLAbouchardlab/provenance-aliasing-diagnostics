"""Companion run records and installed verification commands on synthetic files."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest

import provenance_aliasing.api as api
from provenance_aliasing.api.cli import main


def config_dict(**overrides) -> dict:
    value = {
        "schema_version": "0.1-draft", "name": "recorded-cli-atlas",
        "columns": {"source": ["source"], "unit": ["unit"], "group": ["group"]},
        "grain": {"name": "study", "description": "Synthetic originating study"},
        "unit_description": "One synthetic sample",
    }
    value.update(overrides)
    return value


def write_inputs(tmp_path: Path, *, frame=None, config=None):
    metadata = tmp_path / "metadata with spaces.csv"
    if frame is None:
        metadata.write_bytes(b"source,unit,group\nS1,001,A\nS1,002,B\nS2,003,A\nS2,004,B\n")
    else:
        frame.to_csv(metadata, index=False)
    configuration = tmp_path / "analysis config.json"
    configuration.write_text(json.dumps(config_dict() if config is None else config), encoding="utf-8")
    return metadata, configuration


def run_command(command: str, metadata: Path, configuration: Path, *options) -> int:
    return main([command, str(metadata), "--config", str(configuration), *map(str, options)])


def verify_command(record: Path, metadata: Path, configuration: Path, *options) -> int:
    return main(["verify", str(record), "--metadata", str(metadata), "--config", str(configuration), *map(str, options)])


def strict_json(text: str):
    def reject(value):
        raise AssertionError(f"Nonstandard JSON constant: {value}")
    return json.loads(text, parse_constant=reject)


def digest(payload: dict) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_bundle(tmp_path: Path):
    metadata, configuration = write_inputs(tmp_path)
    recorded = api.run_analysis(metadata, config=configuration)
    record = tmp_path / "run record.json"
    record.write_text(recorded.record.to_json(), encoding="utf-8")
    report = tmp_path / "scientific report.json"
    report.write_text(json.dumps(recorded.report), encoding="utf-8")
    return metadata, configuration, record, report


def test_optional_companion_preserves_scientific_stdout_and_records_its_digest(tmp_path, capsys) -> None:
    command = "validate"
    metadata, configuration = write_inputs(tmp_path)
    expected = api.run_analysis(metadata, config=configuration, command=command)
    record_path = tmp_path / "run record.json"
    assert run_command(command, metadata, configuration, "--run-record", record_path, "--quiet") == 0
    captured = capsys.readouterr()
    assert not captured.err
    payload = strict_json(captured.out)
    assert payload == expected.report
    record = api.RunRecord.from_json(record_path).to_dict()
    assert record == expected.record.to_dict()
    assert record["report"]["sha256"] == digest(payload)
    assert record["command"] == command
    assert record["exit_code"] == 0


def test_invalid_metadata_still_receives_companion_record_with_exit_two(tmp_path, capsys) -> None:
    command = "diagnose"
    frame = pd.DataFrame({"source": ["S1", "S2"], "unit": ["U1", ""], "group": ["A", "B"]})
    metadata, configuration = write_inputs(tmp_path, frame=frame)
    record_path = tmp_path / "invalid run.json"
    assert run_command(command, metadata, configuration, "--run-record", record_path, "--quiet") == 2
    captured = capsys.readouterr()
    assert captured.err
    payload = strict_json(captured.out)
    record = api.RunRecord.from_json(record_path).to_dict()
    assert payload["scope"] == "metadata_validation"
    assert payload["valid"] is False
    assert record["command"] == command
    assert record["exit_code"] == 2
    assert record["report"] == {"scope": "metadata_validation", "sha256": digest(payload)}


def test_partial_resource_limited_report_has_matching_companion_and_exit_four(tmp_path, capsys) -> None:
    n = 600
    frame = pd.DataFrame({"source": [f"S{i}" for i in range(n)], "unit": [f"U{i}" for i in range(n)],
                          "group": ["A", "B"] * (n // 2)})
    metadata, configuration = write_inputs(tmp_path, frame=frame,
                                           config=config_dict(execution={"max_working_memory_mb": 1}))
    record_path = tmp_path / "partial run.json"
    report_path = tmp_path / "partial report.json"
    assert run_command("diagnose", metadata, configuration, "--run-record", record_path, "--output", report_path) == 4
    captured = capsys.readouterr()
    assert not captured.out and captured.err
    report = strict_json(report_path.read_text(encoding="utf-8"))
    record = api.RunRecord.from_json(record_path).to_dict()
    assert record["exit_code"] == 4
    assert any(metric["status"] == "resource_limited" for metric in report["metrics"])
    assert record["report"]["sha256"] == digest(report)


def test_computation_failure_produces_error_report_and_matching_companion(tmp_path, capsys, monkeypatch) -> None:
    metadata, configuration = write_inputs(tmp_path)
    record_path = tmp_path / "failed run.json"

    def fail(frame, *, config):
        raise api.DiagnosticComputationError("entropy", "synthetic failure", validation=api.validate(frame, config=config))

    monkeypatch.setattr(api, "diagnose", fail)
    assert run_command("diagnose", metadata, configuration, "--run-record", record_path, "--quiet") == 3
    captured = capsys.readouterr()
    report = strict_json(captured.out)
    assert captured.err and "Traceback" not in captured.err
    assert report["scope"] == "diagnostic_error"
    assert report["stage"] == "entropy"
    record = api.RunRecord.from_json(record_path).to_dict()
    assert record["exit_code"] == 3
    assert record["report"]["sha256"] == digest(report)


def test_parse_failure_does_not_publish_either_report_or_companion(tmp_path, capsys) -> None:
    metadata, configuration = write_inputs(tmp_path)
    metadata.write_bytes(b"source,unit,group\nS1,too,many,fields\n")
    report = tmp_path / "report.json"
    record = tmp_path / "run.json"
    assert run_command("diagnose", metadata, configuration, "--output", report, "--run-record", record) == 2
    captured = capsys.readouterr()
    assert not captured.out and captured.err
    assert not report.exists()
    assert not record.exists()


@pytest.mark.parametrize("existing", ["report", "record"])
def test_both_destinations_are_preflighted_before_publishing_either_output(tmp_path, capsys, existing) -> None:
    metadata, configuration = write_inputs(tmp_path)
    report = tmp_path / "report.json"
    record = tmp_path / "run.json"
    retained = report if existing == "report" else record
    prospective = record if existing == "report" else report
    retained.write_bytes(b"existing output must be preserved\n")
    before = retained.read_bytes()
    assert run_command("diagnose", metadata, configuration, "--output", report, "--run-record", record) == 3
    captured = capsys.readouterr()
    assert not captured.out and captured.err
    assert retained.read_bytes() == before
    assert not prospective.exists()


@pytest.mark.parametrize("protected", ["metadata", "configuration"])
def test_companion_cannot_replace_inputs_even_with_overwrite(tmp_path, capsys, protected) -> None:
    metadata, configuration = write_inputs(tmp_path)
    source = metadata if protected == "metadata" else configuration
    originals = {metadata: metadata.read_bytes(), configuration: configuration.read_bytes()}
    report = tmp_path / "new report.json"
    record = source
    before = set(tmp_path.iterdir())
    assert run_command("diagnose", metadata, configuration, "--output", report, "--run-record", record, "--overwrite") == 3
    captured = capsys.readouterr()
    assert not captured.out and captured.err
    assert set(tmp_path.iterdir()) == before
    assert all(path.read_bytes() == contents for path, contents in originals.items())


def test_report_and_companion_cannot_alias_the_same_file_through_a_hardlink(tmp_path, capsys) -> None:
    metadata, configuration = write_inputs(tmp_path)
    report = tmp_path / "report.json"
    report.write_bytes(b"existing output")
    record = tmp_path / "same output alias.json"
    try:
        record.hardlink_to(report)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"Hardlinks are unavailable: {exc}")
    assert run_command("diagnose", metadata, configuration, "--output", report, "--run-record", record, "--overwrite") == 3
    captured = capsys.readouterr()
    assert not captured.out and captured.err
    assert report.read_bytes() == record.read_bytes() == b"existing output"


def test_verify_command_writes_matching_verification_json(tmp_path, capsys) -> None:
    metadata, configuration, record, report = write_bundle(tmp_path)
    expected = api.verify_run(record, metadata=metadata, config=configuration, report=report).to_dict()
    assert verify_command(record, metadata, configuration, "--report", report, "--quiet") == 0
    captured = capsys.readouterr()
    assert not captured.err
    assert strict_json(captured.out) == expected


def test_verify_metadata_mismatch_returns_two_and_keeps_machine_readable_checks(tmp_path, capsys) -> None:
    metadata, configuration, record, report = write_bundle(tmp_path)
    metadata.write_bytes(metadata.read_bytes().replace(b"001", b"999"))
    assert verify_command(record, metadata, configuration, "--report", report, "--quiet") == 2
    captured = capsys.readouterr()
    assert captured.err
    payload = strict_json(captured.out)
    assert payload["scope"] == "run_verification"
    assert payload["verified"] is False
    assert any(item["name"] == "metadata" and item["status"] == "mismatch" for item in payload["checks"])


def test_invalid_reference_record_is_an_input_error_without_fabricated_verification(tmp_path, capsys) -> None:
    metadata, configuration = write_inputs(tmp_path)
    record = tmp_path / "invalid record.json"
    record.write_text("not JSON", encoding="utf-8")
    output = tmp_path / "verification.json"
    assert verify_command(record, metadata, configuration, "--output", output) == 2
    captured = capsys.readouterr()
    assert not captured.out and captured.err
    assert not output.exists()


@pytest.mark.parametrize("protected", ["record", "report"])
def test_verification_output_cannot_overwrite_any_verification_input(tmp_path, capsys, protected) -> None:
    metadata, configuration, record, report = write_bundle(tmp_path)
    paths = {"metadata": metadata, "configuration": configuration, "record": record, "report": report}
    originals = {path: path.read_bytes() for path in paths.values()}
    assert verify_command(record, metadata, configuration, "--report", report,
                          "--output", paths[protected], "--overwrite") == 3
    captured = capsys.readouterr()
    assert not captured.out and captured.err
    assert all(path.read_bytes() == original for path, original in originals.items())


def test_installed_verify_entry_point_works_outside_checkout(tmp_path) -> None:
    metadata, configuration, record, report = write_bundle(tmp_path)
    prefix = [sys.executable, "-B", "-m", "provenance_aliasing.api"]
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(prefix + ["verify", str(record), "--metadata", str(metadata),
                                        "--config", str(configuration), "--report", str(report), "--quiet"],
                               cwd=tmp_path, env=env, text=True, capture_output=True, check=False)
    assert completed.returncode == 0, completed.stderr
    assert not completed.stderr
    assert strict_json(completed.stdout)["verified"] is True


def test_second_staging_failure_preserves_both_existing_outputs_and_cleans_temporaries(tmp_path, capsys, monkeypatch) -> None:
    import provenance_aliasing.api.cli as cli_module

    metadata, configuration = write_inputs(tmp_path)
    report = tmp_path / "report.json"
    record = tmp_path / "run.json"
    report.write_bytes(b"old scientific report")
    record.write_bytes(b"old companion record")
    original_paths = set(tmp_path.iterdir())
    original_fsync = cli_module.os.fsync
    calls = 0

    def fail_second_fsync(descriptor):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("synthetic second staging failure")
        return original_fsync(descriptor)

    monkeypatch.setattr(cli_module.os, "fsync", fail_second_fsync)
    assert run_command("validate", metadata, configuration, "--output", report, "--run-record", record, "--overwrite") == 3
    captured = capsys.readouterr()
    assert calls == 2
    assert not captured.out and captured.err
    assert report.read_bytes() == b"old scientific report"
    assert record.read_bytes() == b"old companion record"
    assert set(tmp_path.iterdir()) == original_paths


def test_companion_publication_failure_reports_scientific_output_already_written(tmp_path, capsys, monkeypatch) -> None:
    import provenance_aliasing.api.cli as cli_module

    metadata, configuration = write_inputs(tmp_path)
    report = tmp_path / "report.json"
    record = tmp_path / "run.json"
    report.write_bytes(b"old scientific report")
    record.write_bytes(b"old companion record")
    original_paths = set(tmp_path.iterdir())
    original_replace = cli_module.os.replace

    def fail_companion_replace(source, destination, *args, **kwargs):
        if Path(destination) == record:
            raise OSError("synthetic companion publication failure")
        return original_replace(source, destination, *args, **kwargs)

    monkeypatch.setattr(cli_module.os, "replace", fail_companion_replace)
    assert run_command("validate", metadata, configuration, "--output", report, "--run-record", record, "--overwrite") == 3
    captured = capsys.readouterr()
    assert not captured.out
    assert "scientific report was written" in captured.err.lower()
    assert "companion" in captured.err.lower() and "not published" in captured.err.lower()
    assert strict_json(report.read_text(encoding="utf-8"))["valid"] is True
    assert record.read_bytes() == b"old companion record"
    assert set(tmp_path.iterdir()) == original_paths


def test_verify_rejects_excessively_nested_reference_json_without_traceback(tmp_path, capsys) -> None:
    metadata, configuration = write_inputs(tmp_path)
    record = tmp_path / "deeply nested record.json"
    record.write_text("[" * 2500 + "]" * 2500, encoding="utf-8")
    assert verify_command(record, metadata, configuration, "--quiet") == 2
    captured = capsys.readouterr()
    assert not captured.out
    assert captured.err and "Traceback" not in captured.err
