"""Installed command-line workflows using synthetic metadata and temporary reports."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import sysconfig
from importlib.metadata import version

import pandas as pd
import pytest

import provenance_aliasing.api as api
from provenance_aliasing.api.cli import main


EXAMPLES = Path(__file__).resolve().parent / "fixtures" / "synthetic"
EXPECTED = json.loads((EXAMPLES / "expected.json").read_text(encoding="utf-8"))


def config_dict(**overrides) -> dict:
    config = {
        "schema_version": "0.1-draft", "name": "synthetic-cli",
        "columns": {"source": ["source"], "unit": ["unit"], "group": ["group"]},
        "grain": {"name": "study", "description": "Declared synthetic study"},
        "unit_description": "One synthetic sample",
    }
    config.update(overrides)
    return config


def sample_frame() -> pd.DataFrame:
    return pd.DataFrame({"source": ["S1", "S1", "S2", "S2"],
                         "unit": ["001", "002", "003", "004"],
                         "group": ["A", "B", "A", "B"]})


def write_inputs(tmp_path: Path, *, frame=None, config=None, suffix=".csv", separator=","):
    metadata_path = tmp_path / f"metadata with spaces{suffix}"
    config_path = tmp_path / "analysis mapping.json"
    (sample_frame() if frame is None else frame).to_csv(metadata_path, sep=separator, index=False)
    config_path.write_text(json.dumps(config_dict() if config is None else config), encoding="utf-8")
    return metadata_path, config_path


def invoke(command: str, metadata_path: Path, config_path: Path, *options: str) -> int:
    return main([command, str(metadata_path), "--config", str(config_path), *options])


def strict_json(text: str):
    def reject_constant(value):
        raise AssertionError(f"Nonstandard JSON number: {value}")

    return json.loads(text, parse_constant=reject_constant)


def subprocess_cli(entry_point: str, argv: list[str], cwd: Path):
    if entry_point == "module":
        prefix = [sys.executable, "-B", "-m", "provenance_aliasing.api"]
    else:
        suffix = ".exe" if os.name == "nt" else ""
        prefix = [str(Path(sysconfig.get_path("scripts")) / f"provenance-aliasing{suffix}")]
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(prefix + argv, cwd=cwd, env=env, capture_output=True, text=True, check=False)


@pytest.mark.parametrize("example", EXPECTED["examples"], ids=lambda example: example["id"])
def test_cli_diagnostics_match_python_api_for_all_four_synthetic_omics_examples(example, capsys) -> None:
    metadata_path = EXAMPLES / example["data_file"]
    config_path = EXAMPLES / example["config_file"]
    frame = pd.read_csv(metadata_path, dtype=str, keep_default_na=False)
    config = api.AnalysisConfig.from_json(config_path)
    expected = api.diagnose(frame, config=config).to_dict()
    code = invoke("diagnose", metadata_path, config_path)
    captured = capsys.readouterr()
    assert code == 0
    assert strict_json(captured.out) == expected
    assert captured.err
    assert "Traceback" not in captured.err


@pytest.mark.parametrize("suffix,options", [(".tsv", ()), (".TXT", ("--format", "tsv"))])
def test_tsv_and_explicit_format_preserve_leading_zeros_and_literal_na(tmp_path, capsys, suffix, options) -> None:
    frame = pd.DataFrame({"source": ["NA", "NA", "S2", "S2"],
                          "unit": ["001", "002", "01", "1"], "group": ["NA", "B", "NA", "B"]})
    metadata_path, config_path = write_inputs(tmp_path, frame=frame, suffix=suffix, separator="\t")
    expected = api.diagnose(frame, config=api.AnalysisConfig.from_json(config_path)).to_dict()
    assert invoke("diagnose", metadata_path, config_path, *options, "--quiet") == 0
    captured = capsys.readouterr()
    assert not captured.err
    payload = strict_json(captured.out)
    assert payload == expected
    observations = payload["tables"]["input.observations"]
    unit_column = observations["columns"].index("unit")
    assert [row[unit_column] for row in observations["data"]] == ["001", "002", "01", "1"]


def test_csv_format_override_and_utf8_bom_are_supported(tmp_path, capsys) -> None:
    metadata_path, config_path = write_inputs(tmp_path, suffix=".data")
    content = metadata_path.read_text(encoding="utf-8")
    metadata_path.write_text(content, encoding="utf-8-sig")
    assert invoke("validate", metadata_path, config_path, "--format", "csv", "--output", "-") == 0
    assert strict_json(capsys.readouterr().out)["valid"] is True


@pytest.mark.parametrize("command", ["validate", "diagnose"])
def test_invalid_metadata_writes_validation_report_with_exit_two_even_when_quiet(tmp_path, capsys, command) -> None:
    frame = sample_frame()
    frame.at[1, "unit"] = ""
    metadata_path, config_path = write_inputs(tmp_path, frame=frame)
    expected = api.validate(frame, config=api.AnalysisConfig.from_json(config_path)).to_dict()
    assert invoke(command, metadata_path, config_path, "--quiet") == 2
    captured = capsys.readouterr()
    assert captured.err
    payload = strict_json(captured.out)
    assert payload == expected
    assert payload["scope"] == "metadata_validation"
    assert payload["valid"] is False
    assert "Traceback" not in captured.err


def test_missing_configured_columns_produce_invalid_validation_json(tmp_path, capsys) -> None:
    metadata_path, config_path = write_inputs(tmp_path, frame=sample_frame().drop(columns="group"))
    assert invoke("validate", metadata_path, config_path) == 2
    captured = capsys.readouterr()
    payload = strict_json(captured.out)
    assert payload["scope"] == "metadata_validation"
    assert any(issue["code"] == "missing_columns" for issue in payload["issues"])


@pytest.mark.parametrize("contents", [
    'source,unit,group\nS1,"unterminated,A\n',
    "source,unit,group\nS1,U1,A,extra\n",
    "source,unit,group\nS1,U1\n",
    "source,unit,group,unit\nS1,U1,A,U2\n",
])
def test_malformed_csv_and_duplicate_headers_are_parse_errors_without_fabricated_reports(tmp_path, capsys, contents) -> None:
    metadata_path, config_path = write_inputs(tmp_path)
    metadata_path.write_text(contents, encoding="utf-8")
    output = tmp_path / "report.json"
    assert invoke("validate", metadata_path, config_path, "--output", str(output)) == 2
    captured = capsys.readouterr()
    assert not captured.out
    assert captured.err
    assert "Traceback" not in captured.err
    assert not output.exists()


@pytest.mark.parametrize("problem", ["bad_json", "invalid_config", "missing_metadata", "unknown_format"])
def test_configuration_and_input_errors_return_two_without_writing_reports(tmp_path, capsys, problem) -> None:
    metadata_path, config_path = write_inputs(tmp_path)
    if problem == "bad_json":
        config_path.write_text('{"unfinished":', encoding="utf-8")
    elif problem == "invalid_config":
        config_path.write_text(json.dumps(config_dict(columns={"source": ["source"]})), encoding="utf-8")
    elif problem == "missing_metadata":
        metadata_path = tmp_path / "absent.csv"
    else:
        unusual = tmp_path / "metadata.unknown"
        unusual.write_bytes(metadata_path.read_bytes())
        metadata_path = unusual
    output = tmp_path / "report.json"
    assert invoke("diagnose", metadata_path, config_path, "--output", str(output)) == 2
    captured = capsys.readouterr()
    assert not captured.out
    assert captured.err
    assert "Traceback" not in captured.err
    assert not output.exists()


@pytest.mark.parametrize("command", ["validate", "diagnose"])
def test_warnings_and_unavailable_contrasts_are_successful_and_quiet_only_hides_summary(tmp_path, capsys, command) -> None:
    metadata_path, config_path = write_inputs(tmp_path, frame=sample_frame().assign(group="A"))
    assert invoke(command, metadata_path, config_path, "--quiet") == 0
    captured = capsys.readouterr()
    assert not captured.err
    payload = strict_json(captured.out)
    if command == "validate":
        assert payload["valid"] is True
        assert any(issue["severity"] == "warning" for issue in payload["issues"])
    else:
        assert payload["validation"]["valid"] is True
        assert any(row["status"] in {"undefined", "not_applicable"} for row in payload["metrics"])


def test_output_path_with_spaces_receives_json_without_stdout_noise(tmp_path, capsys) -> None:
    metadata_path, config_path = write_inputs(tmp_path)
    directory = tmp_path / "reports with spaces"
    directory.mkdir()
    output = directory / "validation report.json"
    assert invoke("validate", metadata_path, config_path, "--output", str(output)) == 0
    captured = capsys.readouterr()
    assert not captured.out
    assert captured.err
    payload = strict_json(output.read_text(encoding="utf-8"))
    assert payload == api.validate(sample_frame(), config=api.AnalysisConfig.from_json(config_path)).to_dict()


def test_existing_output_requires_explicit_overwrite(tmp_path, capsys) -> None:
    metadata_path, config_path = write_inputs(tmp_path)
    output = tmp_path / "report.json"
    original = b"existing report must remain untouched\n"
    output.write_bytes(original)
    assert invoke("validate", metadata_path, config_path, "--output", str(output)) == 3
    captured = capsys.readouterr()
    assert not captured.out and captured.err
    assert output.read_bytes() == original
    assert invoke("validate", metadata_path, config_path, "--output", str(output), "--overwrite") == 0
    assert strict_json(output.read_text(encoding="utf-8"))["valid"] is True


@pytest.mark.parametrize("protected", ["metadata", "config"])
@pytest.mark.parametrize("alias", ["direct", "hardlink", "symlink"])
def test_output_can_never_replace_metadata_or_config_even_through_aliases(tmp_path, capsys, protected, alias) -> None:
    metadata_path, config_path = write_inputs(tmp_path)
    source = metadata_path if protected == "metadata" else config_path
    original = source.read_bytes()
    output = source
    if alias != "direct":
        output = tmp_path / f"{alias} report.json"
        try:
            if alias == "hardlink":
                output.hardlink_to(source)
            else:
                output.symlink_to(source)
        except (OSError, NotImplementedError) as exc:
            pytest.skip(f"Filesystem does not permit the requested {alias}: {exc}")
    assert invoke("diagnose", metadata_path, config_path, "--output", str(output), "--overwrite") == 3
    captured = capsys.readouterr()
    assert not captured.out and captured.err
    assert source.read_bytes() == original
    assert output.read_bytes() == original


def test_report_writer_does_not_create_missing_parent_directories(tmp_path, capsys) -> None:
    metadata_path, config_path = write_inputs(tmp_path)
    parent = tmp_path / "nonexistent reports"
    output = parent / "report.json"
    assert invoke("validate", metadata_path, config_path, "--output", str(output)) == 3
    captured = capsys.readouterr()
    assert not captured.out and captured.err
    assert not parent.exists()


def test_partial_resource_limited_diagnosis_is_written_with_exit_four(tmp_path, capsys) -> None:
    n = 600
    frame = pd.DataFrame({"source": [f"S{i}" for i in range(n)], "unit": [f"U{i}" for i in range(n)],
                          "group": ["A", "B"] * (n // 2)})
    metadata_path, config_path = write_inputs(tmp_path, frame=frame,
                                             config=config_dict(execution={"max_working_memory_mb": 1}))
    output = tmp_path / "partial report.json"
    assert invoke("diagnose", metadata_path, config_path, "--output", str(output)) == 4
    captured = capsys.readouterr()
    assert not captured.out and captured.err
    payload = strict_json(output.read_text(encoding="utf-8"))
    assert payload["scope"] == "metadata_diagnostics"
    assert payload["validation"]["valid"] is True
    assert any(row["status"] == "resource_limited" for row in payload["metrics"])
    assert any(row["metric"] == "entropy.ratio" and row["status"] == "ok" for row in payload["metrics"])


def test_cli_and_api_have_identical_seeded_bootstrap_and_resolved_configuration(tmp_path, capsys) -> None:
    frame = pd.DataFrame({"source": ["S1", "S1", "S2", "S2", "S3", "S3"],
                          "unit": [f"U{i}" for i in range(6)], "group": ["A", "B", "A", "A", "B", "B"]})
    configuration = config_dict(diagnostics={"include": ["entropy"], "bootstrap_replicates": 16, "seed": 29})
    metadata_path, config_path = write_inputs(tmp_path, frame=frame, config=configuration)
    expected = api.diagnose(frame, config=api.AnalysisConfig.from_dict(configuration)).to_dict()
    assert invoke("diagnose", metadata_path, config_path, "--quiet") == 0
    captured = capsys.readouterr()
    assert not captured.err
    assert strict_json(captured.out) == expected


def test_computation_failure_returns_typed_json_envelope_and_exit_three_without_traceback(tmp_path, capsys, monkeypatch) -> None:
    metadata_path, config_path = write_inputs(tmp_path)

    def fail(metadata, *, config):
        validation = api.validate(metadata, config=config)
        try:
            raise ValueError("synthetic calculation failure")
        except ValueError as exc:
            raise api.DiagnosticComputationError("entropy", "synthetic calculation failure", validation=validation) from exc

    monkeypatch.setattr(api, "diagnose", fail)
    assert invoke("diagnose", metadata_path, config_path, "--quiet") == 3
    captured = capsys.readouterr()
    assert captured.err
    assert "Traceback" not in captured.err
    payload = strict_json(captured.out)
    assert payload["scope"] == "diagnostic_error"
    assert payload["schema_version"] == "0.1-draft"
    assert payload["stage"] == "entropy"
    assert payload["error"]["type"] == "DiagnosticComputationError"
    assert "synthetic calculation failure" in payload["error"]["message"]
    assert payload["cause"] == {"type": "ValueError", "message": "synthetic calculation failure"}
    assert payload["validation"]["valid"] is True
    assert payload["config"] == api.AnalysisConfig.from_json(config_path).to_dict()
    assert payload["package"]["name"] == "provenance-aliasing-diagnostics"


@pytest.mark.parametrize("entry_point", ["module", "console"])
def test_installed_entry_points_emit_parseable_stdout_json_outside_checkout(tmp_path, entry_point) -> None:
    metadata_path, config_path = write_inputs(tmp_path)
    completed = subprocess_cli(entry_point, ["validate", str(metadata_path), "--config", str(config_path)], tmp_path)
    assert completed.returncode == 0, completed.stderr
    assert strict_json(completed.stdout)["valid"] is True
    assert completed.stderr


@pytest.mark.parametrize("entry_point", ["module", "console"])
def test_installed_entry_points_have_help_version_and_friendly_no_argument_behavior(tmp_path, entry_point) -> None:
    for argv in ([], ["--help"]):
        completed = subprocess_cli(entry_point, argv, tmp_path)
        assert completed.returncode == 0
        assert "validate" in completed.stdout
        assert "diagnose" in completed.stdout
        assert not completed.stderr
    completed = subprocess_cli(entry_point, ["--version"], tmp_path)
    assert completed.returncode == 0
    assert completed.stdout.strip() == f"provenance-aliasing {version('provenance-aliasing-diagnostics')}"
    assert not completed.stderr


@pytest.mark.parametrize("arguments", [["not-a-command"], ["validate"],
                                       ["validate", "input.csv"],
                                       ["validate", "input.csv", "--config", "config.json", "--format", "xlsx"]])
def test_argument_parser_errors_remain_system_exit_two_without_stdout_json(capsys, arguments) -> None:
    with pytest.raises(SystemExit) as caught:
        main(arguments)
    assert caught.value.code == 2
    captured = capsys.readouterr()
    assert not captured.out
    assert captured.err


def test_failed_atomic_replacement_preserves_existing_report_and_cleans_temporary_file(tmp_path, capsys, monkeypatch) -> None:
    import provenance_aliasing.api.cli as cli_module

    metadata_path, config_path = write_inputs(tmp_path)
    output = tmp_path / "report.json"
    original = b"existing report\n"
    output.write_bytes(original)
    original_paths = set(tmp_path.iterdir())

    def fail_replace(*args, **kwargs):
        raise PermissionError("synthetic publication failure")

    monkeypatch.setattr(cli_module.os, "replace", fail_replace)
    assert invoke("validate", metadata_path, config_path, "--output", str(output), "--overwrite") == 3
    captured = capsys.readouterr()
    assert not captured.out
    assert captured.err and "Traceback" not in captured.err
    assert output.read_bytes() == original
    assert set(tmp_path.iterdir()) == original_paths


@pytest.mark.parametrize("overwrite", [False, True])
def test_output_directory_is_an_io_error_and_remains_a_directory(tmp_path, capsys, overwrite) -> None:
    metadata_path, config_path = write_inputs(tmp_path)
    directory = tmp_path / "reports"
    directory.mkdir()
    marker = directory / "existing.txt"
    marker.write_text("unchanged", encoding="utf-8")
    options = ["--output", str(directory)] + (["--overwrite"] if overwrite else [])
    assert invoke("validate", metadata_path, config_path, *options) == 3
    captured = capsys.readouterr()
    assert not captured.out and captured.err
    assert directory.is_dir()
    assert marker.read_text(encoding="utf-8") == "unchanged"


def test_dangling_output_symlink_counts_as_existing_and_is_not_followed_implicitly(tmp_path, capsys) -> None:
    metadata_path, config_path = write_inputs(tmp_path)
    missing_target = tmp_path / "missing target.json"
    output = tmp_path / "linked report.json"
    try:
        output.symlink_to(missing_target)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"Filesystem does not permit symlinks: {exc}")
    assert output.is_symlink() and not output.exists()
    assert invoke("validate", metadata_path, config_path, "--output", str(output)) == 3
    captured = capsys.readouterr()
    assert not captured.out and captured.err
    assert output.is_symlink()
    assert not missing_target.exists()


def test_redirected_stdout_json_preserves_unicode_identifiers(tmp_path) -> None:
    frame = pd.DataFrame({"source": ["研究α", "研究α", "研究β", "研究β"],
                          "unit": ["样本001", "样本002", "样本003", "样本004"],
                          "group": ["对照", "治疗", "对照", "治疗"]})
    metadata_path, config_path = write_inputs(tmp_path, frame=frame,
                                             config=config_dict(name="示例图谱"))
    completed = subprocess_cli("module", ["diagnose", str(metadata_path), "--config", str(config_path), "--quiet"], tmp_path)
    assert completed.returncode == 0, completed.stderr
    assert not completed.stderr
    payload = strict_json(completed.stdout)
    assert payload["config"]["name"] == "示例图谱"
    table = payload["tables"]["input.observations"]
    columns = table["columns"]
    assert table["data"][0][columns.index("source")] == "研究α"
    assert table["data"][0][columns.index("unit")] == "样本001"
    assert table["data"][0][columns.index("group")] == "对照"


def test_result_serialization_failure_preserves_completed_validation_in_error_envelope(tmp_path, capsys, monkeypatch) -> None:
    metadata_path, config_path = write_inputs(tmp_path)
    expected_validation = api.validate(sample_frame(), config=api.AnalysisConfig.from_json(config_path))

    class UnserializableResult:
        validation = expected_validation

        def to_dict(self):
            raise TypeError("synthetic result serialization failure")

    def diagnose_with_unserializable_result(metadata, *, config):
        return UnserializableResult()

    monkeypatch.setattr(api, "diagnose", diagnose_with_unserializable_result)
    assert invoke("diagnose", metadata_path, config_path, "--quiet") == 3
    captured = capsys.readouterr()
    assert captured.err and "Traceback" not in captured.err
    payload = strict_json(captured.out)
    assert payload["scope"] == "diagnostic_error"
    assert payload["stage"] == "diagnose"
    assert payload["error"] == {"type": "TypeError", "message": "synthetic result serialization failure"}
    assert payload["validation"] == expected_validation.to_dict()
    assert payload["validation"]["valid"] is True


def test_closed_stdout_pipe_exits_three_without_a_second_shutdown_flush_error(tmp_path) -> None:
    metadata_path, config_path = write_inputs(tmp_path)
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.pop("PYTHONUNBUFFERED", None)
    process = subprocess.Popen(
        [sys.executable, "-B", "-m", "provenance_aliasing.api", "validate", str(metadata_path),
         "--config", str(config_path)],
        cwd=tmp_path, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        assert process.stdout is not None
        process.stdout.close()
        # communicate must read only stderr after the consumer closes stdout.
        process.stdout = None
        _, stderr = process.communicate(timeout=30)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10)
    assert process.returncode == 3, stderr
    assert stderr
    assert "Exception ignored" not in stderr
    assert "Traceback" not in stderr
