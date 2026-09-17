"""Configuration contracts, isolation, and strict JSON behavior."""
from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import hashlib
import json
from pathlib import Path
import sys

import pytest

from provenance_aliasing.api.config import (
    AnalysisConfig,
    ColumnMapping,
    ConfigurationError,
    DiagnosticSelection,
    ExecutionConfig,
    ValidationPolicies,
    WeightingConfig,
    read_config_with_fingerprint,
)


FIXTURES = Path(__file__).resolve().parent / "fixtures" / "synthetic"


def minimal_config() -> dict:
    return {
        "schema_version": "0.1-draft",
        "name": "synthetic-design",
        "columns": {"source": ["study"], "unit": ["study", "sample"], "group": ["condition"]},
        "grain": {"name": "study", "description": "The contributing study"},
        "unit_description": "One specimen within a study",
    }


def test_required_contract_resolves_defaults() -> None:
    config = AnalysisConfig.from_dict(minimal_config())
    assert config.columns.unit == ("study", "sample")
    assert config.weighting.mode == "incidence"
    assert config.weighting.column is None
    assert dict(config.hierarchy) == {}
    assert (config.policies.missing_required, config.policies.duplicates, config.policies.missing_parent) == (
        "error", "error", "report",
    )
    assert config.policies.missing_tokens == ()
    assert config.diagnostics.include == ("design", "entropy", "ceiling", "structure", "guards")
    assert (config.diagnostics.bootstrap_replicates, config.diagnostics.seed) == (0, 0)
    assert config.execution.max_working_memory_mb == 512


@pytest.mark.parametrize("path", sorted(FIXTURES.glob("*.mapping.json")), ids=lambda path: path.stem)
def test_four_draft_fixture_mappings_are_accepted_without_reinterpretation(path: Path) -> None:
    expected = json.loads(path.read_text(encoding="utf-8"))
    config = AnalysisConfig.from_json(path)
    assert config.to_dict() == expected
    assert AnalysisConfig.from_dict(json.loads(config.to_json())).to_dict() == expected


def test_config_and_export_do_not_share_mutable_inputs() -> None:
    data = minimal_config()
    data.update({
        "hierarchy": {"donor": ["study", "donor"]},
        "policies": {"missing_tokens": ["NA"]},
        "diagnostics": {"include": ["entropy"]},
    })
    config = AnalysisConfig.from_dict(data)
    data["columns"]["unit"].append("run")
    data["hierarchy"]["donor"][0] = "site"
    data["policies"]["missing_tokens"].append("unknown")
    data["diagnostics"]["include"].append("ceiling")
    exported = config.to_dict()
    exported["columns"]["source"][0] = "laboratory"
    exported["hierarchy"]["donor"].clear()
    assert config.columns.unit == ("study", "sample")
    assert config.columns.source == ("study",)
    assert config.hierarchy["donor"] == ("study", "donor")
    assert config.policies.missing_tokens == ("NA",)
    assert config.diagnostics.include == ("entropy",)
    with pytest.raises(FrozenInstanceError):
        config.name = "changed"
    with pytest.raises(FrozenInstanceError):
        config.columns.unit = ("changed",)
    with pytest.raises(TypeError):
        config.hierarchy["donor"] = ("changed",)


def test_direct_construction_validates_and_freezes_nested_mappings() -> None:
    data = minimal_config()
    data["hierarchy"] = {"donor": ["study", "donor"]}
    config = AnalysisConfig(**data)
    data["columns"]["unit"].clear()
    data["hierarchy"]["donor"].clear()
    assert config.columns.unit == ("study", "sample")
    assert config.hierarchy["donor"] == ("study", "donor")
    with pytest.raises(ConfigurationError) as error:
        replace(config, name="  ")
    assert error.value.path == "name"
    with pytest.raises(ConfigurationError) as error:
        replace(config, columns={"source": [], "unit": ["sample"], "group": ["condition"]})
    assert error.value.path == "columns.source"


@pytest.mark.parametrize("construct,path", [
    (lambda: ColumnMapping(source="study", unit=["sample"], group=["condition"]), "columns.source"),
    (lambda: WeightingConfig(mode="column", column="mass"), "weighting.description"),
    (lambda: WeightingConfig(column="mass"), "weighting.column"),
    (lambda: ValidationPolicies(duplicates="drop"), "policies.duplicates"),
    (lambda: DiagnosticSelection(seed=True), "diagnostics.seed"),
    (lambda: ExecutionConfig(max_working_memory_mb=0), "execution.max_working_memory_mb"),
])
def test_direct_nested_construction_cannot_bypass_validation(construct, path: str) -> None:
    with pytest.raises(ConfigurationError) as error:
        construct()
    assert error.value.path == path


@pytest.mark.parametrize("field", ["schema_version", "name", "columns", "grain", "unit_description"])
def test_required_fields_have_actionable_paths(field: str) -> None:
    data = minimal_config()
    del data[field]
    with pytest.raises(ConfigurationError) as error:
        AnalysisConfig.from_dict(data)
    assert error.value.path == field


@pytest.mark.parametrize("block", [None, "columns", "grain", "weighting", "policies", "diagnostics", "execution"])
def test_unknown_keys_are_rejected_at_every_fixed_schema_level(block) -> None:
    data = minimal_config()
    target = data if block is None else data.setdefault(block, {})
    target["typo"] = True
    with pytest.raises(ConfigurationError) as error:
        AnalysisConfig.from_dict(data)
    assert error.value.path == ("typo" if block is None else f"{block}.typo")


@pytest.mark.parametrize("block,field,value", [
    (None, "schema_version", "1.0"),
    (None, "name", "\t"),
    (None, "unit_description", None),
    ("columns", "source", []),
    ("columns", "source", "study"),
    ("columns", "unit", ["sample", "sample"]),
    ("columns", "group", ["  "]),
    ("grain", "name", ""),
    ("grain", "description", 1),
    ("weighting", "mode", "equal_unit"),
    ("policies", "missing_required", "ignore"),
    ("policies", "duplicates", "warn"),
    ("policies", "missing_parent", "drop"),
    ("policies", "missing_tokens", ["NA", "NA"]),
    ("policies", "missing_tokens", [" "]),
    ("policies", "missing_tokens", [" NA"]),
    ("diagnostics", "include", ["entropy", "entropy"]),
    ("diagnostics", "include", ["invented"]),
    ("diagnostics", "bootstrap_replicates", True),
    ("diagnostics", "bootstrap_replicates", -1),
    ("diagnostics", "seed", 1.0),
    ("execution", "max_working_memory_mb", False),
    ("execution", "max_working_memory_mb", 0),
])
def test_invalid_contract_values_are_rejected(block, field: str, value) -> None:
    data = minimal_config()
    target = data if block is None else data.setdefault(block, {})
    target[field] = value
    with pytest.raises(ConfigurationError) as error:
        AnalysisConfig.from_dict(data)
    expected = field if block is None else f"{block}.{field}"
    assert error.value.path == expected or error.value.path.startswith(expected + "[")


@pytest.mark.parametrize("weighting,path", [
    ({"mode": "incidence", "column": None}, "weighting.column"),
    ({"mode": "column", "column": "mass"}, "weighting.description"),
    ({"mode": "column", "description": "Row mass"}, "weighting.column"),
    ({"mode": "column", "column": " ", "description": "Row mass"}, "weighting.column"),
])
def test_weighting_requires_a_coherent_declaration(weighting, path: str) -> None:
    data = minimal_config()
    data["weighting"] = weighting
    with pytest.raises(ConfigurationError) as error:
        AnalysisConfig.from_dict(data)
    assert error.value.path == path


def test_column_names_are_preserved_exactly_and_missing_tokens_are_case_sensitive() -> None:
    data = minimal_config()
    data["columns"]["source"] = [" study "]
    data["policies"] = {"missing_tokens": ["NA", "na"]}
    data["weighting"] = {"mode": "column", "column": " mass ", "description": "Declared row mass"}
    config = AnalysisConfig.from_dict(data)
    assert config.columns.source == (" study ",)
    assert config.weighting.column == " mass "
    assert config.policies.missing_tokens == ("NA", "na")


@pytest.mark.parametrize("hierarchy", [{" donor": ["id"]}, {" ": ["id"]}, {"donor": []}, {"donor": ["id", "id"]}, {1: ["id"]}])
def test_parent_mapping_names_and_keys_are_validated(hierarchy) -> None:
    data = minimal_config()
    data["hierarchy"] = hierarchy
    with pytest.raises(ConfigurationError) as error:
        AnalysisConfig.from_dict(data)
    assert error.value.path.startswith("hierarchy")


def test_json_round_trip_handles_bom_and_unicode(tmp_path: Path) -> None:
    data = minimal_config()
    data["name"] = "Étude α"
    path = tmp_path / "config.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8-sig")
    config = AnalysisConfig.from_json(path)
    assert config.name == "Étude α"
    output = config.to_json()
    assert "Étude α" in output
    assert json.loads(output) == config.to_dict()


@pytest.mark.parametrize("text,path", [
    ('{"name":"one","name":"two"}', "name"),
    ('{"policies":{"duplicates":"error","duplicates":"keep_records"}}', "policies.duplicates"),
    ('{"hierarchy":{"donor":["id"],"donor":["other"]}}', "hierarchy.donor"),
    ('{"diagnostics":{"seed":NaN}}', "diagnostics.seed"),
    ('{"diagnostics":{"seed":Infinity}}', "diagnostics.seed"),
    ('{"diagnostics":{"seed":-Infinity}}', "diagnostics.seed"),
    ('{"diagnostics":{"seed":1e999}}', "diagnostics.seed"),
    ('[]', "$"),
    ('{"broken":', "$"),
])
def test_invalid_json_is_rejected_without_losing_field_paths(tmp_path: Path, text: str, path: str) -> None:
    file = tmp_path / "invalid.json"
    file.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigurationError) as error:
        AnalysisConfig.from_json(file)
    assert error.value.path == path


def test_nonfinite_python_input_is_also_rejected() -> None:
    data = minimal_config()
    data["diagnostics"] = {"seed": float("inf")}
    with pytest.raises(ConfigurationError) as error:
        AnalysisConfig.from_dict(data)
    assert error.value.path == "diagnostics.seed"


def test_missing_file_remains_an_io_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        AnalysisConfig.from_json(tmp_path / "absent.json")


@pytest.mark.parametrize("bom", [b"", b"\xef\xbb\xbf"])
@pytest.mark.parametrize("newline", ["\n", "\r\n", "\r"])
def test_config_fingerprint_records_the_exact_parsed_bytes(tmp_path, bom, newline) -> None:
    data = minimal_config()
    data["name"] = "Étude α"
    content = bom + (json.dumps(data, ensure_ascii=False, indent=2).replace("\n", newline) + newline).encode("utf-8")
    path = tmp_path / "Original.mapping.json"
    path.write_bytes(content)
    config, fingerprint = read_config_with_fingerprint(path)
    assert config.to_dict() == AnalysisConfig.from_dict(data).to_dict()
    assert config.to_dict() == AnalysisConfig.from_json(path).to_dict()
    assert fingerprint == {
        "name": "Original.mapping.json", "sha256": hashlib.sha256(content).hexdigest(), "size_bytes": len(content),
    }
    assert str(tmp_path) not in repr(fingerprint)


def test_config_fingerprint_uses_the_single_buffer_that_was_parsed(tmp_path, monkeypatch) -> None:
    path = tmp_path / "single-read.json"
    data = minimal_config()
    data["name"] = "first exact read"
    content = json.dumps(data, ensure_ascii=False).encode("utf-8-sig")
    reads: list[Path] = []
    original_read = Path.read_bytes

    def single_read(self):
        if self != path:
            return original_read(self)
        reads.append(self)
        assert len(reads) == 1, "The configuration must not be reread for hashing."
        return content

    monkeypatch.setattr(Path, "read_bytes", single_read)
    config, fingerprint = read_config_with_fingerprint(path)
    assert config.name == "first exact read"
    assert reads == [path]
    assert fingerprint["sha256"] == hashlib.sha256(content).hexdigest()
    assert fingerprint["size_bytes"] == len(content)


def test_config_byte_differences_remain_visible_when_resolved_configuration_matches(tmp_path) -> None:
    data = minimal_config()
    compact = tmp_path / "compact.json"
    formatted = tmp_path / "formatted.json"
    compact.write_bytes(json.dumps(data, separators=(",", ":")).encode("utf-8"))
    formatted.write_bytes((json.dumps(data, indent=2).replace("\n", "\r\n") + "\r\n").encode("utf-8-sig"))
    left, left_fingerprint = read_config_with_fingerprint(compact)
    right, right_fingerprint = read_config_with_fingerprint(formatted)
    assert left.to_dict() == right.to_dict()
    assert left_fingerprint["sha256"] != right_fingerprint["sha256"]
    assert left_fingerprint["size_bytes"] != right_fingerprint["size_bytes"]


@pytest.mark.parametrize("content", [
    b'{"name":"one","name":"two"}',
    b'{"policies":{"duplicates":"error","duplicates":"keep_records"}}',
    b'{"diagnostics":{"seed":NaN}}', b'{"diagnostics":{"seed":Infinity}}',
    b'{"diagnostics":{"seed":1e999}}', b'[]', b'{"broken":', b'{"name":"\xff"}',
])
def test_fingerprinted_config_reader_preserves_configuration_errors_and_paths(tmp_path, content) -> None:
    path = tmp_path / "invalid.json"
    path.write_bytes(content)
    with pytest.raises(ConfigurationError) as original:
        AnalysisConfig.from_json(path)
    with pytest.raises(ConfigurationError) as captured:
        read_config_with_fingerprint(path)
    assert str(captured.value) == str(original.value)
    assert captured.value.path == original.value.path


def test_config_cr_only_json_errors_keep_universal_newline_line_numbers(tmp_path) -> None:
    path = tmp_path / "invalid.json"
    path.write_bytes(b'{\r"name":"example",\r"broken":\r}')
    with pytest.raises(ConfigurationError, match="invalid JSON at line 4, column 1"):
        read_config_with_fingerprint(path)


def test_from_json_still_constructs_the_requested_subclass(tmp_path) -> None:
    class SpecializedConfig(AnalysisConfig):
        pass

    path = tmp_path / "specialized.json"
    path.write_bytes(json.dumps(minimal_config()).encode("utf-8"))
    result = SpecializedConfig.from_json(path)
    assert type(result) is SpecializedConfig
    assert result.to_dict() == AnalysisConfig.from_dict(minimal_config()).to_dict()


def test_fingerprinted_config_reader_preserves_filesystem_errors(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        read_config_with_fingerprint(tmp_path / "absent.json")


def test_parser_deep_json_raises_configuration_error_in_both_file_entry_points(tmp_path) -> None:
    depth = sys.getrecursionlimit() + 20
    text = '{"name":' + "[" * depth + "0" + "]" * depth + "}"
    path = tmp_path / "parser-deep.json"
    path.write_bytes(text.encode("utf-8"))
    with pytest.raises(RecursionError):
        json.loads(text)
    for read in (AnalysisConfig.from_json, read_config_with_fingerprint):
        with pytest.raises(ConfigurationError, match="nesting is too deep") as caught:
            read(path)
        assert caught.value.path == "$"


def test_json_accepted_by_parser_but_too_deep_for_normalization_is_a_configuration_error(tmp_path) -> None:
    depth = sys.getrecursionlimit() * 3 // 4
    text = '{"name":' + "[" * depth + "0" + "]" * depth + "}"
    parsed = json.loads(text)
    path = tmp_path / "normalization-deep.json"
    path.write_bytes(text.encode("utf-8"))
    for read in (AnalysisConfig.from_json, read_config_with_fingerprint):
        with pytest.raises(ConfigurationError, match="nesting is too deep") as caught:
            read(path)
        assert caught.value.path == "$"
    with pytest.raises(ConfigurationError, match="nesting is too deep"):
        AnalysisConfig.from_dict(parsed)
