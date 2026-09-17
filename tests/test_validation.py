"""Behavioral contracts for explicit, metadata-only API validation."""
from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from datetime import datetime
from decimal import Decimal
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import pytest

from provenance_aliasing.api import (
    AnalysisConfig,
    MetadataValidationError,
    ValidationReport,
    validate,
)


EXAMPLES = Path(__file__).resolve().parent / "fixtures" / "synthetic"
EXPECTED = json.loads((EXAMPLES / "expected.json").read_text(encoding="utf-8"))
COUNT_FIELDS = {
    "n_records", "n_key_complete", "n_retained", "n_excluded", "n_error_records",
    "n_duplicate_keys", "n_duplicate_rows", "n_sources", "n_units", "n_groups",
    "total_weight",
}


def analysis_config(**overrides) -> AnalysisConfig:
    raw = {
        "schema_version": "0.1-draft",
        "name": "synthetic-validation",
        "columns": {"source": ["source"], "unit": ["unit"], "group": ["group"]},
        "grain": {"name": "study", "description": "Declared synthetic source study"},
        "unit_description": "An explicitly identified synthetic sample",
    }
    raw.update(overrides)
    return AnalysisConfig.from_dict(raw)


def simple_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {"source": ["S1", "S1", "S2", "S2"],
         "unit": ["U1", "U2", "U3", "U4"],
         "group": ["A", "B", "A", "B"]}
    )


def weighted_config(**overrides) -> AnalysisConfig:
    return analysis_config(
        weighting={"mode": "column", "column": "mass", "description": "Toy row mass"},
        **overrides,
    )


def errors(report: ValidationReport):
    return [issue for issue in report.issues if issue.severity == "error"]


def positions_for(report: ValidationReport, *, severity: str, column: str) -> set[int]:
    return {
        position
        for issue in report.issues
        if issue.severity == severity and column in issue.columns
        for position in issue.row_positions
    }


def assert_json_report(report: ValidationReport) -> None:
    # NaN and infinity must never escape as nonstandard JSON number literals.
    payload = report.to_dict()
    assert json.loads(report.to_json()) == json.loads(json.dumps(payload, allow_nan=False))
    assert isinstance(report.issues, tuple)
    for issue in report.issues:
        assert isinstance(issue.code, str) and issue.code
        assert isinstance(issue.message, str) and issue.message
        assert issue.severity in {"error", "warning", "info"}
        assert isinstance(issue.row_positions, tuple)
        assert isinstance(issue.columns, tuple)
        assert all(isinstance(position, int) and position >= 0 for position in issue.row_positions)


def test_valid_report_counts_and_exception_contract() -> None:
    report = validate(simple_frame(), config=analysis_config())
    assert isinstance(report, ValidationReport)
    assert report.valid is True
    assert not errors(report)
    assert COUNT_FIELDS <= set(report.counts)
    assert {key: report.counts[key] for key in COUNT_FIELDS} == {
        "n_records": 4, "n_key_complete": 4, "n_retained": 4, "n_excluded": 0,
        "n_error_records": 0, "n_duplicate_keys": 0, "n_duplicate_rows": 0,
        "n_sources": 2, "n_units": 4, "n_groups": 2, "total_weight": 4.0,
    }
    assert report.raise_for_errors() is None
    assert_json_report(report)


@pytest.mark.parametrize("example", EXPECTED["examples"], ids=lambda example: example["id"])
def test_all_four_synthetic_omics_examples_validate_without_data_changes(example) -> None:
    raw_config = json.loads((EXAMPLES / example["config_file"]).read_text(encoding="utf-8"))
    frame = pd.read_csv(EXAMPLES / example["data_file"], dtype=str, keep_default_na=False)
    before = frame.copy(deep=True)
    config = AnalysisConfig.from_dict(raw_config)
    config_before = config.to_dict()
    report = validate(frame, config=config)
    assert report.valid, report.to_json()
    design = example["design"]
    for reported, expected in {
        "n_records": "n_records", "n_retained": "n_incidences", "n_sources": "n_sources",
        "n_units": "n_units", "n_groups": "n_groups", "total_weight": "total_mass",
    }.items():
        assert report.counts[reported] == design[expected]
    assert report.counts["n_excluded"] == 0
    assert report.counts["n_error_records"] == 0
    pd.testing.assert_frame_equal(frame, before)
    assert config.to_dict() == config_before
    assert_json_report(report)


@pytest.mark.parametrize("missing", [None, pd.NA, np.nan, "", "  "])
def test_missing_required_keys_are_errors_by_default_with_physical_positions(missing) -> None:
    frame = simple_frame()
    frame.loc[1, "unit"] = missing
    frame.index = pd.Index([17, 17, 3, 3], name="original_index")
    report = validate(frame, config=analysis_config())
    assert not report.valid
    assert positions_for(report, severity="error", column="unit") == {1}
    assert report.counts["n_key_complete"] == 3
    assert report.counts["n_error_records"] == 1
    with pytest.raises(MetadataValidationError) as caught:
        report.raise_for_errors()
    assert caught.value.report is report
    assert_json_report(report)


def test_explicit_missing_drop_preserves_input_and_reports_exclusions() -> None:
    frame = simple_frame().assign(mass=[1.0, np.inf, 2.0, 3.0])
    frame.loc[1, "unit"] = None
    frame.index = [5, 5, 10, 10]
    before = frame.copy(deep=True)
    report = validate(frame, config=weighted_config(policies={"missing_required": "drop"}))
    assert report.valid, report.to_json()
    assert report.counts["n_records"] == 4
    assert report.counts["n_key_complete"] == report.counts["n_retained"] == 3
    assert report.counts["n_excluded"] == 1
    assert report.counts["n_error_records"] == 0
    assert report.counts["total_weight"] == 6.0
    assert positions_for(report, severity="warning", column="unit") == {1}
    assert not any(issue.severity == "error" and "mass" in issue.columns for issue in report.issues)
    pd.testing.assert_frame_equal(frame, before)
    assert_json_report(report)


def test_missing_drop_excludes_rows_before_hierarchy_checks() -> None:
    frame = simple_frame().assign(donor=["D1", None, "D3", "D4"])
    frame.loc[1, "source"] = None
    report = validate(frame, config=analysis_config(
        hierarchy={"donor": ["donor"]},
        policies={"missing_required": "drop", "missing_parent": "error"},
    ))
    assert report.valid, report.to_json()
    assert report.counts["n_retained"] == 3
    assert not any(issue.severity == "error" and "donor" in issue.columns for issue in report.issues)


def test_duplicate_keys_fail_by_default_and_report_all_record_positions() -> None:
    frame = pd.concat([simple_frame(), simple_frame().iloc[[0, 0]]], ignore_index=True)
    frame.index = [4] * len(frame)
    report = validate(frame, config=analysis_config())
    assert not report.valid
    assert report.counts["n_duplicate_keys"] == 1
    assert report.counts["n_duplicate_rows"] == 2
    assert report.counts["n_error_records"] == 3
    assert {0, 4, 5} <= positions_for(report, severity="error", column="unit")
    assert_json_report(report)


def test_keep_records_is_explicit_and_keeps_their_mass_with_warning() -> None:
    frame = pd.DataFrame({"source": ["S1", "S1"], "unit": ["U1", "U1"],
                          "group": ["A", "A"], "run": ["R1", "R2"], "mass": [2.0, 3.0]})
    report = validate(frame, config=weighted_config(policies={"duplicates": "keep_records"}))
    assert report.valid, report.to_json()
    assert report.counts["n_retained"] == 2
    assert report.counts["n_units"] == 1
    assert report.counts["total_weight"] == 5.0
    assert report.counts["n_duplicate_rows"] == 1
    assert {0, 1} <= positions_for(report, severity="warning", column="unit")


def test_collapse_identical_uses_normalized_keys_and_equal_numeric_weights() -> None:
    frame = pd.DataFrame({"source": ["S1", " S1 "], "unit": [" U1", "U1 "],
                          "group": [" A", "A "], "run": ["R1", "R1"],
                          "mass": [2.0, "2.0"], "optional_note": [None, None]})
    before = frame.copy(deep=True)
    report = validate(frame, config=weighted_config(policies={"duplicates": "collapse_identical"}))
    assert report.valid, report.to_json()
    assert report.counts["n_key_complete"] == 2
    assert report.counts["n_retained"] == report.counts["n_excluded"] == 1
    assert report.counts["total_weight"] == 2.0
    assert report.counts["n_duplicate_keys"] == report.counts["n_duplicate_rows"] == 1
    assert any(issue.severity == "warning" and "unit" in issue.columns for issue in report.issues)
    pd.testing.assert_frame_equal(frame, before)


@pytest.mark.parametrize("column,values", [("run", ["R1", "R2"]), ("mass", [2.0, 3.0])])
def test_collapse_identical_rejects_varying_record_metadata_or_weights(column, values) -> None:
    frame = pd.DataFrame({"source": ["S1", "S1"], "unit": ["U1", "U1"],
                          "group": ["A", "A"], "run": ["R1", "R1"], "mass": [2.0, 2.0]})
    frame[column] = values
    report = validate(frame, config=weighted_config(policies={"duplicates": "collapse_identical"}))
    assert not report.valid
    assert report.counts["n_error_records"] == 2
    assert errors(report)


@pytest.mark.parametrize("duplicates", ["error", "keep_records", "collapse_identical"])
def test_normalization_group_conflicts_are_checked_before_duplicate_policy(duplicates) -> None:
    frame = pd.DataFrame({"source": ["S1", "S1"], "unit": [" U1 ", "U1"],
                          "group": ["A", "B"]})
    report = validate(frame, config=analysis_config(policies={"duplicates": duplicates}))
    assert not report.valid
    assert positions_for(report, severity="error", column="group") == {0, 1}
    assert report.counts["n_error_records"] == 2


@pytest.mark.parametrize("duplicates", ["error", "keep_records", "collapse_identical"])
def test_conflicting_declared_parents_fail_before_duplicate_handling(duplicates) -> None:
    frame = pd.DataFrame({"source": ["S1", "S1"], "unit": ["U1", "U1"],
                          "group": ["A", "A"], "donor": ["D1", "D2"]})
    report = validate(frame, config=analysis_config(
        hierarchy={"donor": ["donor"]}, policies={"duplicates": duplicates},
    ))
    assert not report.valid
    assert positions_for(report, severity="error", column="donor") == {0, 1}
    assert report.counts["n_error_records"] == 2


def test_partial_parent_mapping_warns_without_inventing_parents() -> None:
    frame = pd.DataFrame({"source": ["S1", "S2", "S3"], "unit": ["U1", "U1", "U2"],
                          "group": ["A", "A", "B"], "donor": ["D1", None, None]})
    before = frame.copy(deep=True)
    report = validate(frame, config=analysis_config(hierarchy={"donor": ["donor"]}))
    assert report.valid, report.to_json()
    assert {1, 2} <= positions_for(report, severity="warning", column="donor")
    assert report.counts["n_retained"] == 3
    assert report.counts["n_units"] == 2
    pd.testing.assert_frame_equal(frame, before)
    strict = validate(frame, config=analysis_config(
        hierarchy={"donor": ["donor"]}, policies={"missing_parent": "error"},
    ))
    assert not strict.valid
    assert {1, 2} <= positions_for(strict, severity="error", column="donor")


@pytest.mark.parametrize("bad_weight", [
    None, pd.NA, np.nan, "invalid", -1.0, np.inf, -np.inf, 1 + 2j, 1 + 0j,
    True, np.bool_(False), datetime(2026, 1, 1), pd.Timestamp("2026-01-01"),
    np.datetime64("2026-01-01"), np.timedelta64(1, "ns"), Decimal("sNaN"),
])
def test_eligible_invalid_weights_are_errors_with_null_total(bad_weight) -> None:
    frame = simple_frame()
    frame["mass"] = pd.Series([1.0, bad_weight, 1.0, 1.0], dtype=object)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        report = validate(frame, config=weighted_config())
    assert not report.valid
    assert positions_for(report, severity="error", column="mass") == {1}
    assert report.counts["total_weight"] is None
    assert_json_report(report)


@pytest.mark.parametrize("weights,expected_total", [([0.0] * 4, 0.0), ([1e308] * 4, None)])
def test_zero_or_overflowing_weight_mass_is_an_error_without_runtime_warning(weights, expected_total) -> None:
    frame = simple_frame().assign(mass=weights)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        report = validate(frame, config=weighted_config())
    assert not report.valid
    assert errors(report)
    assert report.counts["total_weight"] == expected_total
    assert_json_report(report)


def test_zero_weight_rows_are_retained_when_total_mass_is_positive() -> None:
    report = validate(simple_frame().assign(mass=[0, 0, 2, "3"]), config=weighted_config())
    assert report.valid, report.to_json()
    assert report.counts["n_retained"] == report.counts["n_units"] == 4
    assert report.counts["total_weight"] == 5.0


def test_composite_keys_keep_reused_sample_identifiers_and_parent_keys_distinct() -> None:
    frame = pd.DataFrame({"study": ["D1", "D2"], "batch": ["B1", "B1"],
                          "sample": ["001", "001"], "group": ["A", "B"],
                          "donor": ["01", "01"]})
    config = analysis_config(
        columns={"source": ["study", "batch"], "unit": ["study", "sample"], "group": ["group"]},
        hierarchy={"donor": ["study", "donor"]},
    )
    report = validate(frame, config=config)
    assert report.valid, report.to_json()
    assert report.counts["n_sources"] == report.counts["n_units"] == 2
    assert_json_report(report)


def test_homogeneous_tuple_keys_validate_with_the_same_counts_as_explicit_components() -> None:
    frame = pd.DataFrame({"source": [("D1", "B1"), ("D2", "B1")],
                          "unit": [("D1", "001"), ("D2", "001")],
                          "group": [("A", "baseline"), ("B", "baseline")]})
    report = validate(frame, config=analysis_config())
    assert report.valid, report.to_json()
    assert report.counts["n_sources"] == report.counts["n_units"] == report.counts["n_groups"] == 2
    assert_json_report(report)


@pytest.mark.parametrize("key", [["D1", "001"], {"study": "D1"}, np.array(["D1", "001"])])
def test_nonscalar_keys_produce_structured_errors_instead_of_tracebacks(key) -> None:
    frame = simple_frame()
    frame["unit"] = frame["unit"].astype(object)
    frame.at[1, "unit"] = key
    report = validate(frame, config=analysis_config())
    assert not report.valid
    assert 1 in positions_for(report, severity="error", column="unit")
    assert_json_report(report)


def test_mixed_scalar_and_tuple_key_shapes_are_rejected() -> None:
    frame = simple_frame()
    frame.at[1, "unit"] = ("D1", "U2")
    report = validate(frame, config=analysis_config())
    assert not report.valid
    assert any("unit" in issue.columns for issue in errors(report))


@pytest.mark.parametrize("token", ["none", "NaN", "<NA>", " NONE "])
@pytest.mark.parametrize("column", ["source", "unit", "group", "donor"])
def test_core_reserved_literal_keys_require_explicit_missing_token_policy(token, column) -> None:
    frame = simple_frame().assign(donor=["D1", "D2", "D3", "D4"])
    frame.at[1, column] = token
    report = validate(frame, config=analysis_config(hierarchy={"donor": ["donor"]}))
    assert not report.valid
    assert any(issue.code == "core_reserved_key" and issue.severity == "error"
               and 1 in issue.row_positions and column in issue.columns for issue in report.issues)


def test_literal_na_and_leading_zero_identifiers_remain_valid() -> None:
    frame = pd.DataFrame({"source": ["NA", "S2", "S3"], "unit": ["001", "1", "01"],
                          "group": ["NA", "B", "A"]})
    before = frame.copy(deep=True)
    report = validate(frame, config=analysis_config())
    assert report.valid, report.to_json()
    assert report.counts["n_units"] == 3
    assert report.counts["n_groups"] == 3
    pd.testing.assert_frame_equal(frame, before)


def test_custom_missing_tokens_are_case_sensitive_after_trimming() -> None:
    frame = pd.DataFrame({"source": [" MISSING ", "missing", "S1"],
                          "unit": ["U1", "U2", "U3"], "group": ["A", "B", "A"]})
    report = validate(frame, config=analysis_config(
        policies={"missing_required": "drop", "missing_tokens": ["MISSING"]},
    ))
    assert report.valid, report.to_json()
    assert report.counts["n_retained"] == 2
    assert report.counts["n_excluded"] == 1
    assert positions_for(report, severity="warning", column="source") == {0}


def test_declared_reserved_token_is_missing_but_undeclared_case_variant_is_an_error() -> None:
    frame = pd.DataFrame({"source": [" none ", "NONE", "S1"],
                          "unit": ["U1", "U2", "U3"], "group": ["A", "B", "A"]})
    report = validate(frame, config=analysis_config(
        policies={"missing_required": "drop", "missing_tokens": ["none"]},
    ))
    assert not report.valid
    reserved = [issue for issue in report.issues if issue.code == "core_reserved_key"]
    assert reserved and {position for issue in reserved for position in issue.row_positions} == {1}
    assert report.counts["n_excluded"] == 1


def test_custom_missing_tokens_apply_to_required_and_parent_keys() -> None:
    frame = simple_frame().assign(donor=["D1", "D2", "UNKNOWN", "D4"])
    frame.at[0, "unit"] = "UNKNOWN"
    frame.at[1, "group"] = "UNKNOWN"
    report = validate(frame, config=analysis_config(
        hierarchy={"donor": ["donor"]},
        policies={"missing_required": "drop", "missing_tokens": ["UNKNOWN"]},
    ))
    assert report.valid, report.to_json()
    assert report.counts["n_retained"] == report.counts["n_excluded"] == 2
    assert 0 in positions_for(report, severity="warning", column="unit")
    assert 1 in positions_for(report, severity="warning", column="group")
    assert 2 in positions_for(report, severity="warning", column="donor")


def test_key_missing_tokens_do_not_hide_invalid_eligible_weights() -> None:
    frame = simple_frame().assign(mass=["UNKNOWN", "1", "1", "1"])
    report = validate(frame, config=weighted_config(
        policies={"missing_required": "drop", "missing_tokens": ["UNKNOWN"]},
    ))
    assert not report.valid
    assert positions_for(report, severity="error", column="mass") == {0}
    assert report.counts["n_excluded"] == 0
    assert report.counts["total_weight"] is None


def test_numeric_and_trimmed_identifiers_are_tracked_and_numeric_groups_are_categorical() -> None:
    frame = pd.DataFrame({"source": [1, 2, 3], "unit": [" 001 ", "002", "003"],
                          "group": [0.1, 0.2, 0.3]})
    before = frame.copy(deep=True)
    report = validate(frame, config=analysis_config())
    assert report.valid, report.to_json()
    assert report.counts["n_sources"] == report.counts["n_units"] == report.counts["n_groups"] == 3
    tracked = {column for issue in report.issues for column in issue.columns}
    assert {"source", "unit", "group"} <= tracked
    pd.testing.assert_frame_equal(frame, before)


def test_same_unit_in_multiple_sources_is_valid_incidence_weighting() -> None:
    frame = pd.DataFrame({"source": ["S1", "S2", "S3"], "unit": ["U1", "U1", "U2"],
                          "group": ["A", "A", "B"]})
    report = validate(frame, config=analysis_config())
    assert report.valid, report.to_json()
    assert report.counts["n_units"] == 2
    assert report.counts["n_sources"] == 3
    assert report.counts["n_retained"] == report.counts["total_weight"] == 3
    assert report.counts["n_duplicate_keys"] == 0


def test_one_group_has_applicability_warning_without_invalidating_metadata() -> None:
    report = validate(simple_frame().assign(group="A"), config=analysis_config())
    assert report.valid
    assert report.counts["n_groups"] == 1
    assert any(issue.code == "no_group_contrast" and issue.severity == "warning"
               for issue in report.issues)


@pytest.mark.parametrize("case", ["empty", "missing_column", "duplicate_header"])
def test_table_structure_errors_are_reports_and_serialize_safely(case) -> None:
    if case == "empty":
        frame = simple_frame().iloc[:0]
    elif case == "missing_column":
        frame = simple_frame().drop(columns="group")
    else:
        frame = pd.concat([simple_frame(), simple_frame()[["unit"]]], axis=1)
    report = validate(frame, config=analysis_config())
    assert isinstance(report, ValidationReport)
    assert not report.valid
    assert errors(report)
    if case == "empty":
        assert report.counts["total_weight"] == 0.0
    assert_json_report(report)


def test_wrong_programmer_argument_types_raise_instead_of_becoming_data_errors() -> None:
    with pytest.raises(TypeError):
        validate([{"source": "S1"}], config=analysis_config())
    with pytest.raises((TypeError, ValueError)):
        validate(simple_frame(), config={})


def test_row_permutation_and_arbitrary_column_names_preserve_valid_design_counts() -> None:
    frame = simple_frame()
    original = validate(frame, config=analysis_config())
    renamed = frame.rename(columns={"source": "provenance_id", "unit": "entity_id", "group": "category"})
    report = validate(renamed.iloc[[3, 1, 0, 2]], config=analysis_config(
        columns={"source": ["provenance_id"], "unit": ["entity_id"], "group": ["category"]},
    ))
    assert report.valid
    assert dict(report.counts) == dict(original.counts)


def test_report_issue_config_and_counts_are_immutable_and_exports_are_independent() -> None:
    frame = simple_frame()
    frame.at[1, "unit"] = None
    report = validate(frame, config=analysis_config())
    before = report.to_dict()
    with pytest.raises(TypeError):
        report.counts["n_records"] = 999
    with pytest.raises(FrozenInstanceError):
        report.issues = ()
    with pytest.raises(FrozenInstanceError):
        report.issues[0].row_positions = (999,)
    with pytest.raises(FrozenInstanceError):
        report.config.name = "changed"
    exported = report.to_dict()
    exported["counts"]["n_records"] = 999
    exported["issues"][0]["row_positions"].append(999)
    exported["config"]["columns"]["unit"].append("new_column")
    assert report.to_dict() == before


def test_report_copies_caller_owned_issue_and_count_containers() -> None:
    baseline = validate(simple_frame().assign(group="A"), config=analysis_config())
    mutable_issues = list(baseline.issues)
    mutable_counts = dict(baseline.counts)
    report = ValidationReport(config=baseline.config, issues=mutable_issues, counts=mutable_counts)
    mutable_issues.clear()
    mutable_counts["n_records"] = 999
    assert isinstance(report.issues, tuple)
    assert report.issues == baseline.issues
    assert report.counts["n_records"] == 4


def test_invalid_report_retains_candidate_rows_without_reporting_silent_exclusions() -> None:
    frame = pd.DataFrame({"source": ["S1", "S1", "S2"], "unit": ["U1", "U1", "U2"],
                          "group": ["A", "B", "B"], "mass": [1.0, 1.0, np.inf]})
    report = validate(frame, config=weighted_config())
    assert not report.valid
    assert report.counts["n_key_complete"] == report.counts["n_retained"] == 3
    assert report.counts["n_error_records"] == 3
    assert report.counts["n_excluded"] == 0
    assert report.counts["total_weight"] is None


def test_zero_weight_group_is_counted_but_has_no_positive_mass_contrast() -> None:
    frame = pd.DataFrame({"source": ["S1", "S2"], "unit": ["U1", "U2"],
                          "group": ["A", "B"], "mass": [3.0, 0.0]})
    report = validate(frame, config=weighted_config())
    assert report.valid
    assert report.counts["n_groups"] == 2
    assert report.counts["n_retained"] == 2
    assert report.counts["total_weight"] == 3.0
    assert any(issue.code == "no_group_contrast" for issue in report.issues)


def test_tuple_components_do_not_collide_when_their_display_delimiters_overlap() -> None:
    frame = pd.DataFrame({"study": ["a,b", "a"], "sample": ["c", "b,c"],
                          "source": ["S1", "S2"], "group": ["A", "B"]})
    report = validate(frame, config=analysis_config(
        columns={"source": ["source"], "unit": ["study", "sample"], "group": ["group"]},
    ))
    assert report.valid
    assert report.counts["n_units"] == 2
    assert report.counts["n_duplicate_keys"] == 0


def test_missing_tuple_components_follow_explicit_missing_required_policy() -> None:
    frame = pd.DataFrame({"source": ["S1", "S2"], "unit": [("D1", "001"), ("D2", None)],
                          "group": ["A", "B"]})
    report = validate(frame, config=analysis_config(policies={"missing_required": "drop"}))
    assert report.valid
    assert report.counts["n_key_complete"] == report.counts["n_retained"] == 1
    assert report.counts["n_excluded"] == 1
    assert 1 in positions_for(report, severity="warning", column="unit")


def test_tuple_valued_columns_cannot_be_nested_inside_composite_column_mappings() -> None:
    frame = pd.DataFrame({"study": ["D1", "D2"], "sample": [("sample", "001"), ("sample", "002")],
                          "source": ["S1", "S2"], "group": ["A", "B"]})
    report = validate(frame, config=analysis_config(
        columns={"source": ["source"], "unit": ["study", "sample"], "group": ["group"]},
    ))
    assert not report.valid
    assert report.counts["n_error_records"] == 2
    assert_json_report(report)


@pytest.mark.parametrize("duplicates,valid", [("error", False), ("keep_records", True)])
def test_numeric_and_text_identifiers_share_a_key_after_declared_normalization(duplicates, valid) -> None:
    frame = pd.DataFrame({"source": ["S1", "S1"], "unit": pd.Series([1, "1"], dtype=object),
                          "group": ["A", "A"]})
    report = validate(frame, config=analysis_config(policies={"duplicates": duplicates}))
    assert report.valid is valid
    assert report.counts["n_units"] == 1
    assert report.counts["n_retained"] == 2
    assert report.counts["n_duplicate_keys"] == report.counts["n_duplicate_rows"] == 1
    assert any(issue.code == "key_normalized" and issue.row_positions == (0,)
               and "unit" in issue.columns for issue in report.issues)


@pytest.mark.parametrize("conflict", ["group", "parent"])
def test_zero_weight_rows_still_participate_in_group_and_parent_consistency_checks(conflict) -> None:
    frame = pd.DataFrame({"source": ["S1", "S2"], "unit": ["U1", "U1"],
                          "group": ["A", "A"], "donor": ["D1", "D1"], "mass": [1.0, 0.0]})
    column = "group" if conflict == "group" else "donor"
    frame.at[1, column] = "B" if conflict == "group" else "D2"
    report = validate(frame, config=weighted_config(hierarchy={"donor": ["donor"]}))
    assert not report.valid
    assert report.counts["n_key_complete"] == report.counts["n_retained"] == 2
    assert report.counts["n_excluded"] == 0
    assert report.counts["total_weight"] == 1.0
    assert positions_for(report, severity="error", column=column) == {0, 1}


def test_explicitly_dropping_all_missing_key_rows_reports_known_zero_retained_mass() -> None:
    frame = simple_frame().assign(unit=None)
    report = validate(frame, config=analysis_config(policies={"missing_required": "drop"}))
    assert not report.valid
    assert report.counts["n_records"] == report.counts["n_excluded"] == 4
    assert report.counts["n_key_complete"] == report.counts["n_retained"] == 0
    assert report.counts["total_weight"] == 0.0
    assert_json_report(report)


@pytest.mark.parametrize("key", [Decimal("sNaN"), Decimal("Infinity"), Decimal("-Infinity")])
def test_nonfinite_decimal_keys_return_structured_errors(key) -> None:
    frame = simple_frame()
    frame["unit"] = pd.Series(["U1", key, "U3", "U4"], dtype=object)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        report = validate(frame, config=analysis_config())
    assert not report.valid
    assert positions_for(report, severity="error", column="unit") == {1}
    assert_json_report(report)


def test_finite_decimal_weights_are_supported_and_serialize_as_finite_mass() -> None:
    frame = simple_frame()
    frame["mass"] = pd.Series([Decimal("0"), Decimal("1.25"), Decimal("2.75"), Decimal("1")], dtype=object)
    report = validate(frame, config=weighted_config())
    assert report.valid, report.to_json()
    assert report.counts["n_retained"] == 4
    assert report.counts["total_weight"] == 5.0
    assert_json_report(report)


def test_signaling_nan_in_optional_metadata_cannot_crash_duplicate_comparison() -> None:
    frame = pd.DataFrame({"source": ["S1", "S1"], "unit": ["U1", "U1"], "group": ["A", "A"]})
    frame["optional_note"] = pd.Series([Decimal("sNaN"), Decimal("sNaN")], dtype=object)
    report = validate(frame, config=analysis_config(policies={"duplicates": "collapse_identical"}))
    assert not report.valid
    assert report.counts["n_retained"] == 2
    assert report.counts["n_excluded"] == 0
    assert any(issue.code == "nonidentical_duplicate" and issue.row_positions == (0, 1)
               for issue in report.issues)
    assert_json_report(report)
