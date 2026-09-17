"""End-to-end public diagnostics on invented metadata and analytical fixtures."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from provenance_aliasing.api import (
    AnalysisConfig,
    DiagnosticComputationError,
    DiagnosticResult,
    MetadataValidationError,
    diagnose,
    validate,
)


EXAMPLES = Path(__file__).resolve().parent / "fixtures" / "synthetic"
EXPECTED = json.loads((EXAMPLES / "expected.json").read_text(encoding="utf-8"))
METRIC_COLUMNS = {"metric", "family", "value", "status", "reason", "weighting"}
STATUSES = {"ok", "undefined", "not_applicable", "not_requested", "resource_limited"}
AUDIT_TABLES = {"input.observations", "execution.stages"}


def config_for(**overrides) -> AnalysisConfig:
    raw = {
        "schema_version": "0.1-draft",
        "name": "synthetic-diagnosis",
        "columns": {"source": ["source"], "unit": ["unit"], "group": ["group"]},
        "grain": {"name": "study", "description": "Declared synthetic study"},
        "unit_description": "One explicitly identified synthetic sample",
    }
    raw.update(overrides)
    return AnalysisConfig.from_dict(raw)


def crossed_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "source": ["S1", "S1", "S2", "S2"],
        "unit": ["U1", "U2", "U3", "U4"],
        "group": ["A", "B", "A", "B"],
    })


def row_for(result: DiagnosticResult, metric: str) -> pd.Series:
    rows = result.metrics.loc[result.metrics["metric"] == metric]
    assert len(rows) == 1, f"Expected one row for {metric!r}; found {len(rows)}"
    return rows.iloc[0]


def assert_value(result: DiagnosticResult, metric: str, value) -> None:
    row = row_for(result, metric)
    assert row["status"] == "ok", row.to_dict()
    assert row["value"] == pytest.approx(value, rel=0, abs=EXPECTED["absolute_tolerance"])


def assert_unavailable(result: DiagnosticResult, metric: str, status: str) -> None:
    row = row_for(result, metric)
    assert row["status"] == status
    assert pd.isna(row["value"])
    assert isinstance(row["reason"], str) and row["reason"]
    serialized = next(item for item in result.to_dict()["metrics"] if item["metric"] == metric)
    assert serialized["value"] is None


def assert_json_contract(result: DiagnosticResult) -> dict:
    payload = result.to_dict()
    assert json.loads(result.to_json()) == json.loads(json.dumps(payload, allow_nan=False))
    assert payload["scope"] == "metadata_diagnostics"
    assert payload["schema_version"] == result.config.schema_version
    assert payload["validation"] == result.validation.to_dict()
    assert payload["config"] == result.config.to_dict()
    assert payload["package"]["name"] == "provenance-aliasing-diagnostics"
    assert isinstance(payload["package"]["version"], str) and payload["package"]["version"]
    assert isinstance(payload["metrics"], list)
    assert isinstance(payload["audit"], list)
    for table in payload["tables"].values():
        assert {"columns", "index", "index_names", "data"} <= set(table)
    return payload


@pytest.mark.parametrize("example", EXPECTED["examples"], ids=lambda example: example["id"])
def test_omics_examples_match_analytically_derived_metrics_and_counts(example) -> None:
    frame = pd.read_csv(EXAMPLES / example["data_file"], dtype=str, keep_default_na=False)
    config = AnalysisConfig.from_json(EXAMPLES / example["config_file"])
    original = frame.copy(deep=True)
    result = diagnose(frame, config=config)
    assert isinstance(result, DiagnosticResult)
    assert result.validation.valid
    assert METRIC_COLUMNS <= set(result.metrics.columns)
    assert result.metrics["metric"].is_unique
    assert set(result.metrics["status"]) <= STATUSES
    expected = example["design"]
    for metric, key in {
        "design.n_records": "n_records", "design.n_retained": "n_incidences",
        "design.n_sources": "n_sources", "design.n_units": "n_units",
        "design.n_groups": "n_groups", "design.total_weight": "total_mass",
        "structure.n_spanning_sources": "n_spanning_sources_structural",
        "entropy.n_spanning_sources_positive_mass": "n_spanning_sources_positive_mass",
    }.items():
        assert_value(result, metric, expected[key])
    assert_value(result, "design.n_excluded", 0)
    for metric, values in example["metrics"].items():
        if values["status"] == "ok":
            assert_value(result, metric, values["value"])
        else:
            assert_unavailable(result, metric, values["status"])
    assert {"entropy.per_source", "design.sources", "design.units", "structure.pair_sharing",
            "structure.source_removal", "structure.spanning_sources"} <= set(result.tables)
    assert result.audit["row_position"].tolist() == list(range(len(frame)))
    assert result.audit["status"].eq("retained").all()
    pd.testing.assert_frame_equal(frame, original)
    assert_json_contract(result)


def test_drop_and_identical_collapse_share_validation_counts_and_preserve_row_lineage() -> None:
    frame = pd.DataFrame({
        "source": ["S1", " S1 ", "S1", "S2", "S2", "S3"],
        "unit": ["U1", " U1 ", "U2", "U3", "U4", None],
        "group": ["A", " A ", "B", "A", "B", "A"],
        "mass": [1.0, "1", 1.0, 1.0, 1.0, np.inf],
        "run": ["R1", "R1", "R2", "R3", "R4", "R5"],
    }, index=[17, 17, 3, 3, 8, 8])
    config = config_for(
        weighting={"mode": "column", "column": "mass", "description": "Toy observation mass"},
        policies={"missing_required": "drop", "duplicates": "collapse_identical"},
    )
    original = frame.copy(deep=True)
    validation = validate(frame, config=config)
    result = diagnose(frame, config=config)
    assert dict(result.validation.counts) == dict(validation.counts)
    assert_value(result, "design.n_records", 6)
    assert_value(result, "design.n_retained", 4)
    assert_value(result, "design.n_excluded", 2)
    assert_value(result, "design.total_weight", 4.0)
    assert_value(result, "entropy.ratio", 1.0)
    assert_value(result, "ceiling.balanced_accuracy", 1.0)
    ledger = result.audit.set_index("row_position")
    assert ledger.index.tolist() == [0, 1, 2, 3, 4, 5]
    assert ledger["status"].tolist() == ["retained", "collapsed", "retained", "retained", "retained", "excluded"]
    assert ledger.at[1, "reason"] == "identical_duplicate"
    assert ledger.at[1, "representative_row_position"] == 0
    assert ledger.at[1, "projection_row_position"] == ledger.at[0, "projection_row_position"] == 0
    assert {"source", "unit", "group"} <= set(ledger.at[1, "normalized_columns"])
    assert ledger.at[5, "reason"] == "missing_required_key"
    assert pd.isna(ledger.at[5, "representative_row_position"])
    assert pd.isna(ledger.at[5, "projection_row_position"])
    pd.testing.assert_frame_equal(frame, original)
    assert_json_contract(result)


def test_explicit_keep_records_preserves_repeated_observation_mass() -> None:
    frame = pd.concat([crossed_frame(), crossed_frame().iloc[[0]]], ignore_index=True)
    config = config_for(policies={"duplicates": "keep_records"})
    result = diagnose(frame, config=config)
    assert_value(result, "design.n_records", 5)
    assert_value(result, "design.n_retained", 5)
    assert_value(result, "design.n_units", 4)
    assert_value(result, "design.total_weight", 5)
    assert result.validation.counts["n_duplicate_rows"] == 1
    assert result.audit["status"].eq("retained").all()
    assert 0 < row_for(result, "entropy.ratio")["value"] < 1


def test_declared_missing_tokens_are_applied_before_core_diagnostics() -> None:
    frame = pd.concat([
        crossed_frame(),
        pd.DataFrame({"source": ["S3"], "unit": ["U5"], "group": [" MISSING "]}),
    ], ignore_index=True)
    config = config_for(policies={"missing_required": "drop", "missing_tokens": ["MISSING"]})
    result = diagnose(frame, config=config)
    assert_value(result, "design.n_records", 5)
    assert_value(result, "design.n_retained", 4)
    assert_value(result, "design.n_groups", 2)
    assert_value(result, "entropy.ratio", 1)
    assert result.audit.iloc[4]["status"] == "excluded"
    assert result.audit.iloc[4]["reason"] == "missing_required_key"


def test_unused_assay_metadata_does_not_change_diagnostic_values() -> None:
    frame = crossed_frame()
    extra = frame.assign(peptide_count=[100, 200, 300, 400], rna_depth=[2000, 1000, 9000, 20])
    extra["feature_ids"] = [["peptide-A"], ["gene-B", "gene-C"], [], ["metabolite-D"]]
    original = extra.copy(deep=True)
    baseline = diagnose(frame, config=config_for())
    result = diagnose(extra, config=config_for())
    pd.testing.assert_frame_equal(result.metrics, baseline.metrics)
    pd.testing.assert_frame_equal(extra, original)


def test_invalid_metadata_raises_the_structured_validation_error() -> None:
    frame = crossed_frame()
    frame.at[0, "group"] = None
    with pytest.raises(MetadataValidationError) as caught:
        diagnose(frame, config=config_for())
    assert not caught.value.report.valid
    assert any(issue.code == "missing_required_key" and issue.row_positions == (0,)
               for issue in caught.value.report.issues)


def test_single_group_returns_defined_entropy_and_explicit_unavailable_contrasts() -> None:
    result = diagnose(crossed_frame().assign(group="A"), config=config_for())
    assert result.validation.valid
    assert_value(result, "entropy.H_G", 0)
    assert_value(result, "entropy.H_G_given_D", 0)
    assert_value(result, "entropy.mutual_information", 0)
    assert_unavailable(result, "entropy.ratio", "undefined")
    assert_unavailable(result, "entropy.U_G_given_D", "undefined")
    assert_unavailable(result, "ceiling.balanced_accuracy", "not_applicable")
    assert_json_contract(result)


def test_three_groups_have_valid_entropy_and_inapplicable_binary_ceiling() -> None:
    frame = pd.DataFrame({"source": ["S1"] * 3 + ["S2"] * 3,
                          "unit": [f"U{i}" for i in range(6)], "group": ["A", "B", "C"] * 2})
    result = diagnose(frame, config=config_for())
    assert_value(result, "entropy.H_G", np.log2(3))
    assert_value(result, "entropy.ratio", 1)
    assert_unavailable(result, "ceiling.balanced_accuracy", "not_applicable")


@pytest.mark.parametrize("groups", [["total", "NA", "total", "NA"],
                                    [("A", "baseline"), ("B", "baseline")] * 2])
def test_literal_and_tuple_groups_survive_tables_without_formatted_label_collisions(groups) -> None:
    frame = crossed_frame()
    frame["group"] = pd.Series(groups, dtype=object)
    result = diagnose(frame, config=config_for())
    assert_value(result, "entropy.ratio", 1)
    assert_value(result, "ceiling.balanced_accuracy", 1)
    pair_sharing = result.tables["structure.pair_sharing"]
    assert set(pair_sharing.index) == set(groups)
    removal = result.tables["structure.source_removal"]
    assert removal["emptied_total"].to_dict() == {"S1": 2, "S2": 2}
    by_group = result.tables["structure.source_removal_by_group"]
    assert set(by_group["group"]) == set(groups)
    assert by_group["units_emptied"].eq(1).all()
    assert by_group.groupby("source")["units_emptied"].sum().to_dict() == {"S1": 2, "S2": 2}
    assert_json_contract(result)


def test_composite_sources_and_units_are_restored_in_public_tables_and_json() -> None:
    frame = pd.DataFrame({"study": ["D1", "D1", "D2", "D2"], "batch": ["B1"] * 4,
                          "sample": ["001", "002", "001", "002"], "group": ["A", "B", "A", "B"]})
    config = config_for(columns={"source": ["study", "batch"], "unit": ["study", "sample"], "group": ["group"]})
    result = diagnose(frame, config=config)
    assert_value(result, "design.n_units", 4)
    assert_value(result, "entropy.ratio", 1)
    assert set(result.tables["entropy.per_source"].index) == {("D1", "B1"), ("D2", "B1")}
    removed = result.tables["structure.source_removal"]
    expected_units = {("D1", "001"), ("D1", "002"), ("D2", "001"), ("D2", "002")}
    assert {unit for keys in removed["emptied_unit_keys"] for unit in keys} == expected_units
    payload = assert_json_contract(result)
    assert payload["tables"]["entropy.per_source"]["index"] == [["D1", "B1"], ["D2", "B1"]]


def test_zero_weight_incidence_and_positive_mass_source_crossing_are_distinguished() -> None:
    frame = crossed_frame().assign(mass=[1.0, 0.0, 1.0, 1.0])
    config = config_for(weighting={"mode": "column", "column": "mass", "description": "Toy observation mass"})
    result = diagnose(frame, config=config)
    assert_value(result, "design.n_units", 4)
    assert_value(result, "design.total_weight", 3)
    assert_value(result, "structure.n_spanning_sources", 2)
    assert_value(result, "entropy.n_spanning_sources_positive_mass", 1)
    assert result.validation.counts["n_retained"] == 4


def test_requested_entropy_family_keeps_other_families_explicitly_not_requested() -> None:
    result = diagnose(crossed_frame(), config=config_for(diagnostics={"include": ["entropy"]}))
    assert_value(result, "entropy.ratio", 1)
    unrequested = result.metrics.loc[result.metrics["family"] != "entropy"]
    assert not unrequested.empty
    assert unrequested["status"].eq("not_requested").all()
    assert unrequested["value"].isna().all()
    assert set(result.tables) == AUDIT_TABLES | {"entropy.per_source"}
    assert_unavailable(result, "ceiling.balanced_accuracy", "not_requested")


def test_empty_diagnostic_selection_still_validates_and_audits_metadata() -> None:
    result = diagnose(crossed_frame(), config=config_for(diagnostics={"include": []}))
    assert result.validation.valid
    assert not result.metrics.empty
    assert result.metrics["status"].eq("not_requested").all()
    assert result.metrics["value"].isna().all()
    assert set(result.tables) == AUDIT_TABLES
    assert len(result.tables["input.observations"]) == 4
    assert result.tables["execution.stages"].empty
    assert len(result.audit) == 4
    assert result.audit["status"].eq("retained").all()
    assert_json_contract(result)


def test_bootstrap_is_seed_reproducible_and_can_be_requested_without_entropy_family() -> None:
    frame = pd.DataFrame({"source": ["S1", "S1", "S2", "S2", "S3", "S3"],
                          "unit": [f"U{i}" for i in range(6)], "group": ["A", "B", "A", "A", "B", "B"]})
    config = config_for(diagnostics={"include": [], "bootstrap_replicates": 32, "seed": 17})
    first = diagnose(frame, config=config)
    second = diagnose(frame, config=config)
    pd.testing.assert_frame_equal(first.metrics, second.metrics)
    pd.testing.assert_frame_equal(first.tables["bootstrap.replicates"], second.tables["bootstrap.replicates"])
    assert len(first.tables["bootstrap.replicates"]) == 32
    assert_value(first, "bootstrap.ratio_observed", 1 / 3)
    assert_unavailable(first, "entropy.ratio", "not_requested")
    effective = row_for(first, "bootstrap.n_effective")["value"]
    degenerate = row_for(first, "bootstrap.n_degenerate")["value"]
    assert effective + degenerate == 32
    assert 0 < effective <= 32
    assert 0 <= row_for(first, "bootstrap.lower")["value"] <= row_for(first, "bootstrap.upper")["value"] <= 1
    assert_json_contract(first)


def test_entropy_leave_one_source_out_returns_one_row_per_source() -> None:
    result = diagnose(crossed_frame(), config=config_for(diagnostics={"include": ["entropy_loso"]}))
    assert_value(result, "entropy_loso.n_sources", 2)
    assert len(result.tables["entropy_loso.sources"]) == 2
    assert set(result.tables["entropy_loso.sources"].index) == {"S1", "S2"}
    assert_unavailable(result, "entropy.ratio", "not_requested")


def test_rank_guard_detects_multiclass_disconnection_despite_one_crossed_source() -> None:
    frame = pd.DataFrame({"source": ["S1", "S1", "S2"], "unit": ["U1", "U2", "U3"],
                          "group": ["A", "B", "C"]})
    result = diagnose(frame, config=config_for())
    assert_value(result, "structure.n_spanning_sources", 1)
    assert_value(result, "guards.design_rank", 3)
    assert_value(result, "guards.design_columns", 4)
    assert_value(result, "guards.rank_deficient", 1)


def test_memory_budget_limits_dense_structure_and_rank_without_substituting_a_guess(monkeypatch) -> None:
    import provenance_aliasing.api.diagnostics as diagnostic_module

    def forbidden(*args, **kwargs):
        raise AssertionError("The memory preflight must stop this dense function before execution")

    for name in ("pair_sharing_by_group", "structural_leave_one_source_out", "design_rank_deficient"):
        monkeypatch.setattr(diagnostic_module, name, forbidden)
    n = 600
    frame = pd.DataFrame({"source": [f"S{i}" for i in range(n)], "unit": [f"U{i}" for i in range(n)],
                          "group": ["A", "B"] * (n // 2)})
    config = config_for(execution={"max_working_memory_mb": 1})
    result = diagnose(frame, config=config)
    assert result.validation.valid
    assert_value(result, "design.n_units", n)
    assert_value(result, "entropy.ratio", 0)
    assert_unavailable(result, "guards.design_rank", "resource_limited")
    assert_unavailable(result, "guards.rank_deficient", "resource_limited")
    assert_value(result, "structure.n_spanning_sources", 0)
    assert_unavailable(result, "structure.pair_sharing.n_groups", "resource_limited")
    assert_unavailable(result, "structure.source_removal.n_sources", "resource_limited")
    assert "structure.pair_sharing" not in result.tables
    stages = result.tables["execution.stages"].set_index("stage")
    for stage in ("structure.pair_sharing", "structure.source_removal", "guards.design_rank"):
        assert stages.at[stage, "status"] == "resource_limited"
        assert stages.at[stage, "estimated_working_bytes"] > stages.at[stage, "budget_bytes"]
    assert_json_contract(result)


def test_unexpected_numerical_stage_failure_preserves_stage_and_original_cause(monkeypatch) -> None:
    import provenance_aliasing.api.diagnostics as diagnostic_module

    sentinel = ValueError("synthetic numerical failure")

    def fail(*args, **kwargs):
        raise sentinel

    monkeypatch.setattr(diagnostic_module, "label_entropy", fail)
    with pytest.raises(DiagnosticComputationError) as caught:
        diagnose(crossed_frame(), config=config_for(diagnostics={"include": ["entropy"]}))
    assert caught.value.stage == "entropy"
    assert caught.value.__cause__ is sentinel


def test_unrepresentable_relative_weights_raise_instead_of_losing_a_group() -> None:
    frame = pd.DataFrame({"source": ["S1", "S2"], "unit": ["U1", "U2"],
                          "group": ["A", "B"], "mass": [1e308, 1e-100]})
    config = config_for(
        weighting={"mode": "column", "column": "mass", "description": "Extreme synthetic relative mass"},
        diagnostics={"include": ["guards"]},
    )
    assert validate(frame, config=config).valid
    with pytest.raises(DiagnosticComputationError) as caught:
        diagnose(frame, config=config)
    assert caught.value.stage == "guards.cramers_v"
    assert isinstance(caught.value.__cause__, ValueError)
    assert caught.value.validation.valid


@pytest.mark.parametrize("include", [[], ["entropy"]])
def test_unrequested_numerical_families_are_not_called(monkeypatch, include) -> None:
    import provenance_aliasing.api.diagnostics as diagnostic_module

    def forbidden(*args, **kwargs):
        raise AssertionError("An unrequested numerical family must not execute")

    names = ["balanced_accuracy_ceiling", "pair_sharing_by_group", "structural_leave_one_source_out",
             "cramers_v", "design_rank_deficient", "bootstrap_ratio"]
    if not include:
        names.append("label_entropy")
    for name in names:
        monkeypatch.setattr(diagnostic_module, name, forbidden)
    result = diagnose(crossed_frame(), config=config_for(diagnostics={"include": include}))
    assert result.validation.valid
    if include:
        assert_value(result, "entropy.ratio", 1)
    else:
        assert result.metrics["status"].eq("not_requested").all()


def test_two_retained_groups_with_only_one_positive_group_have_no_mass_contrast() -> None:
    frame = crossed_frame().assign(mass=[1.0, 0.0, 1.0, 0.0])
    config = config_for(
        weighting={"mode": "column", "column": "mass", "description": "Toy observation mass"},
        diagnostics={"include": ["design", "entropy", "ceiling", "structure", "guards"],
                     "bootstrap_replicates": 8, "seed": 17},
    )
    result = diagnose(frame, config=config)
    assert_value(result, "design.n_groups", 2)
    assert_value(result, "design.n_retained", 4)
    assert_value(result, "entropy.H_G", 0)
    assert_value(result, "structure.n_spanning_sources", 2)
    assert_value(result, "entropy.n_spanning_sources_positive_mass", 0)
    assert_unavailable(result, "entropy.ratio", "undefined")
    assert_unavailable(result, "ceiling.balanced_accuracy", "not_applicable")
    assert_unavailable(result, "guards.cramers_v", "undefined")
    assert_unavailable(result, "bootstrap.ratio_observed", "undefined")
    assert "bootstrap.replicates" not in result.tables
    assert_json_contract(result)


def test_three_retained_groups_do_not_become_binary_when_one_has_zero_mass() -> None:
    frame = pd.DataFrame({"source": ["S1"] * 3 + ["S2"] * 3,
                          "unit": [f"U{i}" for i in range(6)], "group": ["A", "B", "C"] * 2,
                          "mass": [1.0, 1.0, 0.0] * 2})
    result = diagnose(frame, config=config_for(
        weighting={"mode": "column", "column": "mass", "description": "Toy observation mass"},
    ))
    assert_value(result, "design.n_groups", 3)
    assert_value(result, "entropy.H_G", 1)
    assert_value(result, "entropy.ratio", 1)
    assert_unavailable(result, "ceiling.balanced_accuracy", "not_applicable")
    assert set(result.tables["structure.pair_sharing"].index) == {"A", "B", "C"}


def test_leave_one_source_out_is_inapplicable_with_only_one_source() -> None:
    result = diagnose(crossed_frame().assign(source="S1"), config=config_for(
        diagnostics={"include": ["entropy_loso"]},
    ))
    assert result.validation.valid
    assert_unavailable(result, "entropy_loso.n_sources", "not_applicable")
    assert "entropy_loso.sources" not in result.tables
    assert_json_contract(result)


def test_source_removal_that_leaves_zero_mass_has_explicit_undefined_values_and_reasons() -> None:
    frame = crossed_frame().assign(mass=[1.0, 1.0, 0.0, 0.0])
    result = diagnose(frame, config=config_for(
        weighting={"mode": "column", "column": "mass", "description": "Toy observation mass"},
        diagnostics={"include": ["entropy_loso"]},
    ))
    assert_value(result, "entropy_loso.n_sources", 2)
    table = result.tables["entropy_loso.sources"]
    for metric in ("ratio_without", "H_G_without", "H_G_given_D_without", "delta_ratio"):
        assert pd.isna(table.at["S1", metric])
        assert table.at["S1", f"{metric}_status"] == "undefined"
        assert table.at["S1", f"{metric}_reason"]
    assert table.at["S2", "ratio_without"] == pytest.approx(1)
    assert table.at["S2", "ratio_without_status"] == "ok"
    assert table.at["S1", "original_contribution"] == pytest.approx(1)
    assert table.at["S1", "original_contribution_status"] == "ok"
    payload = assert_json_contract(result)
    serialized = payload["tables"]["entropy_loso.sources"]
    row = serialized["data"][serialized["index"].index("S1")]
    assert row[serialized["columns"].index("ratio_without")] is None


def test_zero_weight_source_has_undefined_conditional_entropy_but_known_zero_contribution() -> None:
    frame = crossed_frame().assign(mass=[1.0, 1.0, 0.0, 0.0])
    result = diagnose(frame, config=config_for(
        weighting={"mode": "column", "column": "mass", "description": "Toy observation mass"},
        diagnostics={"include": ["entropy"]},
    ))
    table = result.tables["entropy.per_source"]
    assert pd.isna(table.at["S2", "H_G_given_d"])
    assert table.at["S2", "H_G_given_d_status"] == "undefined"
    assert table.at["S2", "H_G_given_d_reason"]
    assert table.at["S2", "source_probability"] == 0
    assert table.at["S2", "contribution"] == 0
    assert table.at["S1", "H_G_given_d"] == pytest.approx(1)
    assert table.at["S1", "H_G_given_d_status"] == "ok"
    payload = assert_json_contract(result)
    serialized = payload["tables"]["entropy.per_source"]
    row = serialized["data"][serialized["index"].index("S2")]
    assert row[serialized["columns"].index("H_G_given_d")] is None


def test_source_removal_propagates_unexpected_subanalysis_failure(monkeypatch) -> None:
    import provenance_aliasing.api.diagnostics as diagnostic_module

    original_entropy = diagnostic_module.label_entropy
    sentinel = ValueError("unexpected synthetic source-removal failure")
    calls = 0

    def fail_second_call(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise sentinel
        return original_entropy(*args, **kwargs)

    monkeypatch.setattr(diagnostic_module, "label_entropy", fail_second_call)
    with pytest.raises(DiagnosticComputationError) as caught:
        diagnose(crossed_frame(), config=config_for(diagnostics={"include": ["entropy_loso"]}))
    assert calls == 2
    assert caught.value.stage == "entropy_loso"
    assert caught.value.__cause__ is sentinel


@pytest.mark.parametrize("scale", [1e-300, 1.0, 1e300])
def test_probability_diagnostics_are_invariant_to_finite_uniform_weight_scale(scale) -> None:
    config = config_for(
        weighting={"mode": "column", "column": "mass", "description": "Uniform toy observation mass"},
        diagnostics={"include": ["entropy", "ceiling", "guards"], "bootstrap_replicates": 16, "seed": 11},
    )
    baseline = diagnose(crossed_frame().assign(mass=1.0), config=config)
    scaled = diagnose(crossed_frame().assign(mass=scale), config=config)
    for metric in ("entropy.H_G", "entropy.H_G_given_D", "entropy.ratio", "entropy.U_G_given_D",
                   "ceiling.balanced_accuracy", "guards.cramers_v", "bootstrap.ratio_observed",
                   "bootstrap.lower", "bootstrap.upper", "bootstrap.n_effective", "bootstrap.n_degenerate"):
        assert_value(scaled, metric, row_for(baseline, metric)["value"])
    pd.testing.assert_frame_equal(scaled.tables["bootstrap.replicates"], baseline.tables["bootstrap.replicates"])
    assert_json_contract(scaled)
