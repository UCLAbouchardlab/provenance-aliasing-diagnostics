"""Snapshot integrity and strict exports for completed metadata diagnostics."""
from __future__ import annotations

from dataclasses import FrozenInstanceError
from importlib.metadata import version
import json

import numpy as np
import pandas as pd
import pytest

from provenance_aliasing.api.config import AnalysisConfig
from provenance_aliasing.api.results import DiagnosticComputationError, DiagnosticResult
from provenance_aliasing.api.validation import MetadataValidationError, validate


METRIC_COLUMNS = ["metric", "family", "value", "status", "reason", "weighting"]


@pytest.fixture
def validation():
    config = AnalysisConfig.from_dict({
        "schema_version": "0.1-draft",
        "name": "synthetic-result",
        "columns": {"source": ["source"], "unit": ["unit"], "group": ["group"]},
        "grain": {"name": "study", "description": "The declared contributing study"},
        "unit_description": "An identified synthetic specimen",
    })
    return validate(pd.DataFrame({
        "source": ["S1", "S1", "S2", "S2"],
        "unit": ["U1", "U2", "U3", "U4"],
        "group": ["A", "B", "A", "B"],
    }), config=config)


def metrics_frame(**changes) -> pd.DataFrame:
    metric = {"metric": "entropy.H_G", "family": "entropy", "value": 1.0,
              "status": "ok", "reason": None, "weighting": "incidence"}
    metric.update(changes)
    return pd.DataFrame([metric], columns=METRIC_COLUMNS)


def result_for(validation, *, metrics=None, tables=None, audit=None) -> DiagnosticResult:
    return DiagnosticResult(
        validation=validation,
        metrics=metrics_frame() if metrics is None else metrics,
        tables={} if tables is None else tables,
        audit=pd.DataFrame(columns=["record_position", "status"]) if audit is None else audit,
    )


def test_result_preserves_assessment_and_reports_package_version(validation) -> None:
    metrics = metrics_frame()[list(reversed(METRIC_COLUMNS))]
    result = result_for(validation, metrics=metrics)
    assert result.validation is validation
    assert result.config is validation.config
    assert result.package_version == version("provenance-aliasing-diagnostics")
    assert list(result.metrics.columns) == METRIC_COLUMNS
    payload = result.to_dict()
    assert payload["schema_version"] == "0.1-draft"
    assert payload["scope"] == "metadata_diagnostics"
    assert payload["package"] == {"name": "provenance-aliasing-diagnostics", "version": result.package_version}
    assert payload["config"] == result.config.to_dict()
    assert payload["validation"] == validation.to_dict()
    assert json.loads(result.to_json()) == payload
    with pytest.raises(FrozenInstanceError):
        result.package_version = "changed"


def test_nested_object_cells_and_attributes_are_detached_on_input_and_access(validation) -> None:
    payload = {"keys": [("study-1", "unit-1")], "counts": np.array([1, 2])}
    table = pd.DataFrame({"payload": [payload]})
    table.index = pd.Index([("study-1", "batch-1")], dtype=object, tupleize_cols=False, name="source")
    table.attrs = {"assumptions": ["declared provenance"], "weighting": {"modes": ["incidence"]}}
    audit = pd.DataFrame({"positions": [[0, 1]], "details": [{"reasons": ["retained"]}]})
    metrics = metrics_frame()
    metrics.attrs = {"notes": ["original"]}
    tables = {"detail": table}
    result = result_for(validation, metrics=metrics, tables=tables, audit=audit)

    payload["keys"].append(("changed", "unit"))
    payload["counts"][0] = 999
    table.index.values[0] = ("changed", "source")
    table.attrs["assumptions"].append("changed")
    audit.at[0, "positions"].append(999)
    audit.at[0, "details"]["reasons"].append("changed")
    metrics.at[0, "value"] = 99
    metrics.attrs["notes"].append("changed")
    tables.clear()

    returned = result.tables["detail"]
    assert returned.at[("study-1", "batch-1"), "payload"]["keys"] == [("study-1", "unit-1")]
    np.testing.assert_array_equal(returned.iloc[0, 0]["counts"], [1, 2])
    assert returned.attrs["assumptions"] == ["declared provenance"]
    returned.iloc[0, 0]["keys"].clear()
    returned.iloc[0, 0]["counts"][0] = -1
    returned.attrs["weighting"]["modes"].append("changed")
    returned.index.values[0] = ("another", "source")
    returned_audit = result.audit
    returned_audit.at[0, "positions"].clear()
    returned_audit.at[0, "details"]["reasons"].clear()
    returned_metrics = result.metrics
    returned_metrics.at[0, "value"] = 7
    returned_metrics.attrs["notes"].clear()

    assert result.tables["detail"].index[0] == ("study-1", "batch-1")
    assert result.tables["detail"].iloc[0, 0]["keys"] == [("study-1", "unit-1")]
    np.testing.assert_array_equal(result.tables["detail"].iloc[0, 0]["counts"], [1, 2])
    assert result.tables["detail"].attrs["weighting"] == {"modes": ["incidence"]}
    assert result.audit.at[0, "positions"] == [0, 1]
    assert result.audit.at[0, "details"] == {"reasons": ["retained"]}
    assert result.metrics.at[0, "value"] == 1.0
    assert result.metrics.attrs == {"notes": ["original"]}
    with pytest.raises(TypeError):
        result.tables["new"] = pd.DataFrame()


def test_json_preserves_composite_axes_cells_and_table_attributes(validation) -> None:
    table = pd.DataFrame(
        [[("study-1", "unit-1"), np.int64(2), np.bool_(True)]],
        columns=pd.MultiIndex.from_tuples([("identity", "unit"), ("mass", "A"), ("flag", "crossed")]),
        index=pd.MultiIndex.from_tuples([("study-1", "batch-1")], names=["study", "batch"]),
    )
    table.attrs = {"weighting": "incidence", "assumptions": ("declared source",),
                   "undefined_detail": {"value": np.nan, "reason": "no support"}}
    audit = pd.DataFrame({"record_position": [np.int64(0)], "source": [("study-1", "batch-1")]})
    result = result_for(validation, tables={"source_group": table}, audit=audit)
    payload = result.to_dict()
    exported = payload["tables"]["source_group"]
    assert exported["columns"] == [["identity", "unit"], ["mass", "A"], ["flag", "crossed"]]
    assert exported["index"] == [["study-1", "batch-1"]]
    assert exported["index_names"] == ["study", "batch"]
    assert exported["data"] == [[["study-1", "unit-1"], 2, True]]
    assert exported["attrs"] == {"weighting": "incidence", "assumptions": ["declared source"],
                                 "undefined_detail": {"value": None, "reason": "no support"}}
    assert payload["audit"] == [{"record_position": 0, "source": ["study-1", "batch-1"]}]
    assert json.loads(result.to_json()) == payload
    json.dumps(payload, allow_nan=False)


def test_exports_are_fresh_containers(validation) -> None:
    result = result_for(validation, tables={"detail": pd.DataFrame({"source": [("S1", "batch")]})},
                        audit=pd.DataFrame({"positions": [[0, 1]]}))
    first = result.to_dict()
    first["tables"]["detail"]["data"][0][0].append("changed")
    first["metrics"][0]["value"] = 99
    first["audit"][0]["positions"].append(99)
    first["config"]["columns"]["source"].append("changed")
    first["validation"]["counts"]["n_records"] = 99
    second = result.to_dict()
    assert second["tables"]["detail"]["data"] == [[["S1", "batch"]]]
    assert second["metrics"][0]["value"] == 1.0
    assert second["audit"] == [{"positions": [0, 1]}]
    assert second["config"]["columns"]["source"] == ["source"]
    assert second["validation"]["counts"]["n_records"] == 4


@pytest.mark.parametrize("missing", [None, np.nan, pd.NA, pd.NaT, np.datetime64("NaT")])
def test_missing_detail_values_and_undefined_metric_values_export_as_null(validation, missing) -> None:
    table = pd.DataFrame({"value": pd.Series([missing], dtype=object), "status": ["undefined"]})
    metrics = metrics_frame(value=np.nan, status="undefined", reason="zero label entropy")
    result = result_for(validation, metrics=metrics, tables={"detail": table})
    assert result.to_dict()["metrics"][0]["value"] is None
    assert result.to_dict()["tables"]["detail"]["data"] == [[None, "undefined"]]


@pytest.mark.parametrize("bad_value", [np.inf, -np.inf, np.nan, None, pd.NA, True, "1.0", complex(1, 0)])
def test_successful_metric_requires_a_finite_real_number(validation, bad_value) -> None:
    with pytest.raises(ValueError, match="finite real numeric"):
        result_for(validation, metrics=metrics_frame(value=bad_value))


@pytest.mark.parametrize("status", ["undefined", "not_applicable", "not_requested", "resource_limited"])
def test_unavailable_statuses_require_missing_value_and_reason(validation, status: str) -> None:
    result = result_for(validation, metrics=metrics_frame(value=None, status=status, reason="explicit reason"))
    assert result.to_dict()["metrics"][0]["status"] == status
    with pytest.raises(ValueError, match="missing value"):
        result_for(validation, metrics=metrics_frame(value=0.0, status=status, reason="explicit reason"))
    with pytest.raises(ValueError, match="nonblank reason"):
        result_for(validation, metrics=metrics_frame(value=None, status=status, reason=None))


def test_metric_schema_and_identity_errors_fail_explicitly(validation) -> None:
    with pytest.raises(ValueError, match="columns must be exactly"):
        result_for(validation, metrics=metrics_frame().drop(columns="weighting"))
    with pytest.raises(ValueError, match="identifiers must be unique"):
        result_for(validation, metrics=pd.concat([metrics_frame(), metrics_frame()], ignore_index=True))
    with pytest.raises(ValueError, match="invalid status"):
        result_for(validation, metrics=metrics_frame(status="error"))
    with pytest.raises(ValueError, match="nonblank"):
        result_for(validation, metrics=metrics_frame(weighting=" "))


@pytest.mark.parametrize("in_attrs", [False, True])
@pytest.mark.parametrize("infinity", [np.inf, -np.inf])
def test_table_infinity_is_rejected_instead_of_becoming_null(validation, in_attrs: bool, infinity) -> None:
    table = pd.DataFrame({"value": [1.0 if in_attrs else infinity]})
    if in_attrs:
        table.attrs = {"total_mass": infinity}
    result = result_for(validation, tables={"detail": table})
    with pytest.raises(ValueError, match="infinite"):
        result.to_dict()
    with pytest.raises(ValueError, match="infinite"):
        result.to_json()


def test_unsupported_values_and_mapping_keys_are_not_stringified(validation) -> None:
    result = result_for(validation, tables={"detail": pd.DataFrame({"identity": [object()]})})
    with pytest.raises(TypeError, match="unsupported result value type"):
        result.to_json()
    table = pd.DataFrame({"value": [1.0]})
    table.attrs = {"mass_by_source": {("study", "batch"): 1.0}}
    result = result_for(validation, tables={"detail": table})
    with pytest.raises(TypeError, match="mapping keys must be strings"):
        result.to_dict()


def test_empty_tables_preserve_both_dimensions_and_index_names(validation) -> None:
    zero_rows = pd.DataFrame(columns=["source", "mass"], index=pd.Index([], name="row"))
    zero_columns = pd.DataFrame(index=pd.Index(["S1", "S2"], name="source"))
    result = result_for(validation, metrics=pd.DataFrame(columns=METRIC_COLUMNS),
                        tables={"zero_rows": zero_rows, "zero_columns": zero_columns},
                        audit=pd.DataFrame(index=[0, 1]))
    payload = result.to_dict()
    assert payload["metrics"] == []
    assert payload["tables"]["zero_rows"] == {
        "columns": ["source", "mass"], "index": [], "index_names": ["row"], "data": [], "attrs": {},
    }
    assert payload["tables"]["zero_columns"] == {
        "columns": [], "index": ["S1", "S2"], "index_names": ["source"], "data": [[], []], "attrs": {},
    }
    assert payload["audit"] == [{}, {}]


def test_categorical_and_nullable_frames_keep_types_when_copied(validation) -> None:
    table = pd.DataFrame({"category": pd.Categorical([("S1", "B1"), ("S2", "B2")]),
                          "mass": pd.array([1, None], dtype="Int64")})
    table.index = pd.CategoricalIndex(["a", "b"], name="row")
    result = result_for(validation, tables={"detail": table})
    pd.testing.assert_frame_equal(result.tables["detail"], table)
    assert result.to_dict()["tables"]["detail"]["data"] == [[["S1", "B1"], 1], [["S2", "B2"], None]]


def test_invalid_validation_cannot_produce_successful_diagnostic_result(validation) -> None:
    invalid = validate(pd.DataFrame({"source": ["S1"], "unit": ["U1"], "group": [None]}), config=validation.config)
    assert not invalid.valid
    with pytest.raises(MetadataValidationError) as caught:
        result_for(invalid)
    assert caught.value.report is invalid


def test_audit_columns_cannot_be_lost_through_record_export(validation) -> None:
    with pytest.raises(ValueError, match="unique string column"):
        result_for(validation, audit=pd.DataFrame([[0, 1]], columns=["position", "position"]))
    with pytest.raises(ValueError, match="unique string column"):
        result_for(validation, audit=pd.DataFrame([[0]], columns=[("record", "position")]))


def test_computation_error_preserves_stage_validation_and_original_cause(validation) -> None:
    original = ArithmeticError("synthetic numerical failure")
    with pytest.raises(DiagnosticComputationError) as caught:
        try:
            raise original
        except ArithmeticError as error:
            raise DiagnosticComputationError("entropy", "computation failed", validation=validation) from error
    assert caught.value.stage == "entropy"
    assert caught.value.validation is validation
    assert caught.value.__cause__ is original
    assert str(caught.value) == "entropy: computation failed"
