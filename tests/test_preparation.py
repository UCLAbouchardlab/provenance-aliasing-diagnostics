"""Preparation preserves the assessed projection and every original row's lineage."""
from __future__ import annotations

from decimal import Decimal
import math
import warnings

import numpy as np
import pandas as pd
import pytest

from provenance_aliasing.api import AnalysisConfig, validate
from provenance_aliasing.api.validation import _assess
from provenance_aliasing.incidence import Corpus


AUDIT_COLUMNS = [
    "row_position", "status", "reason", "representative_row_position",
    "projection_row_position", "normalized_columns",
]


def config_for(**overrides) -> AnalysisConfig:
    values = {
        "schema_version": "0.1-draft",
        "name": "synthetic-preparation",
        "columns": {"source": ["source"], "unit": ["unit"], "group": ["group"]},
        "grain": {"name": "study", "description": "Explicit synthetic study"},
        "unit_description": "Explicit synthetic observation",
    }
    values.update(overrides)
    return AnalysisConfig.from_dict(values)


def weighted_config(**overrides) -> AnalysisConfig:
    return config_for(
        weighting={"mode": "column", "column": "mass", "description": "Declared row mass"},
        **overrides,
    )


def build_projected_corpus(assessment, config) -> Corpus:
    """Exercise only the prepared table through the installed reference core."""
    return Corpus.from_frame(
        assessment.projection, source="source", unit="unit", group="group", weight="weight",
        hierarchy={level: f"hierarchy::{level}" for level in config.hierarchy},
        dropna=False, duplicates="keep" if config.policies.duplicates == "keep_records" else "raise",
    )


def test_drop_collapse_and_normalization_have_exact_original_row_lineage() -> None:
    frame = pd.DataFrame({
        "source": [" MISSING ", " S1 ", "S1", "S2", "S2"],
        "unit": [("D0", "bad"), (" D1 ", "001"), ("D1", "001"), ("D2", "002"), ("D2", "003")],
        "group": ["A", " A ", "A", "B", "B"],
        "donor": [["invalid but excluded"], "D1", "D1", "MISSING", None],
        "mass": [np.inf, Decimal("1.25"), "1.25", "2.75", 0.0],
        "note": ["excluded", "same", "same", "second", "third"],
    }, dtype=object, index=[99, 5, 5, 999, "arbitrary"])
    original = frame.copy(deep=True)
    config = weighted_config(
        hierarchy={"donor": ["donor"]},
        policies={"missing_required": "drop", "duplicates": "collapse_identical",
                  "missing_tokens": ["MISSING"]},
    )

    assessment = _assess(frame, config=config)
    assert assessment.report.valid, assessment.report.to_json()
    assert assessment.report.to_dict() == validate(frame, config=config).to_dict()
    pd.testing.assert_frame_equal(frame, original)
    pd.testing.assert_frame_equal(assessment.projection, pd.DataFrame({
        "source": ["S1", "S2", "S2"],
        "unit": [("D1", "001"), ("D2", "002"), ("D2", "003")],
        "group": ["A", "B", "B"], "weight": [1.25, 2.75, 0.0],
        "hierarchy::donor": ["D1", None, None],
    }))
    assert list(assessment.audit.columns) == AUDIT_COLUMNS
    assert isinstance(assessment.audit.index, pd.RangeIndex)
    assert assessment.audit["row_position"].tolist() == list(range(5))
    assert assessment.audit["status"].tolist() == ["excluded", "retained", "collapsed", "retained", "retained"]
    assert assessment.audit["reason"].tolist() == ["missing_required_key", "", "identical_duplicate", "", ""]
    assert assessment.audit["representative_row_position"].tolist() == [None, 1, 1, 3, 4]
    assert assessment.audit["projection_row_position"].tolist() == [None, 0, 0, 1, 2]
    assert assessment.audit["normalized_columns"].tolist() == [
        ("source",), ("source", "unit", "group", "mass"), ("mass",), ("donor", "mass"), (),
    ]
    corpus = build_projected_corpus(assessment, config)
    assert corpus.n_rows == assessment.report.counts["n_retained"] == 3
    assert corpus.total_weight == assessment.report.counts["total_weight"] == 4.0


@pytest.mark.parametrize("policy,expected_retained,expected_status", [
    ("collapse_identical", 2, ["retained", "retained", "collapsed", "collapsed"]),
    ("keep_records", 4, ["retained", "retained", "retained", "retained"]),
])
def test_interleaved_duplicates_preserve_order_and_resolved_policy(policy, expected_retained, expected_status) -> None:
    frame = pd.DataFrame({"source": ["S1", "S2", "S1", "S2"],
                          "unit": ["U1", "U2", "U1", "U2"], "group": ["A", "B", "A", "B"]},
                         index=[4, 4, 1, 1])
    config = config_for(policies={"duplicates": policy})
    assessment = _assess(frame, config=config)
    assert assessment.report.valid, assessment.report.to_json()
    assert assessment.audit["status"].tolist() == expected_status
    expected_representatives = [0, 1, 0, 1] if policy == "collapse_identical" else list(range(4))
    assert assessment.audit["representative_row_position"].tolist() == expected_representatives
    assert assessment.audit["projection_row_position"].tolist() == expected_representatives
    assert len(assessment.projection) == expected_retained
    assert isinstance(assessment.projection.index, pd.RangeIndex)
    assert assessment.projection["weight"].tolist() == [1.0] * expected_retained
    corpus = build_projected_corpus(assessment, config)
    assert corpus.n_rows == expected_retained
    assert corpus.total_weight == assessment.report.counts["total_weight"] == float(expected_retained)


def test_explicit_composite_columns_preserve_component_boundaries_and_parent_keys() -> None:
    frame = pd.DataFrame({"study": ["a,b", "a"], "batch": ["1", "1"],
                          "sample": ["c", "b,c"], "group": ["A", "B"], "donor": ["01", "01"]})
    config = config_for(
        columns={"source": ["study", "batch"], "unit": ["study", "sample"], "group": ["group"]},
        hierarchy={"donor": ["study", "donor"]},
    )
    assessment = _assess(frame, config=config)
    assert assessment.report.valid, assessment.report.to_json()
    assert assessment.projection["source"].tolist() == [("a,b", "1"), ("a", "1")]
    assert assessment.projection["unit"].tolist() == [("a,b", "c"), ("a", "b,c")]
    assert assessment.projection["hierarchy::donor"].tolist() == [("a,b", "01"), ("a", "01")]
    corpus = build_projected_corpus(assessment, config)
    assert corpus.n_units == corpus.n_sources == 2
    assert corpus.unit_spec == ("unit",)


def test_single_column_tuple_group_and_parent_keys_reach_core_without_stringification() -> None:
    frame = pd.DataFrame({"source": [("D1", "run1"), ("D2", "run1")],
                          "unit": [("D1", "001"), ("D2", "001")],
                          "group": [("A", "baseline"), ("B", "baseline")],
                          "donor": [("D1", "p1"), ("D2", None)]})
    config = config_for(hierarchy={"donor": ["donor"]})
    assessment = _assess(frame, config=config)
    assert assessment.report.valid, assessment.report.to_json()
    assert assessment.projection["hierarchy::donor"].tolist() == [("D1", "p1"), None]
    assert assessment.audit.at[1, "normalized_columns"] == ("donor",)
    corpus = build_projected_corpus(assessment, config)
    assert corpus.groups == [("A", "baseline"), ("B", "baseline")]
    assert corpus.units == [("D1", "001"), ("D2", "001")]


def test_decimal_weights_and_numeric_keys_have_a_single_canonical_preparation() -> None:
    frame = pd.DataFrame({"source": [Decimal("1"), Decimal("2")], "unit": [1, "001"],
                          "group": [0, 1], "mass": [Decimal("1.25"), Decimal("2.75")]}, dtype=object)
    config = weighted_config()
    assessment = _assess(frame, config=config)
    assert assessment.report.valid, assessment.report.to_json()
    assert assessment.projection.to_dict("list") == {
        "source": ["1", "2"], "unit": ["1", "001"], "group": ["0", "1"], "weight": [1.25, 2.75],
    }
    assert all(type(value) is float for value in assessment.projection["weight"].tolist())
    corpus = build_projected_corpus(assessment, config)
    assert corpus.total_weight == assessment.report.counts["total_weight"] == 4.0
    assert corpus.n_units == 2


def test_untrimmed_declared_missing_tokens_and_literal_na_have_exact_audit() -> None:
    frame = pd.DataFrame({"source": ["missing", "NA", "S2"], "unit": ["U0", "001", "1"],
                          "group": ["A", "A", "B"], "parent": ["ignored", "missing", " missing "]})
    config = config_for(hierarchy={"donor": ["parent"]}, policies={
        "missing_required": "drop", "missing_tokens": ["missing"],
    })
    assessment = _assess(frame, config=config)
    assert assessment.report.valid, assessment.report.to_json()
    assert assessment.projection["source"].tolist() == ["NA", "S2"]
    assert assessment.projection["unit"].tolist() == ["001", "1"]
    assert assessment.projection["hierarchy::donor"].tolist() == [None, None]
    assert assessment.audit["normalized_columns"].tolist() == [("source",), ("parent",), ("parent",)]
    assert assessment.audit.at[0, "representative_row_position"] is None
    assert assessment.audit.at[0, "projection_row_position"] is None
    assert build_projected_corpus(assessment, config).n_rows == 2


def test_collapsing_huge_identical_weights_prepares_finite_mass_before_core_construction() -> None:
    frame = pd.DataFrame({"source": ["S1", "S1", "S2"], "unit": ["U1", "U1", "U2"],
                          "group": ["A", "A", "B"], "mass": [Decimal("1e308"), "1e308", 0.0]}, dtype=object)
    config = weighted_config(policies={"duplicates": "collapse_identical"})
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        assessment = _assess(frame, config=config)
        assert assessment.report.valid, assessment.report.to_json()
        corpus = build_projected_corpus(assessment, config)
    assert math.isfinite(corpus.total_weight)
    assert corpus.total_weight == assessment.report.counts["total_weight"] == 1e308
    assert assessment.audit["status"].tolist() == ["retained", "collapsed", "retained"]
    assert assessment.audit["projection_row_position"].tolist() == [0, 0, 1]
    assert assessment.projection["weight"].tolist() == [1e308, 0.0]


@pytest.mark.parametrize("problem", ["group", "parent", "metadata", "weight", "nested_key"])
def test_invalid_conflicts_never_produce_a_usable_preparation(problem) -> None:
    frame = pd.DataFrame({"source": ["S1", "S1"], "unit": ["U1", "U1"],
                          "group": ["A", "A"], "parent": ["P1", "P1"],
                          "mass": [1.0, 1.0], "note": ["same", "same"]}, dtype=object)
    if problem == "group":
        frame.at[1, "group"] = "B"
    elif problem == "parent":
        frame.at[1, "parent"] = "P2"
    elif problem == "metadata":
        frame.at[1, "note"] = "different"
    elif problem == "weight":
        frame.at[1, "mass"] = Decimal("sNaN")
    else:
        frame.at[1, "unit"] = (("nested", "tuple"), "unit")
    config = weighted_config(hierarchy={"donor": ["parent"]}, policies={"duplicates": "collapse_identical"})
    assessment = _assess(frame, config=config)
    assert not assessment.report.valid
    assert assessment.report.to_dict() == validate(frame, config=config).to_dict()
    assert assessment.projection.empty
    assert list(assessment.projection.columns) == ["source", "unit", "group", "weight", "hierarchy::donor"]
    assert assessment.audit.empty
    assert list(assessment.audit.columns) == AUDIT_COLUMNS


def test_dropped_required_keys_cannot_make_bad_parent_or_weight_values_reappear() -> None:
    frame = pd.DataFrame({"source": ["S1", "S2", "S3"], "unit": [None, "U2", "U3"],
                          "group": ["A", "A", "B"], "parent": [["unusable"], "P2", "P3"],
                          "mass": [1 + 2j, "2.5", Decimal("1.5")]}, dtype=object)
    config = weighted_config(hierarchy={"donor": ["parent"]}, policies={"missing_required": "drop"})
    assessment = _assess(frame, config=config)
    assert assessment.report.valid, assessment.report.to_json()
    assert assessment.projection["weight"].tolist() == [2.5, 1.5]
    assert assessment.projection["hierarchy::donor"].tolist() == ["P2", "P3"]
    assert assessment.audit.at[0, "normalized_columns"] == ()
    assert assessment.audit.at[0, "status"] == "excluded"
    assert build_projected_corpus(assessment, config).total_weight == 4.0


@pytest.mark.parametrize("problem", ["missing_column", "duplicate_column", "empty"])
def test_schema_failures_return_empty_typed_preparation_tables(problem) -> None:
    frame = pd.DataFrame({"source": ["S1"], "unit": ["U1"], "group": ["A"]})
    if problem == "missing_column":
        frame = frame.drop(columns="unit")
    elif problem == "duplicate_column":
        frame = pd.concat([frame, frame[["unit"]]], axis=1)
    else:
        frame = frame.iloc[:0]
    assessment = _assess(frame, config=config_for())
    assert not assessment.report.valid
    assert assessment.projection.empty and assessment.audit.empty
    assert list(assessment.projection.columns) == ["source", "unit", "group", "weight"]
    assert list(assessment.audit.columns) == AUDIT_COLUMNS
    assert assessment.projection["weight"].dtype == np.dtype(float)


def test_preparation_products_do_not_alias_the_input_or_validation_report() -> None:
    frame = pd.DataFrame({"source": [" S1 ", "S2"], "unit": ["U1", "U2"], "group": ["A", "B"]})
    original = frame.copy(deep=True)
    assessment = _assess(frame, config=config_for())
    before = assessment.report.to_dict()
    assessment.projection.at[0, "source"] = "changed"
    assessment.audit.at[0, "status"] = "changed"
    assert assessment.report.to_dict() == before
    pd.testing.assert_frame_equal(frame, original)
