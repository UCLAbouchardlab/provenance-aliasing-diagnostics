"""Regressions for input integrity and structured identifiers during migration."""
from __future__ import annotations

from dataclasses import replace
import warnings

import numpy as np
import pandas as pd
import pytest

from provenance_aliasing.incidence import Corpus
from provenance_aliasing.structure import structural_leave_one_source_out


def weighted_frame(weights: list[object]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source": ["S1", "S2"],
            "unit": ["U1", "U2"],
            "group": ["A", "B"],
            "weight": weights,
        }
    )


@pytest.mark.parametrize(
    "bad_weight",
    [np.inf, -np.inf, np.nan, "inf", "-inf", complex(2, np.inf), 2 + 3j, 2 + 0j],
)
@pytest.mark.parametrize("direct_constructor", [False, True])
def test_constructor_rejects_nonfinite_weights(bad_weight, direct_constructor: bool) -> None:
    frame = weighted_frame([1.0, bad_weight])
    with pytest.raises(ValueError, match="finite"):
        if direct_constructor:
            Corpus(frame=frame)
        else:
            Corpus.from_frame(frame, weight="weight")


def test_individually_finite_weights_with_overflowing_total_raise_clean_error() -> None:
    frame = weighted_frame([1e308, 1e308])
    assert np.isfinite(frame["weight"]).all()
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with pytest.raises(ValueError, match="sum must be finite"):
            Corpus.from_frame(frame, weight="weight")


@pytest.mark.parametrize("weights", [[1.0, np.inf], [1e308, 1e308]])
def test_lineage_constructor_does_not_bypass_weight_validation(weights) -> None:
    corpus = Corpus.from_frame(weighted_frame([1.0, 1.0]), weight="weight")
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with pytest.raises(ValueError, match="finite"):
            replace(corpus, frame=weighted_frame(weights))


@pytest.mark.parametrize("excluded_weight", [np.inf, -np.inf, np.nan, "invalid", 1e308])
def test_missing_key_exclusions_keep_the_existing_weight_policy(excluded_weight) -> None:
    frame = weighted_frame([2.0, excluded_weight])
    frame.loc[1, "unit"] = None
    corpus = Corpus.from_frame(frame, weight="weight")
    assert corpus.n_rows == 1
    assert corpus.n_records == 2
    assert corpus.total_weight == 2.0
    pd.testing.assert_frame_equal(corpus.records, frame)
    with pytest.raises(ValueError, match="missing source/unit/group"):
        Corpus.from_frame(frame, weight="weight", dropna=False)


@pytest.mark.parametrize("weights", [[1.0, np.inf], [1e308, 1e308]])
def test_parent_projection_revalidates_newly_eligible_weights(weights) -> None:
    frame = weighted_frame(weights)
    frame.loc[1, "unit"] = None
    frame["donor"] = ["D1", "D2"]
    corpus = Corpus.from_frame(frame, weight="weight", hierarchy={"donor": "donor"})
    assert corpus.n_rows == 1
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with pytest.raises(ValueError, match="finite"):
            corpus.at_unit("donor")


@pytest.mark.parametrize("weights", [[0.0, 2.0], [2.0, 3.0], [1e100, 2e100]])
def test_finite_weights_and_parent_projection_remain_supported(weights) -> None:
    frame = weighted_frame(weights)
    frame["donor"] = ["D1", "D2"]
    corpus = Corpus.from_frame(frame, weight="weight", hierarchy={"donor": "donor"})
    parent = corpus.at_unit("donor")
    assert corpus.total_weight == pytest.approx(sum(weights))
    assert parent.total_weight == pytest.approx(sum(weights))
    assert parent.units == ["D1", "D2"]
    np.testing.assert_allclose(corpus.source_group_weight, parent.source_group_weight)


def test_structural_removal_preserves_composite_unit_identities_and_counts() -> None:
    frame = pd.DataFrame(
        {
            "source": ["S1", "S1", "S2", "S2", "S3", "S2"],
            "dataset": ["D1", "D1", "D2", "D1", "D1", "D2"],
            "sample": ["sample-1", "shared", "sample-1", "shared", "shared", "sample, 2"],
            "group": ["A", "B", "A", "B", "B", "B"],
        }
    )
    corpus = Corpus.from_frame(frame, unit=["dataset", "sample"])
    result = structural_leave_one_source_out(corpus)
    assert result.at["S1", "emptied_unit_keys"] == (("D1", "sample-1"),)
    assert result.at["S2", "emptied_unit_keys"] == (
        ("D2", "sample, 2"),
        ("D2", "sample-1"),
    )
    assert result.at["S3", "emptied_unit_keys"] == ()
    assert result.at["S3", "emptied_units"] == ""
    assert result.at["S1", "emptied_units"] == "('D1', 'sample-1')"
    assert result["emptied_total"].to_dict() == {"S2": 2, "S1": 1, "S3": 0}
    assert result["units_covered"].to_dict() == {"S2": 3, "S1": 2, "S3": 1}
    for keys, count in zip(result["emptied_unit_keys"], result["emptied_total"]):
        assert isinstance(keys, tuple)
        assert len(keys) == count
        assert set(keys) <= set(corpus.units)


def test_structural_removal_preserves_scalar_display_compatibility() -> None:
    corpus = Corpus.from_records(
        [
            {"source": "S1", "unit": "U1", "group": "A"},
            {"source": "S1", "unit": "U2", "group": "A"},
            {"source": "S1", "unit": "U3", "group": "B"},
            {"source": "S2", "unit": "U3", "group": "B"},
        ]
    )
    result = structural_leave_one_source_out(corpus)
    assert result.at["S1", "emptied_units"] == "U1, U2"
    assert result.at["S1", "emptied_unit_keys"] == ("U1", "U2")
    assert result.at["S2", "emptied_units"] == ""
    assert result.at["S2", "emptied_unit_keys"] == ()
