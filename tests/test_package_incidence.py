from __future__ import annotations

import importlib
from importlib.metadata import version
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import provenance_aliasing
from provenance_aliasing.incidence import Corpus, DuplicateObservationWarning


def test_package_version_matches_installed_distribution() -> None:
    assert provenance_aliasing.__version__ == version("provenance-aliasing-diagnostics")


def test_all_declared_public_exports_exist_and_resolve_locally() -> None:
    package_dir = Path(provenance_aliasing.__file__).resolve().parent
    for name in provenance_aliasing.__all__:
        value = getattr(provenance_aliasing, name)
        module = importlib.import_module(value.__module__) if hasattr(value, "__module__") else None
        if module is not None and getattr(module, "__file__", None):
            assert Path(module.__file__).resolve().is_relative_to(package_dir)


def test_unknown_package_attribute_raises() -> None:
    with pytest.raises(AttributeError, match="no attribute"):
        getattr(provenance_aliasing, "not_a_real_api")


def test_from_frame_normalizes_keys_without_mutating_input() -> None:
    frame = pd.DataFrame(
        {"deposit": [" S1 ", "S2"], "sample": [" U1 ", "U2"], "label": ["A", "B"]}
    )
    original = frame.copy(deep=True)
    corpus = Corpus.from_frame(frame, source="deposit", unit="sample", group="label")
    pd.testing.assert_frame_equal(frame, original)
    assert corpus.sources == ["S1", "S2"]
    assert corpus.units == ["U1", "U2"]
    assert corpus.total_weight == 2.0
    assert corpus.weighting == "incidence rows"


def test_weight_validation_and_metadata() -> None:
    frame = pd.DataFrame(
        {"source": ["S1", "S2"], "unit": ["U1", "U2"], "group": ["A", "B"], "mass": [2, 3]}
    )
    corpus = Corpus.from_frame(frame, weight="mass", name="weighted")
    assert corpus.total_weight == 5.0
    assert corpus.weighting == "weighted by mass"
    assert corpus.describe()["name"] == "weighted"

    for bad in ([1, -1], [0, 0], [1, "text"]):
        invalid = frame.assign(mass=bad)
        with pytest.raises(ValueError):
            Corpus.from_frame(invalid, weight="mass")


def test_missing_rows_are_dropped_or_rejected() -> None:
    frame = pd.DataFrame(
        {"source": ["S1", None], "unit": ["U1", "U2"], "group": ["A", "B"]}
    )
    corpus = Corpus.from_frame(frame)
    assert corpus.n_rows == 1
    assert corpus.n_records == 2
    assert corpus.records["source"].isna().tolist() == [False, True]
    with pytest.raises(ValueError, match="missing"):
        Corpus.from_frame(frame, dropna=False)


def test_incidence_and_source_group_tables(worked_corpus: Corpus) -> None:
    incidence = worked_corpus.incidence
    weights = worked_corpus.source_group_weight
    assert incidence.shape == (worked_corpus.n_sources, worked_corpus.n_units)
    assert set(np.unique(incidence)) <= {0, 1}
    assert weights.to_numpy().sum() == worked_corpus.total_weight
    assert worked_corpus.units_per_source().max() == 30
    assert worked_corpus.sources_per_unit().min() == 1


def test_subsetting_regrouping_and_crossed_subcorpus(worked_corpus: Corpus) -> None:
    subset = worked_corpus.subset(groups=["Cancer"])
    assert subset.groups == ["Cancer"]
    assert subset.name.endswith("[subset]")

    regrouped = worked_corpus.regroup({"Cancer": "Case", "Normal": "Control"})
    assert regrouped.groups == ["Case", "Control"]
    assert worked_corpus.groups == ["Cancer", "Normal"]

    crossed = worked_corpus.crossed_subcorpus()
    assert crossed.spanning_sources() == ["SRC-SPAN-A", "SRC-SPAN-B", "SRC-SPAN-C"]


def test_construction_rejects_units_with_multiple_groups() -> None:
    with pytest.raises(ValueError, match=r"(?i)group"):
        Corpus.from_records(
            [
                {"source": "S1", "unit": "U1", "group": "A"},
                {"source": "S2", "unit": "U1", "group": "B"},
            ]
        )


def test_empty_subset_and_empty_regroup_raise(worked_corpus: Corpus) -> None:
    with pytest.raises(ValueError, match="empty"):
        worked_corpus.subset(sources=["missing"])
    with pytest.raises(ValueError, match="removed every row"):
        worked_corpus.regroup({"missing": "other"})


def test_duplicate_policies_preserve_raw_caatlas_records() -> None:
    frame = pd.DataFrame(
        {
            "originating_dataset": ["PXD000001", "PXD000001"],
            "sample_id": ["AUT-DN06_Lung", "AUT-DN06_Lung"],
            "condition": ["normal", "normal"],
            "hla_class": ["HLA-I", "HLA-II"],
            "aliquot_id": [pd.NA, "aliquot-2"],
            "run_id": ["run-I", "run-II"],
            "technical_replicate": [1, 2],
        }
    )
    kwargs = {
        "source": "originating_dataset",
        "unit": ["originating_dataset", "sample_id"],
        "group": "condition",
    }

    with pytest.warns(
        DuplicateObservationWarning, match=r"(?i)(duplicate|repeated)"
    ) as caught:
        warned = Corpus.from_frame(frame, **kwargs)
    # Dependency deprecations can also be captured on supported older versions.
    assert sum(issubclass(item.category, DuplicateObservationWarning) for item in caught) == 1
    assert warned.n_rows == 2
    assert warned.n_units == 1
    assert warned.frame["unit"].tolist() == [
        ("PXD000001", "AUT-DN06_Lung"),
        ("PXD000001", "AUT-DN06_Lung"),
    ]
    pd.testing.assert_frame_equal(warned.records, frame)

    with warnings.catch_warnings():
        warnings.simplefilter("error", DuplicateObservationWarning)
        kept = Corpus.from_frame(frame, duplicates="keep", **kwargs)
    assert kept.n_rows == 2

    dropped = Corpus.from_frame(frame, duplicates="drop", **kwargs)
    assert dropped.n_rows == 1
    pd.testing.assert_frame_equal(dropped.records, frame)

    with pytest.raises(ValueError, match=r"(?i)(duplicate|repeated)"):
        Corpus.from_frame(frame, duplicates="raise", **kwargs)
    with pytest.raises(ValueError, match="duplicates"):
        Corpus.from_frame(frame, duplicates="dedupe", **kwargs)


def test_duplicate_drop_rejects_unequal_weights() -> None:
    frame = pd.DataFrame(
        {
            "source": ["PXD1", "PXD1"],
            "unit": ["sample-1", "sample-1"],
            "group": ["normal", "normal"],
            "mass": [2.0, 3.0],
        }
    )
    with pytest.raises(ValueError, match=r"(?i)(weight|unequal)"):
        Corpus.from_frame(frame, weight="mass", duplicates="drop")

    equal = Corpus.from_frame(frame.assign(mass=2.0), weight="mass", duplicates="drop")
    assert equal.n_rows == 1
    assert equal.total_weight == 2.0
    assert len(equal.records) == 2


def test_composite_units_keep_same_sample_id_distinct_across_datasets() -> None:
    frame = pd.DataFrame(
        {
            "dataset": ["PXD1", "PXD2"],
            "sample_id": ["sample-1", "sample-1"],
            "condition": ["normal", "normal"],
        }
    )
    corpus = Corpus.from_frame(
        frame,
        source="dataset",
        unit=["dataset", "sample_id"],
        group="condition",
    )
    assert corpus.n_units == 2
    assert set(corpus.units) == {("PXD1", "sample-1"), ("PXD2", "sample-1")}


def test_hierarchy_reports_donor_nesting_and_can_rescore_at_donor() -> None:
    frame = pd.DataFrame(
        {
            "dataset": ["caAtlas", "caAtlas"],
            "sample_id": ["AUT-DN06_Lung", "AUT-DN06_Brain"],
            "donor_id": ["AUT-DN06", "AUT-DN06"],
            "tissue": ["lung", "brain"],
            "condition": ["normal", "normal"],
        }
    )
    corpus = Corpus.from_frame(
        frame,
        source="dataset",
        unit=["dataset", "sample_id"],
        group="condition",
        hierarchy={"donor": ["dataset", "donor_id"]},
    )
    assert corpus.n_units == 2

    report = corpus.hierarchy_report()
    assert "donor" in report.index
    assert report.loc["donor", "n_parents"] == 1
    assert report.loc["donor", "n_units_missing_parent"] == 0
    assert report.loc["donor", "n_multi_unit_parents"] == 1

    donor = corpus.at_unit("donor")
    assert donor.n_rows == 1
    assert donor.n_units == 1
    assert donor.units == [("caAtlas", "AUT-DN06")]
    assert donor.total_weight == 1.0
    assert len(donor.records) == 2


def test_missing_hierarchy_and_sample_run_counts_stay_missing() -> None:
    frame = pd.DataFrame(
        {
            "dataset": ["caAtlas", "caAtlas"],
            "sample_id": ["sample-known", "sample-unknown"],
            "donor_id": ["AUT-DN06", pd.NA],
            "condition": ["normal", "normal"],
            "sample_run_count": [pd.NA, pd.NA],
            "dataset_total_runs": [1274, 1274],
        }
    )
    corpus = Corpus.from_frame(
        frame,
        source="dataset",
        unit=["dataset", "sample_id"],
        group="condition",
        hierarchy={"donor": ["dataset", "donor_id"]},
    )

    assert corpus.records["donor_id"].isna().tolist() == [False, True]
    assert corpus.records["sample_run_count"].isna().all()
    assert corpus.records["dataset_total_runs"].tolist() == [1274, 1274]
    assert corpus.hierarchy_report().loc["donor", "n_units_missing_parent"] == 1


def test_conflicting_donor_mapping_fails_before_duplicate_drop() -> None:
    frame = pd.DataFrame(
        {
            "dataset": ["caAtlas", "caAtlas"],
            "sample_id": ["sample-1", "sample-1"],
            "donor_id": ["donor-1", "donor-2"],
            "condition": ["normal", "normal"],
            "run_id": ["run-1", "run-2"],
        }
    )

    with pytest.raises(ValueError, match=r"(?i)(multiple.*donor|donor.*multiple)"):
        Corpus.from_frame(
            frame,
            source="dataset",
            unit=["dataset", "sample_id"],
            group="condition",
            duplicates="drop",
            hierarchy={"donor": ["dataset", "donor_id"]},
        )


def test_partial_donor_mapping_is_preserved_and_reported() -> None:
    frame = pd.DataFrame(
        {
            "dataset": ["caAtlas", "caAtlas"],
            "sample_id": ["sample-1", "sample-1"],
            "donor_id": ["donor-1", pd.NA],
            "condition": ["normal", "normal"],
            "run_id": ["run-1", "run-2"],
        }
    )
    corpus = Corpus.from_frame(
        frame,
        source="dataset",
        unit=["dataset", "sample_id"],
        group="condition",
        duplicates="keep",
        hierarchy={"donor": ["dataset", "donor_id"]},
    )

    assert corpus.records["donor_id"].isna().tolist() == [False, True]
    report = corpus.hierarchy_report().loc["donor"]
    assert report["n_partially_mapped_units"] == 1
    assert report["n_records_missing_parent"] == 1
    assert report["n_units_missing_parent"] == 0


def test_at_unit_rejects_unequal_explicit_child_weights() -> None:
    frame = pd.DataFrame(
        {
            "dataset": ["caAtlas", "caAtlas"],
            "sample_id": ["AUT-DN06_Lung", "AUT-DN06_Brain"],
            "donor_id": ["AUT-DN06", "AUT-DN06"],
            "condition": ["normal", "normal"],
            "mass": [2.0, 3.0],
        }
    )
    corpus = Corpus.from_frame(
        frame,
        source="dataset",
        unit=["dataset", "sample_id"],
        group="condition",
        weight="mass",
        hierarchy={"donor": ["dataset", "donor_id"]},
    )

    with pytest.raises(ValueError, match=r"(?i)(unequal weights|reducer)"):
        corpus.at_unit("donor")


def test_subset_and_regroup_keep_only_contributing_raw_lineage() -> None:
    frame = pd.DataFrame(
        {
            "deposit": ["S1", "S1", "S2", "S3"],
            "sample": ["U1", "U1", "U2", "U3"],
            "label": ["A", "A", "B", "C"],
            "assay": ["a1", "a2", "b1", "c1"],
        }
    )
    corpus = Corpus.from_frame(
        frame,
        source="deposit",
        unit="sample",
        group="label",
        duplicates="keep",
    )

    subset = corpus.subset(units=["U1"])
    assert subset.records["assay"].tolist() == ["a1", "a2"]
    assert subset.frame["group"].tolist() == ["A", "A"]

    regrouped = corpus.regroup({"A": "Case", "B": "Control"})
    assert regrouped.records["assay"].tolist() == ["a1", "a2", "b1"]
    assert regrouped.records["label"].tolist() == ["A", "A", "B"]
    assert regrouped.frame["group"].tolist() == ["Case", "Case", "Control"]


def test_records_preserve_input_index_and_are_isolated_from_caller_mutation() -> None:
    frame = pd.DataFrame(
        {
            "source": ["S1", "S2", "S3"],
            "unit": ["U1", "U2", "U3"],
            "group": ["A", "A", "B"],
            "assay": ["a1", "a2", "a3"],
        },
        index=pd.Index([7, 7, 11], name="raw_row"),
    )
    expected = frame.copy(deep=True)
    corpus = Corpus.from_frame(frame)

    frame.loc[7, "assay"] = "changed"
    frame.loc[11, "source"] = "changed"

    pd.testing.assert_frame_equal(corpus.records, expected)
    assert corpus.records.index.tolist() == [7, 7, 11]
    assert corpus.records.index.name == "raw_row"


def test_with_source_after_duplicate_drop_rechecks_all_raw_records() -> None:
    frame = pd.DataFrame(
        {
            "source": ["S1", "S1", "S2"],
            "unit": ["U1", "U1", "U1"],
            "group": ["A", "A", "A"],
            "run_id": ["run-1", "run-2", "run-3"],
        }
    )
    corpus = Corpus.from_frame(frame, duplicates="drop")
    assert corpus.n_rows == 2
    assert corpus.n_records == 3

    rescored = corpus.with_source(pd.DataFrame({"lab": ["LAB", "LAB"]}), "lab")

    assert rescored.n_rows == 1
    assert rescored.n_records == 3
    assert rescored.sources == ["LAB"]
    assert rescored.n_duplicate_keys == 1
    assert rescored.n_duplicate_rows == 2
    assert rescored.duplicate_report.loc[0, "n_records"] == 3
