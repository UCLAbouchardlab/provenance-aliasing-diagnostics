from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from provenance_aliasing.ceiling import balanced_accuracy_ceiling, ceiling_sensitivity
from provenance_aliasing.entropy import bootstrap_ratio, label_entropy, leave_one_source_out, shannon
from provenance_aliasing.incidence import Corpus
from provenance_aliasing.structure import (
    concentration,
    design_summary,
    pair_sharing_by_group,
    shared_source_matrix,
    spanning_source_table,
    structural_leave_one_source_out,
)
from conftest import ANCHORS, HUB


def test_shannon_contract() -> None:
    assert shannon([1, 1]) == pytest.approx(1.0)
    assert shannon([1, 0]) == pytest.approx(0.0)
    assert np.isnan(shannon([]))
    with pytest.raises(ValueError, match="negative"):
        shannon([1, -1])


def test_worked_entropy_matches_gold_anchor(worked_corpus: Corpus) -> None:
    result = label_entropy(worked_corpus)
    assert result.H_G == pytest.approx(ANCHORS["H_G"])
    assert result.H_G_given_D == pytest.approx(ANCHORS["H_G_given_D"])
    assert result.ratio == pytest.approx(ANCHORS["ratio"])
    assert result.n_sources == ANCHORS["n_sources"]
    assert result.n_spanning_sources == ANCHORS["n_spanning"]
    assert result.corpus == "worked"
    assert result.weighting == "incidence rows"


def test_entropy_extremes(crossed_corpus: Corpus, nested_corpus: Corpus) -> None:
    assert label_entropy(crossed_corpus).ratio == pytest.approx(1.0)
    assert label_entropy(nested_corpus).ratio == pytest.approx(0.0)


def test_weighting_changes_entropy(worked_corpus: Corpus, weighted_corpus: Corpus) -> None:
    assert label_entropy(worked_corpus).ratio != pytest.approx(label_entropy(weighted_corpus).ratio)


def test_leave_one_out_covers_every_source(worked_corpus: Corpus) -> None:
    loso = leave_one_source_out(worked_corpus)
    assert HUB in loso.index
    assert len(loso) == worked_corpus.n_sources


def test_bootstrap_is_deterministic(worked_corpus: Corpus) -> None:
    first = bootstrap_ratio(worked_corpus, B=40, seed=17)
    second = bootstrap_ratio(worked_corpus, B=40, seed=17)
    np.testing.assert_array_equal(first.values, second.values)
    assert first.observed == pytest.approx(ANCHORS["ratio"])


def test_ceiling_contract_on_extreme_designs(crossed_corpus: Corpus, nested_corpus: Corpus) -> None:
    crossed = balanced_accuracy_ceiling(crossed_corpus)
    nested = balanced_accuracy_ceiling(nested_corpus)
    assert crossed.exact == pytest.approx(1.0)
    assert nested.exact == pytest.approx(0.5)
    assert crossed.relaxation >= crossed.exact
    assert nested.chance == pytest.approx(0.5)


def test_ceiling_rejects_nonbinary_and_unknown_positive_group(worked_corpus: Corpus) -> None:
    three = Corpus.from_records(
        [{"source": f"S{i}", "unit": f"U{i}", "group": g} for i, g in enumerate("ABC")]
    )
    with pytest.raises(ValueError, match="two-group"):
        balanced_accuracy_ceiling(three)
    with pytest.raises(ValueError, match="positive"):
        balanced_accuracy_ceiling(worked_corpus, positive_group="missing")


def test_ceiling_sensitivity_is_deterministic(worked_corpus: Corpus) -> None:
    a = ceiling_sensitivity(worked_corpus, HUB, [1, 10, 100])
    b = ceiling_sensitivity(worked_corpus, HUB, [1, 10, 100])
    pd.testing.assert_frame_equal(a, b)
    assert len(a) == 3


def test_shared_source_matrix_and_pair_rates(worked_corpus: Corpus) -> None:
    matrix = shared_source_matrix(worked_corpus)
    pd.testing.assert_frame_equal(matrix, matrix.T)
    assert (np.diag(matrix) == worked_corpus.sources_per_unit().to_numpy()).all()
    rates = pair_sharing_by_group(worked_corpus)
    assert set(rates.index) == set(worked_corpus.groups)
    assert rates["pair_sharing_rate"].between(0, 1).all()


def test_structural_tables_cover_every_source(worked_corpus: Corpus) -> None:
    loso = structural_leave_one_source_out(worked_corpus)
    assert set(loso.index) == set(worked_corpus.sources)
    assert loso.loc[HUB, "emptied_total"] > 0
    spanning = spanning_source_table(worked_corpus)
    assert set(spanning.index) == {"SRC-SPAN-A", "SRC-SPAN-B", "SRC-SPAN-C"}


def test_concentration_and_design_summary_are_self_consistent(worked_corpus: Corpus) -> None:
    conc = concentration(worked_corpus)
    assert conc["top_source"] == HUB
    assert 0 <= conc["pair_sharing_rate"] <= 1
    summary = design_summary(worked_corpus)
    assert summary["n_rows"] == worked_corpus.n_rows
    assert summary["top_source"] == HUB


def test_pair_sharing_rejects_group_impure_units() -> None:
    corpus = Corpus.from_records(
        [
            {"source": "S1", "unit": "U1", "group": "A"},
            {"source": "S2", "unit": "U1", "group": "A"},
        ]
    )
    # Corpus construction now rejects this inconsistency early. Retain the
    # downstream guard as protection against external mutation of ``frame``.
    corpus.frame.loc[1, "group"] = "B"
    with pytest.raises(ValueError, match="exactly one group"):
        pair_sharing_by_group(corpus)


def test_composite_sources_flow_through_entropy_ceiling_and_structure() -> None:
    frame = pd.DataFrame(
        {
            "laboratory": ["L1", "L1", "L2", "L2"],
            "batch": ["B1", "B1", "B2", "B2"],
            "unit": ["U1", "U2", "U3", "U4"],
            "group": ["A", "B", "A", "B"],
        }
    )
    corpus = Corpus.from_frame(
        frame,
        source=["laboratory", "batch"],
        unit="unit",
        group="group",
        duplicates="keep",
        name="composite-source",
    )
    source = ("L1", "B1")

    entropy = label_entropy(corpus)
    assert entropy.ratio == pytest.approx(1.0)
    assert set(entropy.per_source.index) == set(corpus.sources)

    loso = leave_one_source_out(corpus)
    assert set(loso.index) == set(corpus.sources)
    assert loso["original_contribution"].notna().all()

    ceiling = balanced_accuracy_ceiling(corpus)
    assert ceiling.exact == pytest.approx(1.0)
    sensitivity = ceiling_sensitivity(corpus, source, [1.0, 4.0])
    np.testing.assert_allclose(sensitivity["exact"].to_numpy(), 1.0)

    concentrated = concentration(corpus)
    assert concentrated["top_source"] in {str(key) for key in corpus.sources}
    assert concentrated["top_source_units"] == 2
    summary = design_summary(corpus)
    assert summary["top_source"] in {str(key) for key in corpus.sources}


def test_composite_groups_flow_through_structural_summaries() -> None:
    frame = pd.DataFrame(
        {
            "source": ["S1", "S1", "S2", "S2"],
            "unit": ["U1", "U2", "U3", "U4"],
            "condition": ["case", "control", "case", "control"],
            "state": ["primary", "baseline", "primary", "baseline"],
        }
    )
    corpus = Corpus.from_frame(
        frame,
        source="source",
        unit="unit",
        group=["condition", "state"],
        duplicates="keep",
        name="composite-group",
    )

    rates = pair_sharing_by_group(corpus)
    assert set(rates.index) == set(corpus.groups)

    summary = design_summary(corpus)
    for group in corpus.groups:
        assert summary[f"n_units[{group}]"] == 2
        assert summary[f"n_pairs[{group}]"] == 1
        assert np.isfinite(summary[f"worst_removal_fraction[{group}]"])
