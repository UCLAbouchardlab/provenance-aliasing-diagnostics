from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from provenance_aliasing.grain import declare_grain, grain_sweep, score_grains
from provenance_aliasing.guards import cramers_v, design_rank_deficient, guard_report, min_batch_size, theils_u
from provenance_aliasing.incidence import DuplicateObservationWarning
from provenance_aliasing.report import diagnose


def test_score_grains_reports_each_declared_grain(worked_frame: pd.DataFrame) -> None:
    with pytest.warns(DuplicateObservationWarning):
        result = score_grains(
            worked_frame,
            {"deposit": "source", "lab": "laboratory"},
            unit="unit",
            group="group",
            name="worked",
        )
    assert set(result["grain"]) == {"deposit", "lab"}
    assert set(result["column"]) == {"source", "laboratory"}
    assert result["ratio"].between(0, 1).all()


def test_grain_sweep_preserves_the_base_corpus(worked_corpus, worked_frame: pd.DataFrame) -> None:
    before = worked_corpus.frame.copy(deep=True)
    with pytest.warns(DuplicateObservationWarning):
        result = grain_sweep(worked_corpus, worked_frame, ["source", "laboratory"])
    pd.testing.assert_frame_equal(worked_corpus.frame, before)
    assert len(result) == 2


def test_grain_declaration_names_weighting_and_source_column(worked_frame: pd.DataFrame) -> None:
    row = score_grains(worked_frame, ["source"], unit="unit", group="group").iloc[0]
    declaration = declare_grain(row)
    assert "source" in declaration
    assert "incidence" in declaration.lower()


def test_grain_validation_rejects_unknown_columns(worked_frame: pd.DataFrame) -> None:
    with pytest.raises(KeyError, match="missing"):
        score_grains(worked_frame, ["missing"], unit="unit", group="group")


def test_guard_scalar_metrics_have_expected_extremes(crossed_corpus, nested_corpus) -> None:
    for fn in (cramers_v, theils_u):
        assert 0 <= fn(crossed_corpus) <= 1
        assert 0 <= fn(nested_corpus) <= 1
        assert fn(nested_corpus) >= fn(crossed_corpus)


def test_batch_sizes_and_rank_guard(worked_corpus) -> None:
    sizes = min_batch_size(worked_corpus)
    assert set(sizes.index) == set(worked_corpus.sources)
    assert sizes.max() == 30
    rank = design_rank_deficient(worked_corpus)
    assert {"deficient", "rank", "n_columns"} <= set(rank)
    assert isinstance(rank["deficient"], (bool, np.bool_))


def test_guard_report_schema_and_metadata(worked_corpus) -> None:
    report = guard_report(worked_corpus, grain="deposit")
    assert {"value", "fires", "what_it_means", "citation"} <= set(report.columns)
    assert report.attrs["corpus"] == "worked"
    assert report.attrs["grain"] == "deposit"


def test_diagnosis_renders_all_public_forms(crossed_corpus) -> None:
    result = diagnose(crossed_corpus, grain="synthetic source", bootstrap=10, seed=3)
    frame = result.to_frame()
    assert {"quantity", "value", "weighting", "grain", "note"} == set(frame.columns)
    assert result.ratio == pytest.approx(1.0)
    assert result.corpus == "crossed"
    assert "crossed" in result.summary()
    assert "synthetic source" in result.to_markdown()
    assert isinstance(result.verdict(), str) and result.verdict()
    assert result.as_dict()["corpus"] == "crossed"


def test_diagnose_strict_propagates_stage_failure(monkeypatch, crossed_corpus) -> None:
    import provenance_aliasing.report as report_module

    def fail(_corpus):
        raise RuntimeError("sentinel stage failure")

    monkeypatch.setattr(report_module, "label_entropy", fail)
    with pytest.raises(RuntimeError, match="sentinel"):
        report_module.diagnose(crossed_corpus, strict=True)


def test_diagnose_nonstrict_records_optional_stage_failure(monkeypatch, crossed_corpus) -> None:
    import provenance_aliasing.structure as structure_module

    def fail(_corpus):
        raise RuntimeError("sentinel optional failure")

    monkeypatch.setattr(structure_module, "pair_sharing_by_group", fail)
    result = diagnose(crossed_corpus, strict=False)
    assert any("sentinel optional failure" in note for note in result.notes)
