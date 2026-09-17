from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from provenance_aliasing.calibration import (
    KBreakEven,
    arbitrary_split_null,
    dispersion_gap,
    dispersion_gap_batch,
    k_diagnostic,
    median_split_indices,
    n_partitions,
    named_split,
)
from provenance_aliasing.metrics import METRICS, aitchison, aitchison_pseudocount_sweep, metric_agreement, pearson


@pytest.mark.parametrize("name", sorted(METRICS))
def test_metric_matrix_contract(name: str, profiles: pd.DataFrame) -> None:
    result = METRICS[name](profiles)
    for matrix in (result.distance, result.similarity):
        assert matrix.shape == (len(profiles), len(profiles))
        np.testing.assert_allclose(matrix, matrix.T, atol=1e-12)
        assert np.isfinite(matrix).all()
    np.testing.assert_allclose(np.diag(result.distance), 0, atol=1e-12)
    assert result.name == name


@pytest.mark.parametrize("name", sorted(METRICS))
def test_dataframe_and_array_metrics_agree(name: str, profiles: pd.DataFrame) -> None:
    a = METRICS[name](profiles)
    b = METRICS[name](profiles.to_numpy())
    np.testing.assert_allclose(a.distance, b.distance)
    np.testing.assert_allclose(a.similarity, b.similarity)


def test_pearson_similarity_is_correlation(profiles: pd.DataFrame) -> None:
    result = pearson(profiles)
    np.testing.assert_allclose(result.similarity, np.corrcoef(profiles.to_numpy()))


def test_compositional_metrics_reject_invalid_simplex() -> None:
    invalid = np.array([[0.5, 0.5], [1.2, -0.2]])
    for name in ("fisher_rao", "hellinger", "jensen_shannon"):
        with pytest.raises(ValueError):
            METRICS[name](invalid)


def test_aitchison_requires_declared_pseudocount_for_zeros() -> None:
    profiles = np.array([[0.0, 0.5, 0.5], [0.2, 0.3, 0.5]])
    with pytest.raises(ValueError, match="zero"):
        aitchison(profiles)
    assert np.isfinite(aitchison(profiles, eps=1e-6).distance).all()


def test_metric_agreement_is_symmetric_and_bounded(profiles: pd.DataFrame) -> None:
    result = metric_agreement(profiles)
    pd.testing.assert_frame_equal(result, result.T)
    assert np.nanmin(result.to_numpy()) >= -1
    assert np.nanmax(result.to_numpy()) <= 1


def test_pseudocount_sweep_is_deterministic() -> None:
    profiles = np.array([[0.0, 0.5, 0.5], [0.2, 0.3, 0.5], [0.1, 0.7, 0.2]])
    scorer = lambda result: float(result.distance.max())
    a = aitchison_pseudocount_sweep(profiles, [1e-6, 1e-3], scorer)
    b = aitchison_pseudocount_sweep(profiles, [1e-6, 1e-3], scorer)
    pd.testing.assert_frame_equal(a, b)


def test_dispersion_gap_known_value_and_batch_agreement(similarity: np.ndarray) -> None:
    gap = dispersion_gap(similarity, [0, 1], [2, 3])
    assert gap == pytest.approx(0.2)
    masks = np.array([[True, True, False, False], [True, False, True, False]])
    batch = dispersion_gap_batch(similarity, masks)
    assert batch[0] == pytest.approx(gap)


def test_split_validation(similarity: np.ndarray) -> None:
    with pytest.raises(ValueError, match="overlap"):
        dispersion_gap(similarity, [0, 1], [1, 2])
    assert n_partitions(4, 2) == 3
    assert named_split(similarity, {"a": [0, 1], "b": [2, 3]}, ["a", "b"]) == pytest.approx(0.2)


def test_exhaustive_null_and_sampled_null_are_reproducible(similarity: np.ndarray) -> None:
    exact = arbitrary_split_null(similarity, 2, exhaustive=True, keep_values=True)
    assert exact.exhaustive
    assert exact.n_evaluated == n_partitions(4, 2)
    a = arbitrary_split_null(similarity, 2, exhaustive=False, B=3, seed=8, keep_values=True)
    b = arbitrary_split_null(similarity, 2, exhaustive=False, B=3, seed=8, keep_values=True)
    np.testing.assert_array_equal(a.values, b.values)


def test_null_p_value_and_quantiles_are_bounded(similarity: np.ndarray) -> None:
    null = arbitrary_split_null(similarity, 2, exhaustive=True, observed=[0.1], keep_values=True)
    assert 0 <= null.p_value(0.1) <= 1
    bin_width = float(np.diff(null.hist_edges).max())
    assert null.quantile(0.5) <= null.maximum + bin_width


def test_k_diagnostic_and_median_split_contract(similarity: np.ndarray) -> None:
    break_even = KBreakEven(1.0, 1.0, 2, 2, 2, 2, derivation="test fixture")
    result = k_diagnostic(similarity, [0, 1], [2, 3], break_even=break_even, basis="synthetic")
    assert result.verdict in {"conservative", "anti-conservative", "undetermined"}
    assert result.basis == "synthetic"
    variants = median_split_indices([1, 2, 2, 3], n_a=2)
    assert variants
    assert all(len(a) == len(b) == 2 for a, b in variants)
