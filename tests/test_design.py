from __future__ import annotations

import pandas as pd
import pytest

from provenance_aliasing.design import (
    design_curve,
    estimator_behavior,
    minimum_viable_crossing,
    operating_characteristic,
    simulate_corpus,
    simulated_ceiling_envelope,
)


def test_simulate_corpus_delivers_requested_extremes() -> None:
    nested = simulate_corpus(20, 10, 0.0, seed=1)
    crossed = simulate_corpus(20, 10, 1.0, seed=1)
    assert len(nested.spanning_sources()) == 0
    assert len(crossed.spanning_sources()) == crossed.n_sources
    assert nested.n_units == crossed.n_units == 20


def test_simulation_is_deterministic_and_self_describing() -> None:
    a = simulate_corpus(18, 8, 0.5, seed=11)
    b = simulate_corpus(18, 8, 0.5, seed=11)
    pd.testing.assert_frame_equal(a.frame, b.frame)
    assert a.frame.attrs == b.frame.attrs
    assert "x=0.5" in a.name.lower()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_units": 1, "n_sources": 4, "crossing_fraction": 0.5},
        {"n_units": 10, "n_sources": 0, "crossing_fraction": 0.5},
        {"n_units": 10, "n_sources": 4, "crossing_fraction": -0.1},
        {"n_units": 10, "n_sources": 4, "crossing_fraction": 1.1},
    ],
)
def test_simulation_rejects_invalid_designs(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        simulate_corpus(**kwargs)


def test_design_curve_is_deterministic_and_bounded() -> None:
    kwargs = dict(n_units=20, n_sources=10, crossing_fractions=[0, 0.5, 1], n_rep=4, seed=2)
    a = design_curve(**kwargs)
    b = design_curve(**kwargs)
    pd.testing.assert_frame_equal(a, b)
    assert set(a["crossing_fraction"]) == {0, 0.5, 1}
    for column in [c for c in a if "ratio" in c or "ceiling" in c]:
        assert a[column].dropna().between(0, 1).all(), column


def test_design_curve_rejects_empty_sweep() -> None:
    with pytest.raises(ValueError, match="empty"):
        design_curve(20, 10, [], n_rep=2)


def test_minimum_viable_crossing_returns_one_row_per_grid_cell() -> None:
    result = minimum_viable_crossing(
        0.7,
        n_units_grid=[16],
        n_sources_grid=[8],
        crossing_grid=[0.0, 0.5, 1.0],
        n_rep=3,
        seed=4,
    )
    assert len(result) == 1
    assert {"n_units", "n_sources", "min_crossing_fraction"} <= set(result.columns)
    assert result.iloc[0]["n_units"] == 16


def test_simulated_envelope_is_deterministic_and_bounded() -> None:
    kwargs = dict(ratio_grid=[0.0, 0.5, 1.0], n_rep=24, seed=5, min_count=1)
    a = simulated_ceiling_envelope(**kwargs)
    b = simulated_ceiling_envelope(**kwargs)
    pd.testing.assert_frame_equal(a, b)
    for column in [c for c in a if "ceiling" in c]:
        assert a[column].dropna().between(0, 1).all(), column


def test_estimator_behavior_small_grid_is_reproducible() -> None:
    kwargs = dict(
        target_ratio=0.5,
        n_sources_grid=[6],
        n_units_grid=[12],
        n_rep=3,
        seed=6,
        reference_n_sources=12,
        reference_n_units=20,
        calibration_points=3,
        calibration_rep=2,
        reference_rep=3,
    )
    a = estimator_behavior(**kwargs)
    b = estimator_behavior(**kwargs)
    pd.testing.assert_frame_equal(a, b)
    assert len(a) == 1
    assert {"n_sources", "n_units"} <= set(a.columns)


def test_operating_characteristic_rates_are_bounded() -> None:
    result = operating_characteristic([0.25, 0.5, 0.75], n_designs=18, seed=7)
    assert len(result) == 3
    rate_columns = [c for c in result if c.endswith("_rate")]
    assert rate_columns
    for column in rate_columns:
        assert result[column].dropna().between(0, 1).all(), column
