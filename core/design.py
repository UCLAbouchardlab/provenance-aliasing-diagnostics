"""
Prospective design sensitivity under a declared simulation model.

How much source-group crossing might a design need before the contrast retains
useful information under provenance invariance? This module constructs synthetic
incidence tables, scores them with the same entropy and ceiling calculations used
for observed corpora, and reports how those quantities change with the design.

Simulation:
-----------
Units carry fixed group labels. A specified number of sources are forced to
span at least two groups; the remaining sources are restricted to one group.
Every unit and source receives support, then additional memberships are drawn.
Optional lognormal row masses introduce unequal weight without changing the
declared source or unit grain.

Interpretation:
---------------
The outputs describe the chosen generator, group sizes, incidence density,
crossing rule, and mass distribution. They do not supply a universal minimum
crossing fraction, an empirical validation result, or a guarantee that a biological
contrast can be recovered. R is an entropy ratio; an accuracy ceiling must be
computed from the full source x group table rather than inferred from R alone.

The simulator can construct multiple groups, but the scoring routines here
include the exact binary balanced-accuracy ceiling and therefore require a
two-group contrast. Source counts and unit counts are distinct design axes.

Contents:
---------
simulate_corpus - construct one synthetic incidence design.
design_curve - entropy and ceiling distributions across crossing fractions.
simulated_ceiling_envelope - observed ceiling ranges in realized R bins.
minimum_viable_crossing - a conservative lookup over a stated simulation grid.
estimator_behavior - finite-design variation against an independently drawn reference.
operating_characteristic - ratio-screen error rates under a stipulated ceiling label.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from string import ascii_uppercase

import numpy as np
import pandas as pd

from .entropy import label_entropy
from .incidence import Corpus
from .ceiling import balanced_accuracy_ceiling

SIMULATED_GRAIN = "simulated source (one synthetic provenance stratum per source label)"

_MAX_SEED = 2**31 - 1


def _weighting_label(mass_dispersion: float) -> str:
    """
    Describe whether simulation rows have equal mass or lognormal relative mass.

    The label is carried beside summary scores. Positive dispersion refers to
    lognormal row draws subsequently rescaled to mean 1 within each corpus.
    """
    if mass_dispersion <= 0:
        return "incidence rows (every (source, unit) row weighs 1)"
    return f"simulated data mass, lognormal(0, {mass_dispersion:g}) per row"


def _group_name(i: int) -> str:
    """
    Return a deterministic synthetic group label: A through Z, then numbered G labels.
    """
    return ascii_uppercase[i] if i < len(ascii_uppercase) else f"G{i:02d}"


def _resolve_group_sizes(
    n_units: int, group_sizes: Sequence[int] | Mapping[str, int] | None
) -> dict[str, int]:
    """
    Resolve group labels and positive unit counts for a synthetic contrast.

    Parameters:
    ----------
    n_units : int - required total number of analysis units.
    group_sizes : sequence, mapping, or None - explicit counts, named counts,
    or a default split into A and B as nearly balanced as possible.

    Returns:
    -------
    dict - group display names to integer counts, preserving supplied group order.

    Raises:
    ------
    ValueError
    - if fewer than two groups remain, any count is below one, or the counts
    do not sum to n_units.

    Notes:
    ------
    Explicit counts are converted with int and mapping labels with str. Sequence
    counts receive deterministic synthetic group names.
    """
    if group_sizes is None:
        half = n_units // 2
        sizes = {"A": half, "B": n_units - half}
    elif isinstance(group_sizes, Mapping):
        sizes = {str(k): int(v) for k, v in group_sizes.items()}
    else:
        sizes = {_group_name(i): int(v) for i, v in enumerate(group_sizes)}

    if len(sizes) < 2:
        raise ValueError("a contrast needs at least two groups")
    if any(v < 1 for v in sizes.values()):
        raise ValueError(f"every group needs at least one unit; got {sizes}")
    if sum(sizes.values()) != n_units:
        raise ValueError(f"group_sizes sum to {sum(sizes.values())}, expected n_units={n_units}")
    return sizes


def _allocate_nested(n_nested: int, counts: np.ndarray) -> np.ndarray:
    """
    Allocate non-crossing sources among groups in proportion to unit counts.

    Parameters:
    ----------
    n_nested : int - number of sources restricted to one group.
    counts : np.ndarray - positive unit counts in group order.

    Returns:
    -------
    np.ndarray, shape (n_nested,) - group index assigned to each nested source.

    Computation:
    -----------
    Integer quotas use floors followed by the largest fractional remainders.
    When there are at least as many nested sources as groups, allocations are
    adjusted to give every group at least one. The allocation is deterministic
    and does not consume the simulation's random stream.
    """
    n_groups = len(counts)
    if n_nested == 0:
        return np.empty(0, dtype=int)
    share = counts / counts.sum()
    quota = share * n_nested
    base = np.floor(quota).astype(int)
    order = np.argsort(-(quota - base))
    base[order[: n_nested - int(base.sum())]] += 1
    if n_nested >= n_groups:
        # pigeonhole: while some group has none, some other has two or more
        while (base == 0).any():
            base[int(np.argmax(base))] -= 1
            base[int(np.argmin(base))] += 1
    return np.repeat(np.arange(n_groups), base)


def _summary_statistic(values: np.ndarray, statistic: str) -> float:
    """
    Summarize finite draw values by mean, median, or a named percentile.

    Parameters:
    ----------
    values : array-like - numerical draw values; non-finite entries are excluded.
    statistic : str - ``"mean"``, ``"median"``, or ``"p<q>"`` with q in
    ``[0, 100]``, such as ``"p05"``.

    Returns:
    -------
    float - requested summary, or ``nan`` when no finite values remain.

    Raises:
    ------
    ValueError
    - if a statistic applied to a non-empty finite sample is unsupported or malformed.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan")
    if statistic == "mean":
        return float(v.mean())
    if statistic == "median":
        return float(np.median(v))
    if statistic.startswith("p"):
        try:
            q = float(statistic[1:])
        except ValueError as exc:  # pragma: no cover - message path
            raise ValueError(f"unparsable statistic {statistic!r}") from exc
        if not 0.0 <= q <= 100.0:
            raise ValueError(f"percentile out of range in {statistic!r}")
        return float(np.percentile(v, q))
    raise ValueError(f"statistic must be 'mean', 'median' or 'p<q>'; got {statistic!r}")


def _spread(values: np.ndarray, prefix: str) -> dict[str, float]:
    """
    Return mean, sample SD, 5th percentile, median, and 95th percentile of finite draws.

    Dictionary keys begin with the supplied prefix. An empty finite sample gives
    all ``nan``; one finite draw gives an undefined SD because ``ddof=1``.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {f"{prefix}_{k}": float("nan") for k in ("mean", "sd", "p05", "median", "p95")}
    return {
        f"{prefix}_mean": float(v.mean()),
        f"{prefix}_sd": float(np.std(v, ddof=1)) if v.size > 1 else float("nan"),
        f"{prefix}_p05": float(np.percentile(v, 5)),
        f"{prefix}_median": float(np.median(v)),
        f"{prefix}_p95": float(np.percentile(v, 95)),
    }


def _ceiling_of(corpus: Corpus) -> float:
    """
    Extract and check the exact binary balanced-accuracy ceiling for one corpus.

    Returns a float, preserving undefined results as ``nan``. A finite value
    outside ``[0.5, 1]`` raises RuntimeError. Binary-contrast validation is
    delegated to :func:`balanced_accuracy_ceiling`.
    """
    result = balanced_accuracy_ceiling(corpus)

    value = float(result.exact)

    if not np.isfinite(value):
        return float("nan")

    if not 0.5 <= value <= 1.0:
        raise RuntimeError(f"balanced_accuracy_ceiling returned an invalid. value {value}. expected a value in [0.5, 1.0]")
    return value


def _score(corpus: Corpus) -> dict[str, float]:
    """
    Score one synthetic binary design using entropy, exact ceiling, and design counts.

    Returns a dictionary with R, ceiling, marginal and conditional entropy in bits,
    analytical row count, and number of structurally spanning sources.
    """
    ent = label_entropy(corpus)
    return {
        "ratio": float(ent.ratio),
        "ceiling": _ceiling_of(corpus),
        "H_G": float(ent.H_G),
        "H_G_given_D": float(ent.H_G_given_D),
        "n_rows": float(corpus.n_rows),
        "n_spanning": float(len(corpus.spanning_sources())),
    }


def _seed_stream(seed: int, n: int) -> np.ndarray:
    """
    Generate n reproducible integer seeds from one local NumPy generator.

    Seeds lie in ``[0, _MAX_SEED)``. They assign draws their own random streams
    without changing NumPy's global random state; uniqueness is not guaranteed.
    """
    return np.random.default_rng(seed).integers(0, _MAX_SEED, size=n, dtype=np.int64)

def simulate_corpus(
    n_units: int,
    n_sources: int,
    crossing_fraction: float,
    *,
    group_sizes: Sequence[int] | Mapping[str, int] | None = None,
    seed: int = 0,
    extra_rows_per_source: float = 1.0,
    mass_dispersion: float = 0.0,
    name: str | None = None,
) -> Corpus:
    """
    Construct an incidence corpus with a stated number of sources spanning groups.

    A spanning source receives at least one unit from each of two different groups.
    A nested source is restricted to its allocated group. Additional rows preserve
    those restrictions, so the realized number of spanning sources is fixed by
    the requested fraction after integer rounding.

    Parameters:
    ----------
    n_units : int - number of distinct analysis units, at least 2. Converted to int.
    n_sources : int - number of provenance strata, at least 1. Converted to int.
    crossing_fraction : float - fraction in ``[0, 1]``; the spanning-source count
    is ``round(crossing_fraction * n_sources)`` using Python's rounding rule.
    group_sizes : sequence, mapping, or None, default None, keyword-only - positive
    unit counts summing to n_units for at least two groups. None gives an almost
    balanced binary contrast. Mappings name groups; sequences receive generated labels.
    seed : int, default 0, keyword-only - seed for the local random generator.
    extra_rows_per_source : float, default 1.0, keyword-only - non-negative Poisson
    mean for additional membership draws per source, beyond source and unit coverage.
    mass_dispersion : float, default 0.0, keyword-only - non-negative lognormal
    sigma for independent row masses. Zero gives unit mass; positive values draw
    ``lognormal(0, sigma)`` and rescale masses to mean 1 within the corpus.
    name : str or None, default None, keyword-only - corpus name; by default the
    label records design settings and seed.

    Returns:
    -------
    Corpus - synthetic source, unit, and group keys, with duplicate source-unit
    pairs collapsed before construction. Weighting is ``"incidence rows"`` when
    dispersion is zero and ``"weighted by mass"`` otherwise.

    Construction:
    -------------
    Crossing sources are seeded from two groups; for multigroup designs those groups
    are sampled without replacement in proportion to their unit counts. Nested
    sources are allocated deterministically by group size. Uncovered units are
    assigned to eligible sources, followed by Poisson membership draws. Collapsing
    repeated memberships means realized extra rows can be fewer than the draw count.

    Raises:
    ------
    ValueError
    - if dimensions, fractions, group counts, or mass settings are invalid, or a
    group has no eligible source under the requested crossing design.
    RuntimeError
    - if the constructed unit/source counts or spanning-source count differ from
    the requested design.

    Notes:
    ------
    An infeasible design is rejected rather than silently changing its crossing.
    In a multigroup corpus, a spanning source need not contain every group. The
    generator models incidence and relative mass only; it does not simulate
    molecular features, correction performance, or independent biological replication.
    """
    n_units = int(n_units)
    n_sources = int(n_sources)
    if n_units < 2:
        raise ValueError(f"n_units must be at least 2; got {n_units}")
    if n_sources < 1:
        raise ValueError(f"n_sources must be at least 1; got {n_sources}")
    if not 0.0 <= float(crossing_fraction) <= 1.0:
        raise ValueError(f"crossing_fraction must lie in [0, 1]; got {crossing_fraction}")
    if extra_rows_per_source < 0:
        raise ValueError("extra_rows_per_source must be non-negative")
    if mass_dispersion < 0:
        raise ValueError("mass_dispersion must be non-negative")

    sizes = _resolve_group_sizes(n_units, group_sizes)
    names = list(sizes)
    counts = np.array([sizes[g] for g in names], dtype=int)
    n_groups = len(names)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    unit_group = np.repeat(np.arange(n_groups), counts)

    rng = np.random.default_rng(seed)
    n_cross = int(min(max(round(float(crossing_fraction) * n_sources), 0), n_sources))
    n_nested = n_sources - n_cross
    nested_group = _allocate_nested(n_nested, counts)

    def pick_units(g: np.ndarray) -> np.ndarray:
        """
        Draw one unit uniformly within each supplied group using the simulation's local RNG.

        Units occupy contiguous group blocks, so group offsets convert within-group
        draws into global unit indices. An empty group vector returns an empty integer array.
        """
        g = np.asarray(g, dtype=int)
        if g.size == 0:
            return np.empty(0, dtype=int)
        return starts[g] + rng.integers(0, counts[g])

    src_parts: list[np.ndarray] = []
    unit_parts: list[np.ndarray] = []

    #seed rows for crossing sources
    if n_cross:
        if n_groups == 2:
            pairs = np.tile(np.array([0, 1]), (n_cross, 1))
        else:
            share = counts / counts.sum()
            pairs = np.stack(
                [rng.choice(n_groups, size=2, replace=False, p=share) for _ in range(n_cross)]
            )
        cross_idx = np.arange(n_cross)
        for col in range(2):
            src_parts.append(cross_idx)
            unit_parts.append(pick_units(pairs[:, col]))

    #seed row for each nested source, inside its own group
    if n_nested:
        src_parts.append(np.arange(n_cross, n_sources))
        unit_parts.append(pick_units(nested_group))

    #cover any unit no source has reached yet
    covered = np.zeros(n_units, dtype=bool)
    if unit_parts:
        covered[np.concatenate(unit_parts)] = True
    for g in range(n_groups):
        need = np.where((~covered) & (unit_group == g))[0]
        if need.size == 0:
            continue
        eligible = np.concatenate(
            [np.arange(n_cross), n_cross + np.where(nested_group == g)[0]]
        ).astype(int)
        if eligible.size == 0:
            raise ValueError(
                f"group {names[g]!r} has no eligible source: crossing_fraction="
                f"{crossing_fraction} leaves {n_nested} nested sources for {n_groups} "
                "groups. Raise crossing_fraction or n_sources."
            )
        src_parts.append(eligible[rng.integers(0, eligible.size, size=need.size)])
        unit_parts.append(need)

    #extra rows, drawn in-group for nested sources so crossing stays as requested
    if extra_rows_per_source > 0:
        k = rng.poisson(extra_rows_per_source, size=n_sources)
        if n_cross and k[:n_cross].sum():
            n_extra = int(k[:n_cross].sum())
            src_parts.append(np.repeat(np.arange(n_cross), k[:n_cross]))
            unit_parts.append(rng.integers(0, n_units, size=n_extra))
        if n_nested and k[n_cross:].sum():
            src_parts.append(np.repeat(np.arange(n_cross, n_sources), k[n_cross:]))
            unit_parts.append(pick_units(np.repeat(nested_group, k[n_cross:])))

    src = np.concatenate(src_parts)
    unit = np.concatenate(unit_parts)
    frame = pd.DataFrame({"_s": src, "_u": unit}).drop_duplicates()
    src = frame["_s"].to_numpy()
    unit = frame["_u"].to_numpy()

    df = pd.DataFrame(
        {
            "source": [f"S{i:04d}" for i in src],
            "unit": [f"U{i:04d}" for i in unit],
            "group": [names[i] for i in unit_group[unit]],
        }
    )
    if mass_dispersion > 0:
        mass = rng.lognormal(mean=0.0, sigma=float(mass_dispersion), size=len(df))
        df["mass"] = mass / mass.mean()

    label = name or (
        f"sim[u={n_units},d={n_sources},x={float(crossing_fraction)!r},"
        f"g={'/'.join(f'{g}:{sizes[g]}' for g in names)},"
        f"e={float(extra_rows_per_source)!r},m={float(mass_dispersion)!r},seed={seed}]"
    )
    corpus = Corpus.from_frame(
        df, weight="mass" if mass_dispersion > 0 else None, name=label
    )

    # the design must be the design that was asked for, not one that drifted
    if corpus.n_units != n_units or corpus.n_sources != n_sources:
        raise RuntimeError(
            f"generator lost strata: built {corpus.n_units} units / {corpus.n_sources} "
            f"sources, asked for {n_units} / {n_sources}"
        )
    realised = len(corpus.spanning_sources())
    if realised != n_cross:
        raise RuntimeError(
            f"realised {realised} spanning sources, constructed {n_cross}; the "
            "in-group restriction on nested extra rows has been violated"
        )
    return corpus


def _random_design(
    rng: np.random.Generator,
    crossing_fraction: float,
    *,
    n_units_range: tuple[int, int],
    n_sources_range: tuple[int, int],
    minority_fraction_range: tuple[float, float],
    extra_rows_range: tuple[float, float],
    mass_dispersion_range: tuple[float, float],
) -> Corpus:
    """
    Draw one binary design from the stated simulation ranges at fixed crossing.

    Parameters:
    ----------
    rng : np.random.Generator - stream used for settings and the corpus child seed.
    crossing_fraction : float - requested fraction of spanning sources.
    n_units_range, n_sources_range : pairs of int, keyword-only - inclusive
    integer ranges for design dimensions.
    minority_fraction_range : pair of float, keyword-only - uniform range for
    the A-group fraction, converted to a rounded count clipped to ``[1, n_units-1]``.
    extra_rows_range, mass_dispersion_range : pairs of float, keyword-only -
    uniform ranges for extra membership density and lognormal mass dispersion.

    Returns:
    -------
    Corpus - one :func:`simulate_corpus` draw, with its generated name recording
    the realized settings. Invalid or infeasible settings propagate as errors.
    """
    n_units = int(rng.integers(n_units_range[0], n_units_range[1] + 1))
    n_sources = int(rng.integers(n_sources_range[0], n_sources_range[1] + 1))
    minority = float(rng.uniform(*minority_fraction_range))
    n_minor = int(np.clip(round(minority * n_units), 1, n_units - 1))
    return simulate_corpus(
        n_units,
        n_sources,
        crossing_fraction,
        group_sizes={"A": n_minor, "B": n_units - n_minor},
        seed=int(rng.integers(0, _MAX_SEED)),
        extra_rows_per_source=float(rng.uniform(*extra_rows_range)),
        mass_dispersion=float(rng.uniform(*mass_dispersion_range)),
    )

def design_curve(
    n_units: int,
    n_sources: int,
    crossing_fractions: Sequence[float],
    *,
    n_rep: int = 200,
    seed: int = 0,
    group_sizes: Sequence[int] | Mapping[str, int] | None = None,
    extra_rows_per_source: float = 1.0,
    mass_dispersion: float = 0.0,
) -> pd.DataFrame:
    """
    Measure how R and the binary ceiling change across simulated crossing fractions.

    For each requested fraction, independently draw n_rep corpora of the stated
    size and summarize the diagnostic distributions. The curve describes crossing
    under this generator's within-source mixing and weighting assumptions.

    Parameters:
    ----------
    n_units, n_sources : int - fixed unit and source counts for every draw.
    crossing_fractions : sequence of float - non-empty set of requested fractions
    in ``[0, 1]``, retained in supplied order. Source counts are rounded per design.
    n_rep : int, default 200, keyword-only - draws per fraction, at least 1.
    seed : int, default 0, keyword-only - seed assigning a child seed to each draw.
    group_sizes : sequence, mapping, or None, default None, keyword-only - binary
    group counts summing to n_units; None gives an almost balanced contrast.
    extra_rows_per_source : float, default 1.0, keyword-only - Poisson extra
    membership mean, as defined by :func:`simulate_corpus`.
    mass_dispersion : float, default 0.0, keyword-only - lognormal row-mass sigma;
    zero gives equal row mass.

    Returns:
    -------
    pd.DataFrame - one row per fraction, with requested and realized crossing
    counts, design dimensions, n_rep, and ``ratio_*``/``ceiling_*`` mean, sample
    SD, p05, median, and p95 fields. Also reports ``frac_ratio_zero`` using
    ``np.isclose``, mean row and spanning-source counts, weighting, and source grain.
    Attributes record seed, generator name, and replicate count.

    Raises:
    ------
    ValueError
    - if the fraction list is empty, n_rep is below 1, or a generated design is invalid.
    Errors from the binary ceiling calculation propagate for non-binary inputs.

    Notes:
    ------
    Percentiles describe variation among simulated designs. They are not
    confidence intervals for an observed atlas. With one replicate, sample SD is
    undefined. Different requested fractions can round to the same crossing count.
    """
    fracs = [float(f) for f in crossing_fractions]
    if not fracs:
        raise ValueError("crossing_fractions is empty")
    if n_rep < 1:
        raise ValueError("n_rep must be at least 1")
    seeds = _seed_stream(seed, len(fracs) * n_rep)

    rows = []
    for i, f in enumerate(fracs):
        draws = [
            _score(
                simulate_corpus(
                    n_units,
                    n_sources,
                    f,
                    group_sizes=group_sizes,
                    seed=int(seeds[i * n_rep + j]),
                    extra_rows_per_source=extra_rows_per_source,
                    mass_dispersion=mass_dispersion,
                )
            )
            for j in range(n_rep)
        ]
        d = pd.DataFrame(draws)
        ratio = d["ratio"].to_numpy()
        rows.append(
            {
                "crossing_fraction": f,
                "n_units": int(n_units),
                "n_sources": int(n_sources),
                "n_crossing_sources": int(round(f * n_sources)),
                "n_rep": int(n_rep),
                **_spread(ratio, "ratio"),
                **_spread(d["ceiling"].to_numpy(), "ceiling"),
                "frac_ratio_zero": float(np.mean(np.isclose(ratio, 0.0))),
                "n_rows_mean": float(d["n_rows"].mean()),
                "spanning_sources_mean": float(d["n_spanning"].mean()),
                "weighting": _weighting_label(mass_dispersion),
                "grain": SIMULATED_GRAIN,
            }
        )
    out = pd.DataFrame(rows)
    out.attrs.update(seed=seed, generator="simulate_corpus", n_rep=n_rep)
    return out

def simulated_ceiling_envelope(
    ratio_grid: Sequence[float] | None = None,
    *,
    n_rep: int = 200,
    seed: int = 0,
    n_units_range: tuple[int, int] = (12, 120),
    n_sources_range: tuple[int, int] = (6, 60),
    minority_fraction_range: tuple[float, float] = (0.2, 0.5),
    extra_rows_range: tuple[float, float] = (0.0, 3.0),
    mass_dispersion_range: tuple[float, float] = (0.0, 1.5),
    min_count: int = 5,
) -> pd.DataFrame:
    """
    Summarize the ceilings observed among simulated designs with similar R.

    R is a scalar summary of the source x group distribution. It does not uniquely
    determine the exact balanced-accuracy ceiling. This function illustrates that
    distinction by drawing heterogeneous designs and grouping them by realized R.

    Parameters:
    ----------
    ratio_grid : sequence of float or None, default None - finite, strictly
    increasing bin edges within ``[0, 1]``. Defaults to 11 equally spaced edges.
    Bins are ``[lo, hi)`` except that the last bin includes its upper edge.
    n_rep : int, default 200, keyword-only - attempted designs aimed toward each
    bin; total draws are ``n_rep * (len(edges) - 1)``.
    seed : int, default 0, keyword-only - simulation generator seed.
    n_units_range : pair of int, default (12, 120), keyword-only - inclusive unit range.
    n_sources_range : pair of int, default (6, 60), keyword-only - inclusive source range.
    minority_fraction_range : pair of float, default (0.2, 0.5), keyword-only -
    uniform group-A fraction before rounding to unit counts.
    extra_rows_range : pair of float, default (0.0, 3.0), keyword-only - uniform
    range for the extra-membership Poisson mean.
    mass_dispersion_range : pair of float, default (0.0, 1.5), keyword-only -
    uniform range for lognormal row-mass sigma.
    min_count : int, default 5, keyword-only - minimum realized bin occupancy
    before ceiling summaries are reported; must be at least 1.

    Returns:
    -------
    pd.DataFrame - one row per bin with edges, occupancy n, observed ratio range,
    ``ceiling_min``, ``ceiling_median``, ``ceiling_max``, ``ceiling_spread``,
    and labels of designs attaining the extrema. Sparse bins retain their counts
    and ratio range but have undefined ceiling summaries. Rows carry weighting
    and grain; attributes include seed, draw count, and minimum occupancy.

    Computation:
    -----------
    A uniform value inside each target bin, jittered by a normal draw with SD
    0.15 and clipped to ``[0, 1]``, supplies the crossing fraction. Binning uses
    the resulting entropy ratio, not the target. Designs outside a restricted
    edge range receive no summary bin.

    Raises:
    ------
    ValueError
    - if edges, replicate count, or minimum occupancy are invalid, or simulation
    settings produce an invalid or infeasible design.

    Notes:
    ------
    The extrema are observed simulation values within finite-width ratio bins.
    They are not mathematical bounds over all designs, and some spread can reflect
    variation of R within a bin. Counts per bin need not equal n_rep. Report the
    simulation ranges and binning when using this result to illustrate the inference gap.
    """
    edges = (
        np.linspace(0.0, 1.0, 11)
        if ratio_grid is None
        else np.asarray([float(x) for x in ratio_grid], dtype=float)
    )
    if edges.size < 2:
        raise ValueError("ratio_grid must give at least two bin edges")
    if np.any(np.diff(edges) <= 0):
        raise ValueError("ratio_grid edges must be strictly ascending")
    if n_rep < 1:
        raise ValueError(f"n_rep must be at least 1; got {n_rep}")

    min_count = int(min_count)

    if min_count < 1:
        raise ValueError(f'min count must be at least 1. received {min_count}')

    if np.any(~np.isfinite(edges)):
        raise ValueError("ratio_grid contains non-finite values")

    if edges[0] < 0.0 or edges[-1] > 1.0:
        raise ValueError("ratio_grid must lie within [0,1]")
    
    n_bins = edges.size - 1
    rng = np.random.default_rng(seed)
    records = []
    for b in range(n_bins):
        lo, hi = float(edges[b]), float(edges[b + 1])
        for _ in range(n_rep):
            target = float(rng.uniform(lo, hi))
            #jitter keeps aim from becoming a hidden constraint on prior
            aim = float(np.clip(target + rng.normal(0.0, 0.15), 0.0, 1.0))
            corpus = _random_design(
                rng,
                aim,
                n_units_range=n_units_range,
                n_sources_range=n_sources_range,
                minority_fraction_range=minority_fraction_range,
                extra_rows_range=extra_rows_range,
                mass_dispersion_range=mass_dispersion_range,
            )
            rec = _score(corpus)
            rec["design"] = corpus.name
            rec["aimed_at_bin"] = b
            records.append(rec)

    draws = pd.DataFrame(records)
    idx = np.clip(np.digitize(draws["ratio"].to_numpy(), edges, right=False) - 1, 0, n_bins - 1)
    inside = (draws["ratio"].to_numpy() >= edges[0]) & (draws["ratio"].to_numpy() <= edges[-1])
    draws["bin"] = np.where(inside, idx, -1)

    rows = []
    for b in range(n_bins):
        d = draws[draws["bin"] == b]
        if len(d) < min_count:
            rows.append(
                {
                    "ratio_lo": float(edges[b]),
                    "ratio_hi": float(edges[b + 1]),
                    "n": int(len(d)),
                    "ratio_observed_min": float(d["ratio"].min()) if len(d) else float("nan"),
                    "ratio_observed_max": float(d["ratio"].max()) if len(d) else float("nan"),
                    "ceiling_min": float("nan"),
                    "ceiling_median": float("nan"),
                    "ceiling_max": float("nan"),
                    "ceiling_spread": float("nan"),
                    "design_at_min": "",
                    "design_at_max": "",
                    "weighting": "mixed: incidence rows to lognormal data mass "
                    f"(sigma in [{mass_dispersion_range[0]:g}, {mass_dispersion_range[1]:g}])",
                    "grain": SIMULATED_GRAIN,
                }
            )
            continue
        c = d["ceiling"].to_numpy()
        rows.append(
            {
                "ratio_lo": float(edges[b]),
                "ratio_hi": float(edges[b + 1]),
                "n": int(len(d)),
                "ratio_observed_min": float(d["ratio"].min()),
                "ratio_observed_max": float(d["ratio"].max()),
                "ceiling_min": float(np.nanmin(c)),
                "ceiling_median": float(np.nanmedian(c)),
                "ceiling_max": float(np.nanmax(c)),
                "ceiling_spread": float(np.nanmax(c) - np.nanmin(c)),
                "design_at_min": str(d["design"].to_numpy()[int(np.nanargmin(c))]),
                "design_at_max": str(d["design"].to_numpy()[int(np.nanargmax(c))]),
                "weighting": "mixed: incidence rows to lognormal data mass "
                f"(sigma in [{mass_dispersion_range[0]:g}, {mass_dispersion_range[1]:g}])",
                "grain": SIMULATED_GRAIN,
            }
        )
    out = pd.DataFrame(rows)
    out.attrs.update(
        seed=seed,
        n_draws=int(len(draws)),
        min_count=int(min_count),
        note="ceiling_spread is the width of the inference gap: a ratio in this bin is "
        "compatible with any ceiling in [ceiling_min, ceiling_max] under this prior, "
        "and with more outside it under a wider one",
    )
    return out

def minimum_viable_crossing(
    target_ceiling: float,
    n_units_grid: Sequence[int] = (20, 40, 80),
    n_sources_grid: Sequence[int] = (10, 20, 40),
    *,
    group_sizes: Sequence[int] | Mapping[str, int] | None = None,
    crossing_grid: Sequence[float] | None = None,
    n_rep: int = 25,
    statistic: str = "p05",
    seed: int = 0,
    extra_rows_per_source: float = 1.0,
    mass_dispersion: float = 0.0,
) -> pd.DataFrame:
    """
    Find the first simulated crossing fraction meeting a conservative ceiling target.

    The returned requirement is a lookup over a declared grid. It is conditional
    on the simulator, group sizes, row density, weights, and chosen summary of
    replicate ceilings.

    Parameters:
    ----------
    target_ceiling : float - desired binary ceiling in ``(0.5, 1]``.
    n_units_grid : sequence of int, default (20, 40, 80) - unit counts to evaluate.
    n_sources_grid : sequence of int, default (10, 20, 40) - source counts crossed
    with the unit grid. Both grids must be non-empty and contain valid counts.
    group_sizes : sequence, mapping, or None, default None, keyword-only - explicit
    binary group counts. Fixed counts require a single distinct n_units value.
    crossing_grid : sequence of float or None, default None, keyword-only - finite,
    strictly increasing fractions in ``[0, 1]``; defaults to steps of 0.05 from 0 to 1.
    n_rep : int, default 25, keyword-only - draws per size/fraction combination.
    statistic : str, default "p05", keyword-only - ceiling summary: ``"mean"``,
    ``"median"``, or ``"p<q>"`` for a percentile q in ``[0, 100]``.
    seed : int, default 0, keyword-only - seed assigned across all grid draws.
    extra_rows_per_source : float, default 1.0, keyword-only - extra membership mean.
    mass_dispersion : float, default 0.0, keyword-only - lognormal row-mass sigma.

    Returns:
    -------
    pd.DataFrame - one row per unit/source combination. ``min_crossing_fraction``
    and ``min_crossing_sources`` give the first qualifying grid value and rounded
    source count. ``achieved_ceiling`` is its conservative summary;
    ``raw_ceiling_at_crossing`` is the local unsmoothed summary, and
    ``achieved_ratio`` is the local R summary. Also reports target, statistic,
    ``reached``, ``curve_nonmonotonic``, grid maximum and median step, n_rep,
    weighting, and grain. No hit gives ``reached=False`` and undefined achieved fields.

    Computation:
    -----------
    For each fraction, summarize finite replicate ceilings. Then replace each
    summary by the minimum of the finite summaries at that fraction and all later
    grid points. Select the first value whose future minimum reaches the target.
    This prevents an isolated upward fluctuation from determining the requirement.

    Raises:
    ------
    ValueError
    - if targets, grids, counts, or statistic are invalid, fixed group counts
    conflict with the unit grid, or a simulated design is infeasible.

    Notes:
    ------
    The default p05 describes the lower tail of simulated ceilings; it is not a
    confidence bound with guaranteed coverage. The future-minimum rule checks
    the evaluated grid only. It neither interpolates a universal minimum nor
    guarantees monotonicity of real designs outside the simulated settings.
    """

    target_ceiling = float(target_ceiling)

    if not 0.5 < target_ceiling <= 1.0:
        raise ValueError(
            f"target_ceiling must lie in (0.5, 1]; "
            f"got {target_ceiling}"
        )

    n_rep = int(n_rep)

    if n_rep < 1:
        raise ValueError(
            f"n_rep must be at least 1; got {n_rep}"
        )

    units_grid = [int(x) for x in n_units_grid]
    sources_grid = [int(x) for x in n_sources_grid]

    if not units_grid:
        raise ValueError("n_units_grid is empty")

    if not sources_grid:
        raise ValueError("n_sources_grid is empty")

    if any(x < 2 for x in units_grid):
        raise ValueError(
            "every n_units value must be at least 2"
        )

    if any(x < 1 for x in sources_grid):
        raise ValueError(
            "every n_sources value must be at least 1"
        )

    grid = (np.round(np.arange(0.0, 1.0 + 1e-9, 0.05),4, ) if crossing_grid is None else np.asarray([float(f) for f in crossing_grid],dtype=float,))

    if grid.size == 0:
        raise ValueError("crossing_grid is empty")

    if np.any(~np.isfinite(grid)):
        raise ValueError("crossing_grid contains non-finite values")

    if np.any((grid < 0.0) | (grid > 1.0)):
        raise ValueError("crossing_grid values must lie in [0, 1]")

    if np.any(np.diff(grid) <= 0):
        raise ValueError("crossing_grid must be strictly ascending")

    if (group_sizes is not None and len(set(units_grid)) > 1):
        raise ValueError(
            "fixed group_sizes cannot be used with multiple "
            "n_units values. Use a single n_units value or run "
            "separate calls for each design."
        )

    step = (float(np.median(np.diff(grid))) if grid.size > 1 else float("nan"))

    n_cells = (len(units_grid) * len(sources_grid))

    per_cell = int(grid.size) * n_rep

    # Every cell/grid-point/replicate gets an addressed seed.
    # Early decisions therefore cannot shift later randomness.
    seeds = _seed_stream(seed,n_cells * per_cell, ).reshape(n_cells, int(grid.size), n_rep,)

    rows = []
    cell = -1

    for n_units in units_grid:

        for n_sources in sources_grid:

            cell += 1
            curve_rows = []

            for gi, f in enumerate(grid):
                draws = []
                for rep in range(n_rep):
                    draws.append(
                        _score(simulate_corpus(
                                n_units,
                                n_sources,
                                float(f),
                                group_sizes=group_sizes,
                                seed=int(
                                    seeds[cell, gi, rep,]),
                                extra_rows_per_source=(extra_rows_per_source),
                                mass_dispersion=(mass_dispersion),
                            )
                        )
                    )

                d = pd.DataFrame(draws)

                curve_rows.append(
                    {
                        "crossing_fraction": float(f),
                        "ceiling": _summary_statistic(d["ceiling"].to_numpy(),statistic, ),
                        "ratio": _summary_statistic(d["ratio"].to_numpy(),statistic, ),
                    }
                )

            curve = pd.DataFrame(curve_rows)

            raw_ceiling = (curve["ceiling"].to_numpy(dtype=float))

            finite = np.isfinite(raw_ceiling)

            finite_values = (raw_ceiling[finite])

            curve_nonmonotonic = bool(finite_values.size > 1 and np.any(np.diff(finite_values)< 0.0))

            future_min_input = np.where(finite, raw_ceiling, np.inf,)

            conservative_ceiling = (np.minimum.accumulate(future_min_input[::-1])[::-1])

            conservative_ceiling[~np.isfinite(conservative_ceiling)] = np.nan

            hits = np.flatnonzero(np.isfinite(conservative_ceiling) & (conservative_ceiling >= target_ceiling))

            if hits.size:
                i = int( hits[0])
                hit = {
                    "min_crossing_fraction": float(curve.iloc[i]["crossing_fraction"]),
                    "achieved_ceiling": float(conservative_ceiling[i]),
                    "raw_ceiling_at_crossing":float(raw_ceiling[i]),
                    "achieved_ratio": float(curve.iloc[i]["ratio"]),
                }

            else:
                hit = None

            rows.append(
                {"n_units": n_units,
                "n_sources": n_sources,
                "target_ceiling": target_ceiling,
                "statistic": statistic,
                "min_crossing_fraction": (hit["min_crossing_fraction"] if hit else float("nan")),
                "min_crossing_sources": (float(round(hit["min_crossing_fraction"]* n_sources)) if hit else float("nan")),
                "achieved_ceiling":(hit["achieved_ceiling"]if hit else float("nan")),
                "raw_ceiling_at_crossing":(hit["raw_ceiling_at_crossing"]if hit else float("nan")),
                "achieved_ratio":(hit["achieved_ratio"] if hit else float("nan")),
                "reached": bool(hit),
                "curve_nonmonotonic": curve_nonmonotonic,
                "crossing_grid_max": float(grid[-1]),
                "crossing_grid_step": step,
                "n_rep": n_rep,
                "weighting":_weighting_label(mass_dispersion),
                "grain":SIMULATED_GRAIN,
                })
    out = pd.DataFrame(
        rows
    )
    out.attrs.update(
        seed=seed,
        note=(
            "min_crossing_fraction is the first grid value "
            "whose conservative future-minimum ceiling reaches "
            "the target. NaN means the target was not reached. "
            "The recommendation is conditional on the stated "
            "simulation model."
        ),
    )

    return out


def estimator_behavior(
    target_ratio: float,
    n_sources_grid: Sequence[int] = (5,10,20,40,80,),
    n_units_grid: Sequence[int] = (20,50,100,),
    *,
    n_rep: int = 200,
    seed: int = 0,
    reference_n_sources: int = 200,
    reference_n_units: int = 200,
    calibration_points: int = 11,
    calibration_rep: int = 5,
    reference_rep: int = 50,
    extra_rows_per_source: float = 1.0,
    mass_dispersion: float = 0.0,
) -> pd.DataFrame:
    """
    Compare finite-design R against an independently simulated reference ratio.

    Only positive-mass spanning sources contribute conditional label entropy. The
    number of sources and their allocation across units can therefore affect R
    even when the requested crossing fraction is held fixed.

    Parameters:
    ----------
    target_ratio : float - desired reference-design R in ``[0, 1]``.
    n_sources_grid : sequence of int, default (5, 10, 20, 40, 80) - source counts.
    n_units_grid : sequence of int, default (20, 50, 100) - unit counts crossed
    with the source grid. Both grids must be non-empty.
    n_rep : int, default 200, keyword-only - draws per finite-design combination.
    seed : int, default 0, keyword-only - finite-design seed; calibration and
    reference streams use seed+1 and seed+2.
    reference_n_sources : int, default 200, keyword-only - reference source count.
    reference_n_units : int, default 200, keyword-only - reference unit count.
    calibration_points : int, default 11, keyword-only - equally spaced crossing
    fractions from 0 to 1, at least 2.
    calibration_rep : int, default 5, keyword-only - calibration draws per fraction.
    reference_rep : int, default 50, keyword-only - independent reference draws,
    at least 2.
    extra_rows_per_source : float, default 1.0, keyword-only - extra membership mean.
    mass_dispersion : float, default 0.0, keyword-only - lognormal row-mass sigma.

    Returns:
    -------
    pd.DataFrame - one row per source/unit size. Reports target and actual
    ``reference_ratio``, reference SD/SEM and finite replicate count,
    ``calibration_error``, chosen crossing fraction, R distribution summaries,
    ``bias``, ``rmse``, ``frac_zero`` using np.isclose, mean spanning-source count,
    n_rep, weighting, and grain. Attributes retain the reference design and seed context.

    Computation:
    -----------
    Mean R is evaluated over the calibration crossing grid and made non-decreasing
    with a cumulative maximum. Interpolation selects a crossing fraction for the
    target. Independent draws at that fraction define the reference mean. Bias
    and RMSE compare finite-design ratios with this mean, not with target_ratio.

    Raises:
    ------
    ValueError
    - if settings are invalid, the target lies outside the calibrated range, or
    the chosen generator settings cannot construct a requested design.
    RuntimeError
    - if a calibration point has no finite scores or fewer than two finite
    reference scores are obtained.

    Notes:
    ------
    The reference is simulated and uncertain; ``reference_ratio_sem`` reports its
    Monte Carlo mean uncertainty. Changing corpus dimensions also changes the
    generator's design distribution, so bias here is relative to that reference,
    not proof of a general estimator bias for observed resources. All groups use
    the simulator's default nearly balanced binary allocation.
    """


    target_ratio = float(target_ratio)
    n_rep = int(n_rep)

    calibration_points = int(calibration_points)

    calibration_rep = int(calibration_rep)

    reference_rep = int(reference_rep)

    reference_n_sources = int(reference_n_sources)

    reference_n_units = int(reference_n_units)

    if not 0.0 <= target_ratio <= 1.0:
        raise ValueError(f"target_ratio must lie in [0, 1]; "
            f"got {target_ratio}")

    if n_rep < 1:
        raise ValueError(
            f"n_rep must be at least 1; "
            f"got {n_rep}")

    if calibration_points < 2:
        raise ValueError("calibration_points must be at least 2")

    if calibration_rep < 1:
        raise ValueError(
            f"calibration_rep must be at least 1; "
            f"got {calibration_rep}")

    if reference_rep < 2:
        raise ValueError(
            f"reference_rep must be at least 2; "
            f"got {reference_rep}")

    if reference_n_sources < 1:
        raise ValueError("reference_n_sources must be at least 1")

    if reference_n_units < 2:
        raise ValueError("reference_n_units must be at least 2")

    sources_grid = [int(x) for x in n_sources_grid]

    units_grid = [int(x) for x in n_units_grid]

    if not sources_grid:
        raise ValueError("n_sources_grid is empty")

    if not units_grid:
        raise ValueError("n_units_grid is empty")

    if any(x < 1 for x in sources_grid):
        raise ValueError("every n_sources value must be at least 1")

    if any(x < 2 for x in units_grid):
        raise ValueError("every n_units value must be at least 2")

    cal_f = np.linspace(0.0, 1.0, calibration_points,)

    cal_seeds = _seed_stream(seed + 1,cal_f.size * calibration_rep,)

    cal_ratio = np.empty(cal_f.size, dtype=float,)

    for i, f in enumerate(cal_f):
        vals = np.asarray(
            [
                _score(
                    simulate_corpus(
                        reference_n_units,
                        reference_n_sources,
                        float(f),
                        seed=int(cal_seeds[i * calibration_rep + j]),

                        extra_rows_per_source=(extra_rows_per_source),

                        mass_dispersion=(mass_dispersion), ))["ratio"]
                for j in range(calibration_rep)], dtype=float, )
        vals = vals[np.isfinite(vals)]

        if vals.size == 0:
            raise RuntimeError(
                "all calibration draws were "
                "non-finite at crossing_fraction="
                f"{f:.4f}"
            )

        cal_ratio[i] = float(vals.mean())

    monotone_ratio = (np.maximum.accumulate(cal_ratio))

    if not (monotone_ratio[0]<= target_ratio <= monotone_ratio[-1]):
        raise ValueError(
            f"target_ratio={target_ratio} "
            "is outside the range this generator "
            "reaches at the reference size "
            f"({monotone_ratio[0]:.4f} to "
            f"{monotone_ratio[-1]:.4f}); "
            "change the reference design or target"
        )

    f_star = float(np.interp(target_ratio, monotone_ratio,cal_f,))

    ref_seeds = _seed_stream(seed + 2,reference_rep,)

    reference_values = np.asarray(
        [
            _score(
                simulate_corpus(
                    reference_n_units,
                    reference_n_sources,
                    f_star,

                    seed=int(s),

                    extra_rows_per_source=(extra_rows_per_source),
                    mass_dispersion=(mass_dispersion),))["ratio"] for s in ref_seeds
        ], dtype=float,
    )

    reference_values = (reference_values[np.isfinite(reference_values)])

    if reference_values.size < 2:
        raise RuntimeError(
            "fewer than two finite reference "
            "draws were produced"
        )

    reference_ratio = float(reference_values.mean())

    reference_sd = float(np.std(reference_values, ddof=1,))

    reference_sem = float(reference_sd / np.sqrt(reference_values.size))

    seeds = _seed_stream(seed, len(sources_grid)* len(units_grid)* n_rep, )

    counter = 0
    rows = []

    for n_sources in sources_grid:
        for n_units in units_grid:
            vals = []
            spanning = []
            for _ in range(n_rep):

                sc = _score(
                    simulate_corpus(
                        n_units,
                        n_sources,
                        f_star,

                        seed=int(seeds[counter]),

                        extra_rows_per_source=(extra_rows_per_source),

                        mass_dispersion=(mass_dispersion),
                    )
                )

                counter += 1

                vals.append(sc["ratio"])

                spanning.append(sc["n_spanning"])

            v = np.asarray(vals, dtype=float,)

            finite = v[np.isfinite(v)]

            rows.append(
                {
                    "n_sources": n_sources,
                    "n_units": n_units,
                    "target_ratio": target_ratio,
                    "reference_ratio": reference_ratio,
                    "reference_ratio_sd": reference_sd,
                    "reference_ratio_sem": reference_sem,
                    "reference_rep": int(reference_values.size),
                    "calibration_error":(reference_ratio - target_ratio),
                    "crossing_fraction":f_star,**_spread(v, "ratio", ),
                    "bias":(float(finite.mean()) - reference_ratio if finite.size else float("nan")),
                    "rmse":(float(np.sqrt(np.mean((finite - reference_ratio) ** 2)))) if finite.size else float("nan"),
                    "frac_zero": (float(np.mean(np.isclose(finite, 0.0))) if finite.size else float("nan")),
                    "spanning_sources_mean" : float(np.mean(spanning)),
                    "n_rep": n_rep,
                    "weighting": _weighting_label(mass_dispersion),
                    "grain": SIMULATED_GRAIN, }
            )

    out = pd.DataFrame(rows)

    out.attrs.update(
        seed=seed,

        f_star=f_star,

        reference=(reference_n_units, reference_n_sources, ),

        reference_rep=int(reference_values.size),

        reference_ratio=(reference_ratio),

        reference_ratio_sd=(reference_sd),

        note=(
            "bias is measured against reference_ratio, "
            "the value actually attained by the large "
            "synthetic reference design at f_star; "
            "calibration_error is "
            "reference_ratio - target_ratio"
        ),
    )

    return out

def operating_characteristic(
    threshold_grid: Sequence[float] | None = None,
    *,
    estimable_ceiling: float = 0.75,
    n_designs: int = 600,
    seed: int = 0,
    n_units_range: tuple[int, int] = (12, 120),
    n_sources_range: tuple[int, int] = (6, 60),
    minority_fraction_range: tuple[float, float] = (0.2, 0.5),
    extra_rows_range: tuple[float, float] = (0.0, 3.0),
    mass_dispersion_range: tuple[float, float] = (0.0, 1.5),
) -> pd.DataFrame:
    """
    Evaluate R as a screen against a stipulated binary-ceiling design label.

    A design is labeled estimable when its exact ceiling reaches the caller's
    ``estimable_ceiling``. The screen flags designs with ``R < threshold``.
    The positive class for sensitivity and predictive values is therefore the
    designs below the stipulated ceiling, not the biological group of an observation.

    Parameters:
    ----------
    threshold_grid : sequence of float or None, default None - R thresholds in
    supplied order; defaults to 0.05 through 0.95 in steps of 0.05. Supply finite,
    interpretable values in ``[0, 1]``; this function only checks that the grid is non-empty.
    estimable_ceiling : float, default 0.75, keyword-only - ceiling at or above
    which the synthetic design receives the estimable label, in ``(0.5, 1]``.
    n_designs : int, default 600, keyword-only - designs drawn before excluding
    non-finite scores, at least 1.
    seed : int, default 0, keyword-only - random generator seed.
    n_units_range : pair of int, default (12, 120), keyword-only - inclusive unit range.
    n_sources_range : pair of int, default (6, 60), keyword-only - inclusive source range.
    minority_fraction_range : pair of float, default (0.2, 0.5), keyword-only -
    uniform group-A fraction before rounding.
    extra_rows_range : pair of float, default (0.0, 3.0), keyword-only - uniform
    range for the extra-membership Poisson mean.
    mass_dispersion_range : pair of float, default (0.0, 1.5), keyword-only -
    uniform range for lognormal mass dispersion.

    Returns:
    -------
    pd.DataFrame - one row per threshold, with finite design count, stipulated
    ceiling, ``prevalence_estimable``, number flagged, ``false_alarm_rate``,
    ``miss_rate``, ``sensitivity``, ``specificity``, ``youden_j``, ``ppv``,
    ``npv``, and ``accuracy``. Rows also carry weighting and source grain.
    Attributes record seed and class counts. A rate with an empty denominator is ``nan``.

    Computation:
    -----------
    Crossing fractions are drawn uniformly from ``[0, 1]``. All thresholds score
    the same finite designs. False alarms are flagged designs among those meeting
    the stipulated ceiling; misses are unflagged designs among those falling below it.

    Raises:
    ------
    ValueError
    - if the grid is empty, ceiling label or draw count is invalid, or simulation
    settings are invalid or infeasible.
    RuntimeError
    - if no design has both a finite R and a finite ceiling.

    Notes:
    ------
    These error rates are conditional on the simulation prior and the chosen
    ceiling label. Random draws do not guarantee that both classes are populated.
    The rates are not empirical operating characteristics for an omics assay,
    and rows share one simulated sample rather than independent evaluations.
    """
    thresholds = (
        np.round(np.arange(0.05, 1.0, 0.05), 4)
        if threshold_grid is None
        else np.asarray([float(t) for t in threshold_grid], dtype=float)
    )
    if thresholds.size == 0:
        raise ValueError("threshold_grid is empty")
    if not 0.5 < float(estimable_ceiling) <= 1.0:
        raise ValueError(f"estimable_ceiling must lie in (0.5, 1]; got {estimable_ceiling}")
    if n_designs < 1:
        raise ValueError(f"n_designs must be at least 1; got {n_designs}")

    rng = np.random.default_rng(seed)
    ratios = np.empty(n_designs)
    ceilings = np.empty(n_designs)
    for i in range(n_designs):
        corpus = _random_design(
            rng,
            float(rng.uniform(0.0, 1.0)),
            n_units_range=n_units_range,
            n_sources_range=n_sources_range,
            minority_fraction_range=minority_fraction_range,
            extra_rows_range=extra_rows_range,
            mass_dispersion_range=mass_dispersion_range,
        )
        sc = _score(corpus)
        ratios[i] = sc["ratio"]
        ceilings[i] = sc["ceiling"]

    ok = np.isfinite(ratios) & np.isfinite(ceilings)
    ratios, ceilings = ratios[ok], ceilings[ok]
    if ratios.size == 0:
        raise RuntimeError("every simulated design scored NaN; nothing to characterise")
    estimable = ceilings >= float(estimable_ceiling)
    n_pos, n_neg = int((~estimable).sum()), int(estimable.sum())

    def rate(num: int, den: int) -> float:
        """
        Return an empirical rate, preserving an empty class or prediction set as ``nan``.
        """
        return float(num) / den if den else float("nan")

    weighting = (
        "mixed: incidence rows to lognormal data mass "
        f"(sigma in [{mass_dispersion_range[0]:g}, {mass_dispersion_range[1]:g}])"
    )
    rows = []
    for t in thresholds:
        flagged = ratios < float(t)
        tp = int((flagged & ~estimable).sum())
        fp = int((flagged & estimable).sum())
        fn = int((~flagged & ~estimable).sum())
        tn = int((~flagged & estimable).sum())
        sens = rate(tp, n_pos)
        spec = rate(tn, n_neg)
        rows.append(
            {
                "threshold": float(t),
                "n_designs": int(ratios.size),
                "prevalence_estimable": rate(n_neg, int(ratios.size)),
                "estimable_ceiling": float(estimable_ceiling),
                "n_flagged": int(flagged.sum()),
                "false_alarm_rate": rate(fp, n_neg),
                "miss_rate": rate(fn, n_pos),
                "sensitivity": sens,
                "specificity": spec,
                "youden_j": (
                    sens + spec - 1.0
                    if np.isfinite(sens) and np.isfinite(spec)
                    else float("nan")
                ),
                "ppv": rate(tp, tp + fp),
                "npv": rate(tn, tn + fn),
                "accuracy": rate(tp + tn, int(ratios.size)),
                "weighting": weighting,
                "grain": SIMULATED_GRAIN,
            }
        )
    out = pd.DataFrame(rows)
    out.attrs.update(
        seed=seed,
        n_estimable=n_neg,
        n_not_estimable=n_pos,
        positive_class="not estimable (the screen fires when the ratio is below the threshold)",
        note="rows share one simulated draw, so thresholds are not independent evaluations",
    )
    return out
