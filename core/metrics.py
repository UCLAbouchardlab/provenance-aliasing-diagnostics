"""
Sensitivity of unit comparisons to the chosen geometry.

Does a conclusion drawn from a pairwise matrix persist when the same units are
compared in another geometry? This module provides six metrics behind one result
interface. It operates on caller-supplied profiles; it does not infer features,
aggregate assay records into biological units, or diagnose provenance by itself.

Input structure:
----------------
A two-dimensional array has shape ``(n_units, n_features)`` and is treated as
one block. A three-dimensional array has shape
``(n_units, n_blocks, n_categories)``. Categories must have matching meanings
and order across units. Compositional metrics require every block to be finite,
non-negative, and sum to 1 within the specified tolerance.

Geometries:
-----------
Pearson compares flattened profiles by correlation. Euclidean measures raw-scale
differences. Fisher-Rao, Hellinger, and Jensen-Shannon compare probability
distributions. Aitchison compares log-ratios and requires an explicit treatment
of zero cells. Except for Pearson's flattened correlation, block distances are
combined on a root-mean-square scale.

Interpretation:
---------------
Raw distances and dispersion-gap magnitudes have different scales across metric
families. Compare ranks, effect direction, or separately calibrated conclusions.
Agreement among metrics describes geometric sensitivity; it does not remove
shared provenance or establish independent biological replication.

Contents:
---------
MetricResult - distance, similarity, labels, and the geometry's reporting context.
pearson, fisher_rao, hellinger, jensen_shannon, euclidean, aitchison - pairwise metrics.
METRICS - registry of the six built-in metric functions.
metric_agreement - Spearman agreement over unordered unit pairs.
aitchison_pseudocount_sweep - sensitivity of a caller's score to zero treatment.
"""

from __future__ import annotations

import inspect
import math
from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.special import rel_entr
from scipy.stats import rankdata

#: Default absolute tolerance on ``|block.sum() - 1|`` for the simplex check.
SIMPLEX_TOL = 1e-6

_LN2 = float(np.log(2.0))

@dataclass(frozen=True)
class MetricResult:
    """
    A pairwise geometry over units, together with the choices that produced it.

    Attributes:
    -----------
    distance : np.ndarray, shape (n_units, n_units) - finite symmetric
    dissimilarities with a zero diagonal and non-negative entries up to tolerance.
    similarity : np.ndarray, shape (n_units, n_units) - finite symmetric values.
    For the distance-native built-ins this is ``-distance`` with diagonal 0;
    Pearson instead stores correlation r with diagonal 1.
    name : str, default "metric" - geometry name, matching the registry key
    for built-in metrics.
    is_distance : bool, default True - whether distance is the native quantity.
    n_blocks : int, default 1 - number of blocks used in metric scaling.
    n_categories : int or None, default None - categories per block, when known.
    params : mapping - answer-changing parameters retained by the metric, such as
    Aitchison's pseudocount and centering convention or Jensen-Shannon's log base.
    labels : tuple of str or None - optional unit labels in matrix order.
    unit_grain : str or None - caller's description of what one unit represents.
    note : str - interpretation or scaling caveat attached at construction.

    Notes:
    ------
    Construction validates matrix shapes, finite values, symmetry, distance
    diagonal, distance non-negativity, and label count. It does not test the
    triangle inequality or verify a mathematical relationship between the two
    matrices for caller-created results. The frozen dataclass does not make
    contained arrays or parameter mappings immutable.

    Negative distance is a comparison convention, not a bounded correlation.
    Report the geometry and parameters whenever a downstream statistic uses it.
    """

    distance: np.ndarray = field(repr=False)
    similarity: np.ndarray = field(repr=False)
    name: str = "metric"
    is_distance: bool = True
    n_blocks: int = 1
    n_categories: int | None = None
    params: Mapping[str, object] = field(default_factory=dict)
    labels: tuple[str, ...] | None = None
    unit_grain: str | None = None
    note: str = ""

    def __post_init__(self) -> None:
        """
        Validate compatible square matrices, finite entries, symmetry, and label count.

        Distance must have a zero diagonal and no values below ``-1e-12``. Symmetry
        uses absolute tolerance ``1e-10``. Validation raises ValueError without
        silently repairing caller-supplied matrices or replacing stored array objects.
        """
        d = np.asarray(self.distance, dtype=float)
        s = np.asarray(self.similarity, dtype=float)

        if d.ndim != 2 or d.shape[0] != d.shape[1]:
            raise ValueError(f"{self.name}: distance must be square, got {d.shape}")

        if s.shape != d.shape:
            raise ValueError(
                f"{self.name}: similarity shape {s.shape} "
                f"!= distance shape {d.shape}")

        if not np.isfinite(d).all():
            raise ValueError(f"{self.name}: distance contains non-finite values")

        if not np.isfinite(s).all():
            raise ValueError(f"{self.name}: similarity contains non-finite values")

        if not np.allclose(d, d.T, atol=1e-10, rtol=0.0):
            raise ValueError(f"{self.name}: distance is not symmetric")

        if not np.allclose(s, s.T, atol=1e-10, rtol=0.0):
            raise ValueError(f"{self.name}: similarity is not symmetric")

        if not np.allclose(np.diag(d), 0.0, atol=1e-10):
            raise ValueError(f"{self.name}: distance diagonal must be zero")

        if np.any(d < -1e-12):
            raise ValueError(f"{self.name}: distance contains negative values")

        if self.labels is not None and len(self.labels) != d.shape[0]:
            raise ValueError(
                f"{self.name}: {len(self.labels)} labels "
                f"for {d.shape[0]} units")

    @property
    def n_units(self) -> int:
        """
        Number of units on either axis of the pairwise distance matrix.
        """
        return int(np.asarray(self.distance).shape[0])

    @property
    def geometry(self) -> str:
        """
        Geometry name followed by its parameters in sorted-key order.

        Use this label beside results whose interpretation depends on the metric settings.
        """
        if not self.params:
            return self.name
        bits = ", ".join(f"{k}={v!r}" for k, v in sorted(self.params.items()))
        return f"{self.name}({bits})"

    def matrix(self, which: str = "distance") -> np.ndarray:
        """
        Select the distance or similarity matrix as a NumPy array.

        Parameters:
        ----------
        which : str, default "distance" - either ``"distance"`` or ``"similarity"``.

        Returns:
        -------
        np.ndarray - the selected matrix. A copy is not guaranteed.

        Raises:
        ------
        ValueError
        - if the selection is not one of the two supported names.
        """
        if which == "distance":
            return np.asarray(self.distance)
        if which == "similarity":
            return np.asarray(self.similarity)
        raise ValueError(f"which must be 'distance' or 'similarity', got {which!r}")

    def pairs(self, which: str = "distance") -> np.ndarray:
        """
        Return one value for each unordered unit pair, excluding the diagonal.

        Parameters:
        ----------
        which : str, default "distance" - matrix selected by :meth:`matrix`.

        Returns:
        -------
        np.ndarray, shape (n_units * (n_units - 1) // 2,) - upper-triangle values
        in NumPy row order. Matrices compared through these vectors must share unit order.
        """
        m = self.matrix(which)
        return m[np.triu_indices(m.shape[0], 1)]

    def as_frame(self, which: str = "distance") -> pd.DataFrame:
        """
        Return the selected matrix as a DataFrame with matching row and column labels.

        Parameters:
        ----------
        which : str, default "distance" - matrix selected by :meth:`matrix`.

        Returns:
        -------
        pd.DataFrame - uses stored unit labels or integer positions when labels are absent.
        """
        idx = list(self.labels) if self.labels is not None else list(range(self.n_units))
        return pd.DataFrame(self.matrix(which), index=idx, columns=idx)

    def summary(self) -> str:
        """
        One-line description of geometry, dimensions, native orientation, unit grain, and note.
        """
        native = "distance (similarity = -distance)" if self.is_distance else "similarity"
        grain = self.unit_grain or "grain unspecified"
        cats = "?" if self.n_categories is None else self.n_categories
        extra = f"; {self.note}" if self.note else ""
        return (
            f"{self.geometry}: {self.n_units} units, {self.n_blocks} blocks x {cats} "
            f"categories, native {native}, {grain}{extra}"
        )

    def __repr__(self) -> str:
        """
        Compact representation naming the geometry and number of units.
        """
        return f"MetricResult({self.geometry}, {self.n_units} units)"


@dataclass(frozen=True)
class _Profiles:
    """
    Validated profiles in a common three-dimensional representation.

    Attributes:
    -----------
    blocks : np.ndarray, shape (n_units, n_blocks, n_categories) - numerical profiles.
    was_flat : bool - True when the caller supplied a two-dimensional single block.
    labels : tuple of str or None - unit labels in row order.

    Notes:
    ------
    ``flat`` concatenates blocks for calculations that use feature vectors.
    Validation and any permitted simplex normalization occur in :func:`_prepare`.
    """

    blocks: np.ndarray
    was_flat: bool
    labels: tuple[str, ...] | None

    @property
    def n_units(self) -> int:
        """
        Length of the unit axis.
        """
        return int(self.blocks.shape[0])

    @property
    def n_blocks(self) -> int:
        """
        Number of blocks in each unit profile.
        """
        return int(self.blocks.shape[1])

    @property
    def n_categories(self) -> int:
        """
        Number of categories in each block.
        """
        return int(self.blocks.shape[2])

    @property
    def flat(self) -> np.ndarray:
        """
        Profiles reshaped to ``(n_units, n_blocks * n_categories)`` in block order.
        """
        return self.blocks.reshape(self.n_units, -1)


def _unit_name(labels: tuple[str, ...] | None, i: int) -> str:
    """
    Identify a unit by its supplied label or positional index in an error message.
    """
    return f"{labels[i]!r}" if labels is not None else f"index {i}"


def _prepare(
    X,
    *,
    metric: str,
    require_simplex: bool,
    tol: float = SIMPLEX_TOL,
    labels: Sequence[str] | None = None,
) -> _Profiles:
    """
    Coerce profiles, validate dimensions, and optionally enforce a per-block simplex.

    Parameters:
    ----------
    X : array-like or pd.DataFrame - two-dimensional profiles or a three-dimensional
    unit x block x category stack. DataFrame indices supply labels if none are given.
    metric : str, keyword-only - geometry name used in validation errors.
    require_simplex : bool, keyword-only - require non-negative blocks summing to 1.
    tol : float, default SIMPLEX_TOL, keyword-only - maximum absolute deviation
    of a block sum from 1 before a simplex input is rejected.
    labels : sequence or None, keyword-only - optional labels matching the unit count.

    Returns:
    -------
    _Profiles - finite profiles with at least two units and non-empty block axes.
    Accepted simplex blocks are renormalized exactly by their positive sums.

    Raises:
    ------
    ValueError
    - if dimensions, label counts, finite-value checks, or simplex requirements fail.

    Notes:
    ------
    A two-dimensional input is one block, even if its columns were assembled by
    concatenating several compositions. Reshape those compositions explicitly;
    the function cannot infer their boundaries from the values.
    """
    if isinstance(X, (pd.DataFrame, pd.Series)):
        if labels is None and isinstance(X, pd.DataFrame):
            labels = [str(i) for i in X.index]
        X = X.to_numpy()
    A = np.asarray(X, dtype=float)

    if A.ndim == 2:
        A = A[:, None, :]
        was_flat = True
    elif A.ndim == 3:
        was_flat = False
    else:
        raise ValueError(
            f"{metric}: expected (n_units, n_blocks, n_categories) or "
            f"(n_units, n_features), got array with shape {A.shape}"
        )

    n, b, c = A.shape
    if n < 2:
        raise ValueError(f"{metric}: need at least 2 units to form a pairwise matrix, got {n}")
    if b < 1 or c < 1:
        raise ValueError(f"{metric}: empty block structure, got shape {A.shape}")

    lab = tuple(str(x) for x in labels) if labels is not None else None
    if lab is not None and len(lab) != n:
        raise ValueError(f"{metric}: {len(lab)} labels for {n} units")

    if not np.isfinite(A).all():
        k = int(np.argmax(~np.isfinite(A)))
        i, j, _ = np.unravel_index(k, A.shape)
        raise ValueError(
            f"{metric}: input contains non-finite values, first at unit "
            f"{_unit_name(lab, int(i))}, block {int(j)}"
        )

    if require_simplex:
        if A.min() < 0.0:
            k = int(np.argmin(A))
            i, j, q = np.unravel_index(k, A.shape)
            raise ValueError(
                f"{metric}: compositional input must be non-negative; most negative cell "
                f"is {A.flat[k]:.6g} at unit {_unit_name(lab, int(i))}, block {int(j)}, "
                f"category {int(q)}"
            )
        sums = A.sum(axis=-1)
        dev = np.abs(sums - 1.0)
        k = int(np.argmax(dev))
        i, j = np.unravel_index(k, dev.shape)
        if dev.flat[k] > tol:
            hint = (
                f" Input was 2-D, so it was read as one block of {c} categories; if it is "
                f"really a concatenation of several blocks, reshape to "
                f"(n_units, n_blocks, n_categories) first."
                if was_flat
                else ""
            )
            raise ValueError(
                f"{metric}: every block must sum to 1 within tol={tol:g}. Worst offender: "
                f"unit {_unit_name(lab, int(i))}, block {int(j)}, sum "
                f"{sums.flat[k]:.12g} (deviation {dev.flat[k]:.3g}).{hint}"
            )
        if sums.min() <= 0.0:
            z = int(np.argmin(sums))
            zi, zj = np.unravel_index(z, sums.shape)
            raise ValueError(
                f"{metric}: block {int(zj)} of unit {_unit_name(lab, int(zi))} has zero "
                f"total mass; tol={tol:g} is too loose to be a simplex check"
            )
        A = A / sums[..., None]

    return _Profiles(blocks=A, was_flat=was_flat, labels=lab)


def _symmetrise(D: np.ndarray) -> np.ndarray:
    """
    Average a matrix with its transpose and set its diagonal to zero.

    Used to remove round-off asymmetry from computed distances; it does not
    validate an arbitrary input as a metric.
    """
    D = 0.5 * (D + D.T)
    np.fill_diagonal(D, 0.0)
    return D


def _sq_distances(Z: np.ndarray) -> np.ndarray:
    """
    Pairwise squared Euclidean distances between the rows of Z.

    Uses the Gram identity ``||z_i||^2 + ||z_j||^2 - 2*z_i.z_j`` and clips
    negative round-off values to zero. Returns a dense square array.
    """
    g = Z @ Z.T
    d = np.diag(g)
    return np.clip(d[:, None] + d[None, :] - 2.0 * g, 0.0, None)


def _bhattacharyya(P: np.ndarray) -> np.ndarray:
    """
    Compute ``sum_k sqrt(p_ik * p_jk)`` separately for each probability block.

    P has shape ``(n_units, n_blocks, n_categories)``. The returned array has
    shape ``(n_blocks, n_units, n_units)`` and coefficients clipped to ``[0, 1]``.
    """
    S = np.sqrt(P)
    return np.clip(np.einsum("ibk,jbk->bij", S, S), 0.0, 1.0)

def pearson(
    X,
    *,
    labels: Sequence[str] | None = None,
    unit_grain: str | None = None,
) -> MetricResult:
    """
    Pearson correlation between flattened unit profiles.

    All blocks are concatenated into one feature vector per unit. Correlation
    centers and scales that vector; the native result is similarity r.

    Parameters:
    ----------

    X : array-like or pd.DataFrame - shape ``(n_units, n_features)`` for one
    block, or ``(n_units, n_blocks, n_categories)``. At least two units and
    non-empty feature/block axes are required. Unit and category order must match.

    labels : sequence or None, default None, keyword-only - one label per unit.
    When X is a DataFrame, its index supplies labels if none are passed.
    unit_grain : str or None, default None, keyword-only - description of what
    one profile represents; retained for reporting without changing the calculation.

    Returns:
    -------
    MetricResult - ``similarity = r`` in ``[-1, 1]`` with diagonal 1, and
    ``distance = sqrt(1 - r)`` with diagonal 0. Original block dimensions and
    unit labels are retained, with ``is_distance=False``.

    Raises:
    ------
    ValueError
    - if profiles are non-finite, have fewer than two total features, are constant
    within a unit, or yield a non-finite correlation; also for invalid shapes or labels.

    Notes:
    ------
    Inputs may contain negative values and need not sum to 1. Block boundaries
    do not enter the correlation calculation. A high correlation describes profile
    shape after centering and scaling, not equality of absolute abundance.
    """
    prof = _prepare(X, metric="pearson", require_simplex=False, labels=labels)
    F = prof.flat

    if F.shape[1] < 2:
        raise ValueError("pearson: need at least 2 features per unit")
    const = np.ptp(F, axis=1) == 0.0
    if const.any():
        bad = [_unit_name(prof.labels, int(i)) for i in np.flatnonzero(const)[:5]]
        raise ValueError(
            f"pearson: correlation is undefined for {int(const.sum())} unit(s) with a "
            f"constant profile (e.g. {', '.join(bad)}); drop them or use a metric that "
            f"does not standardise, such as euclidean or hellinger"
        )

    with np.errstate(invalid="ignore", divide="ignore"):
        R = np.corrcoef(F)
    if not np.isfinite(R).all():
        k = int(np.argmax(~np.isfinite(R)))
        i, _ = np.unravel_index(k, R.shape)
        raise ValueError(
            f"pearson: correlation is not finite for unit {_unit_name(prof.labels, int(i))}; "
            f"the profile is numerically degenerate"
        )

    R = np.clip(0.5 * (R + R.T), -1.0, 1.0)
    np.fill_diagonal(R, 1.0)
    D = _symmetrise(np.sqrt(np.clip(1.0 - R, 0.0, None)))

    return MetricResult(
        distance=D,
        similarity=R,
        name="pearson",
        is_distance=False,
        n_blocks=prof.n_blocks,
        n_categories=prof.n_categories,
        params={},
        labels=prof.labels,
        unit_grain=unit_grain,
        note="native similarity is r in [-1, 1]; block structure ignored",
    )


def fisher_rao(
    X,
    *,
    tol: float = SIMPLEX_TOL,
    labels: Sequence[str] | None = None,
    unit_grain: str | None = None,
) -> MetricResult:
    """
    Fisher-Rao geodesic distance between probability profiles.

    For block b, ``d_b = 2 * arccos(sum_k sqrt(p_bk * q_bk))``.
    The returned distance is ``sqrt(mean_b(d_b^2))``.

    Parameters:
    ----------

    X : array-like or pd.DataFrame - shape ``(n_units, n_features)`` for one
    block, or ``(n_units, n_blocks, n_categories)``. At least two units and
    non-empty feature/block axes are required. Unit and category order must match.

    tol : float, default SIMPLEX_TOL (1e-6), keyword-only - maximum absolute
    deviation of each block sum from 1. Accepted blocks are renormalized by their sums.

    labels : sequence or None, default None, keyword-only - one label per unit.
    When X is a DataFrame, its index supplies labels if none are passed.
    unit_grain : str or None, default None, keyword-only - description of what
    one profile represents; retained for reporting without changing the calculation.

    Returns:
    -------
    MetricResult - distance in ``[0, pi]`` and similarity equal to negative distance.
    Both matrices are symmetric with zero diagonals, and the block dimensions
    and unit labels accompany the result.

    Raises:
    ------
    ValueError
    - if shapes or label counts are invalid, profiles contain non-finite or negative
    entries, or a block does not have positive mass summing to 1 within tolerance.

    Notes:
    ------
    Blocks receive equal weight in the mean squared geodesic distance. Zero
    categories are allowed and require no pseudocount.
    Raw counts or concatenated compositions must be prepared explicitly before
    calling this function; the simplex check does not infer the intended blocks.
    """
    prof = _prepare(X, metric="fisher_rao", require_simplex=True, tol=tol, labels=labels)
    G = _bhattacharyya(prof.blocks)
    D = _symmetrise(np.sqrt(((2.0 * np.arccos(G)) ** 2).sum(axis=0) / prof.n_blocks))
    return MetricResult(
        distance=D,
        similarity=-D,
        name="fisher_rao",
        is_distance=True,
        n_blocks=prof.n_blocks,
        n_categories=prof.n_categories,
        params={},
        labels=prof.labels,
        unit_grain=unit_grain,
        note="Bhattacharyya family; RMS over blocks; similarity is -distance",
    )


def hellinger(
    X,
    *,
    tol: float = SIMPLEX_TOL,
    labels: Sequence[str] | None = None,
    unit_grain: str | None = None,
) -> MetricResult:
    """
    Hellinger distance between probability profiles.

    For block b, ``h_b^2 = 1 - sum_k sqrt(p_bk * q_bk)``.
    The returned distance is ``sqrt(mean_b(h_b^2))``.

    Parameters:
    ----------

    X : array-like or pd.DataFrame - shape ``(n_units, n_features)`` for one
    block, or ``(n_units, n_blocks, n_categories)``. At least two units and
    non-empty feature/block axes are required. Unit and category order must match.

    tol : float, default SIMPLEX_TOL (1e-6), keyword-only - maximum absolute
    deviation of each block sum from 1. Accepted blocks are renormalized by their sums.

    labels : sequence or None, default None, keyword-only - one label per unit.
    When X is a DataFrame, its index supplies labels if none are passed.
    unit_grain : str or None, default None, keyword-only - description of what
    one profile represents; retained for reporting without changing the calculation.

    Returns:
    -------
    MetricResult - distance in ``[0, 1]`` and similarity equal to negative distance.
    Both matrices are symmetric with zero diagonals, and the block dimensions
    and unit labels accompany the result.

    Raises:
    ------
    ValueError
    - if shapes or label counts are invalid, profiles contain non-finite or negative
    entries, or a block does not have positive mass summing to 1 within tolerance.

    Notes:
    ------
    Zero categories are allowed. Blocks receive equal weight, and a zero distance
    means that all corresponding probability blocks are identical.
    Raw counts or concatenated compositions must be prepared explicitly before
    calling this function; the simplex check does not infer the intended blocks.
    """
    prof = _prepare(X, metric="hellinger", require_simplex=True, tol=tol, labels=labels)
    G = _bhattacharyya(prof.blocks)
    D = _symmetrise(np.sqrt(np.clip((1.0 - G).sum(axis=0), 0.0, None) / prof.n_blocks))
    return MetricResult(
        distance=D,
        similarity=-D,
        name="hellinger",
        is_distance=True,
        n_blocks=prof.n_blocks,
        n_categories=prof.n_categories,
        params={},
        labels=prof.labels,
        unit_grain=unit_grain,
        note="Bhattacharyya family; distance in [0, 1]; similarity is -distance",
    )


def jensen_shannon(
    X,
    *,
    tol: float = SIMPLEX_TOL,
    labels: Sequence[str] | None = None,
    unit_grain: str | None = None,
) -> MetricResult:
    """
    Square-root Jensen-Shannon divergence, averaged over probability blocks.

    For each block, let ``m = (p + q)/2`` and
    ``JS(p, q) = (KL(p||m) + KL(q||m))/2`` using base-2 logarithms.
    The returned distance is ``sqrt(mean_b(JS_b))``.

    Parameters:
    ----------

    X : array-like or pd.DataFrame - shape ``(n_units, n_features)`` for one
    block, or ``(n_units, n_blocks, n_categories)``. At least two units and
    non-empty feature/block axes are required. Unit and category order must match.

    tol : float, default SIMPLEX_TOL (1e-6), keyword-only - maximum absolute
    deviation of each block sum from 1. Accepted blocks are renormalized by their sums.

    labels : sequence or None, default None, keyword-only - one label per unit.
    When X is a DataFrame, its index supplies labels if none are passed.
    unit_grain : str or None, default None, keyword-only - description of what
    one profile represents; retained for reporting without changing the calculation.

    Returns:
    -------
    MetricResult - distance in ``[0, 1]``, similarity equal to negative distance,
    and ``params={"base": 2}`` recording the divergence scale.
    Both matrices are symmetric with zero diagonals, and the block dimensions
    and unit labels accompany the result.

    Raises:
    ------
    ValueError
    - if shapes or label counts are invalid, profiles contain non-finite or negative
    entries, or a block does not have positive mass summing to 1 within tolerance.

    Notes:
    ------
    Zero categories are handled through relative entropy and need no pseudocount.
    The square root is taken after averaging block divergences; the result is not
    the average of the individual block distances.
    Raw counts or concatenated compositions must be prepared explicitly before
    calling this function; the simplex check does not infer the intended blocks.
    """
    prof = _prepare(X, metric="jensen_shannon", require_simplex=True, tol=tol, labels=labels)
    P = prof.blocks
    n = prof.n_units

    D2 = np.zeros((n, n), dtype=float)
    for i in range(n):
        a = P[i][None, :, :]                      # (1, blocks, cats)
        M = 0.5 * (a + P)                         # (n, blocks, cats)
        with np.errstate(invalid="ignore", divide="ignore"):
            kl = rel_entr(a, M).sum(axis=-1) + rel_entr(P, M).sum(axis=-1)  # nats, (n, b)
        D2[i] = 0.5 * kl.mean(axis=1) / _LN2      # bits, averaged over blocks

    D = _symmetrise(np.sqrt(np.clip(D2, 0.0, None)))
    return MetricResult(
        distance=D,
        similarity=-D,
        name="jensen_shannon",
        is_distance=True,
        n_blocks=prof.n_blocks,
        n_categories=prof.n_categories,
        params={"base": 2},
        labels=prof.labels,
        unit_grain=unit_grain,
        note="sqrt of mean per-block JS divergence in bits; similarity is -distance",
    )


def euclidean(
    X,
    *,
    labels: Sequence[str] | None = None,
    unit_grain: str | None = None,
) -> MetricResult:
    """
    Raw-profile Euclidean distance on a root-mean-square block scale.

        distance(i, j) = sqrt(sum_b sum_k (x_ibk - x_jbk)^2 / n_blocks)

    Parameters:
    ----------

    X : array-like or pd.DataFrame - shape ``(n_units, n_features)`` for one
    block, or ``(n_units, n_blocks, n_categories)``. At least two units and
    non-empty feature/block axes are required. Unit and category order must match.

    labels : sequence or None, default None, keyword-only - one label per unit.
    When X is a DataFrame, its index supplies labels if none are passed.
    unit_grain : str or None, default None, keyword-only - description of what
    one profile represents; retained for reporting without changing the calculation.

    Returns:
    -------
    MetricResult - non-negative distance and its negative as similarity, both
    with zero diagonals. A two-dimensional input uses ordinary Euclidean distance.

    Raises:
    ------
    ValueError
    - if dimensions or label counts are invalid, profiles are non-finite, or
    the computed matrix fails MetricResult validation.

    Notes:
    ------
    The input is neither standardized nor normalized. Negative values are allowed,
    and feature units and abundance scale affect the answer. Blocks are averaged
    through squared distance, while categories within a block are summed.
    """
    prof = _prepare(X, metric="euclidean", require_simplex=False, labels=labels)
    D = _symmetrise(np.sqrt(_sq_distances(prof.flat) / prof.n_blocks))
    return MetricResult(
        distance=D,
        similarity=-D,
        name="euclidean",
        is_distance=True,
        n_blocks=prof.n_blocks,
        n_categories=prof.n_categories,
        params={},
        labels=prof.labels,
        unit_grain=unit_grain,
        note="raw-scale distance, RMS over blocks; similarity is -distance",
    )


def aitchison(
    X,
    eps: float = 0.0,
    per_block: bool = True,
    *,
    tol: float = SIMPLEX_TOL,
    labels: Sequence[str] | None = None,
    unit_grain: str | None = None,
) -> MetricResult:
    """
    Aitchison distance between centered log-ratio coordinates.

    For a positive composition q, ``clr(q)_k = log(q_k) - mean(log(q))``.
    The calculation applies the selected centering convention and returns
    Euclidean distance between the coordinates divided by ``sqrt(n_blocks)``.

    Parameters:
    ----------

    X : array-like or pd.DataFrame - shape ``(n_units, n_features)`` for one
    block, or ``(n_units, n_blocks, n_categories)``. At least two units and
    non-empty feature/block axes are required. Unit and category order must match.

    eps : float, default 0.0 - finite non-negative pseudocount added to every
    probability cell, followed by within-block normalization. With eps=0,
    any zero cell raises because its log-ratio is undefined.
    per_block : bool, default True - center log values within each block before
    concatenation. False centers once over all concatenated log values, after
    the same within-block normalization.

    tol : float, default SIMPLEX_TOL (1e-6), keyword-only - maximum absolute
    deviation of each block sum from 1. Accepted blocks are renormalized by their sums.

    labels : sequence or None, default None, keyword-only - one label per unit.
    When X is a DataFrame, its index supplies labels if none are passed.
    unit_grain : str or None, default None, keyword-only - description of what
    one profile represents; retained for reporting without changing the calculation.

    Returns:
    -------
    MetricResult - non-negative distance and similarity equal to its negative,
    with ``eps`` and ``per_block`` retained in ``params``.

    Raises:
    ------
    ValueError
    - if profile or simplex validation fails, eps is negative or non-finite,
    or zero cells occur with eps=0.

    Notes:
    ------
    The pseudocount is on the normalized probability scale, not the raw-count
    scale. It changes the answer and has no universally appropriate positive
    default. Use :func:`aitchison_pseudocount_sweep` to report sensitivity to
    defensible choices. The logarithm is natural, so distance scale follows that
    choice; distances are not bounded like Hellinger or Jensen-Shannon distance.
    """
    prof = _prepare(X, metric="aitchison", require_simplex=True, tol=tol, labels=labels)
    P = prof.blocks

    if not np.isfinite(eps) or eps < 0.0:
        raise ValueError(f"aitchison: eps must be finite and non-negative, got {eps!r}")

    if eps == 0.0:
        zero = P <= 0.0
        if zero.any():
            per_unit = zero.reshape(prof.n_units, -1).sum(axis=1)
            worst = int(np.argmax(per_unit))
            raise ValueError(
                f"aitchison: log-ratio geometry is undefined on zero cells and eps=0. "
                f"{int(zero.sum())} of {P.size} cells are zero; the worst unit is "
                f"{_unit_name(prof.labels, worst)} with {int(per_unit[worst])} zeros. "
                f"Choose a pseudocount deliberately and pass it as eps=... -- it is a "
                f"modelling choice with consequences, so sweep it with "
                f"aitchison_pseudocount_sweep() and report the range, do not adopt a "
                f"default."
            )
        Q = P
    else:
        Q = P + eps

    Q = Q / Q.sum(axis=-1, keepdims=True)
    L = np.log(Q)
    if per_block:
        Z = (L - L.mean(axis=-1, keepdims=True)).reshape(prof.n_units, -1)
    else:
        Lf = L.reshape(prof.n_units, -1)
        Z = Lf - Lf.mean(axis=1, keepdims=True)

    D = _symmetrise(np.sqrt(_sq_distances(Z) / prof.n_blocks))
    return MetricResult(
        distance=D,
        similarity=-D,
        name="aitchison",
        is_distance=True,
        n_blocks=prof.n_blocks,
        n_categories=prof.n_categories,
        params={"eps": float(eps), "per_block": bool(per_block)},
        labels=prof.labels,
        unit_grain=unit_grain,
        note=(
            "clr geometry; the eps in params changes the answer -- quote a range, "
            "not a point"
        ),
    )

METRICS: dict[str, Callable[..., MetricResult]] = {
    "pearson": pearson,
    "fisher_rao": fisher_rao,
    "hellinger": hellinger,
    "jensen_shannon": jensen_shannon,
    "euclidean": euclidean,
    "aitchison": aitchison,
}


def _resolve_metrics(
    metrics: Sequence[str | Callable[..., MetricResult]]
    | Mapping[str, Callable[..., MetricResult]]
    | None,
) -> dict[str, Callable[..., MetricResult]]:
    """
    Resolve registry names or caller-provided functions into a named metric panel.

    None selects all built-ins. A sequence accepts names or callables; a mapping
    provides explicit display names. Unknown registry names raise KeyError, and
    an empty selection raises ValueError.
    """
    if metrics is None:
        return dict(METRICS)
    if isinstance(metrics, Mapping):
        out = dict(metrics)
    else:
        out = {}
        for m in metrics:
            if callable(m):
                out[getattr(m, "__name__", f"metric_{len(out)}")] = m
            elif m in METRICS:
                out[str(m)] = METRICS[str(m)]
            else:
                raise KeyError(f"unknown metric {m!r}; known metrics: {sorted(METRICS)}")
    if not out:
        raise ValueError("no metrics selected")
    return out


def _accepts_labels(fn: Callable[..., MetricResult], labels: Sequence[str] | None) -> bool:
    """
    Determine whether a supplied labels argument can be forwarded to a metric.

    Requires non-None labels and a signature accepting ``labels`` or arbitrary
    keyword arguments. Uninspectable signatures return False.
    """
    if labels is None:
        return False
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):  # builtins, C callables
        return False
    return "labels" in params or any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
    )

def metric_agreement(
    X,
    metrics: Sequence[str | Callable[..., MetricResult]]
    | Mapping[str, Callable[..., MetricResult]]
    | None = None,
    *,
    which: str = "distance",
    labels: Sequence[str] | None = None,
) -> pd.DataFrame:
    """
    Compare geometries by Spearman correlation over unordered unit pairs.

    Every metric evaluates the same input profiles. The upper-triangle pair vectors
    are ranked with average ranks for ties, then correlated. This removes the raw
    scale difference between geometries while retaining disagreement in pair ordering.

    Parameters:
    ----------
    X : array-like or pd.DataFrame - profiles accepted by every selected metric,
    with identical unit and feature order throughout the panel.
    metrics : sequence, mapping, or None, default None - all ``METRICS`` when
    None; otherwise registry names, callables, or a name-to-callable mapping.
    Use a configured callable when a metric needs non-default parameters.
    which : str, default "distance", keyword-only - compare distance or similarity
    pair vectors consistently across the panel.
    labels : sequence or None, default None, keyword-only - labels forwarded only
    to metric functions whose signatures accept them.

    Returns:
    -------
    pd.DataFrame, shape (n_metrics, n_metrics) - Spearman correlations, indexed
    and columned by metric name. A constant pair vector gives an undefined row
    and column, including its diagonal. Attributes record unit and pair counts,
    selection, orientation, and degenerate metrics.

    Raises:
    ------
    KeyError
    - if a registry name is unknown.
    TypeError
    - if a callable returns something other than MetricResult.
    ValueError
    - if the selection or orientation is invalid, metrics return different unit
    counts, or a selected metric rejects the profiles.

    Notes:
    ------
    The default panel includes Aitchison with ``eps=0`` and therefore rejects
    zero cells. Configure its pseudocount deliberately for sparse compositions.
    Custom callables must preserve unit order; matching counts alone do not verify it.
    Switching all built-ins from distance to similarity reverses all pair ranks
    together and leaves their Spearman agreement unchanged, up to numerical effects.
    Agreement is empirical and does not prove robustness of every downstream result.
    """
    if which not in ("distance", "similarity"):
        raise ValueError(f"which must be 'distance' or 'similarity', got {which!r}")

    chosen = _resolve_metrics(metrics)
    vectors: dict[str, np.ndarray] = {}
    n_units = 0
    for name, fn in chosen.items():
        res = fn(X, labels=labels) if _accepts_labels(fn, labels) else fn(X)
        if not isinstance(res, MetricResult):
            raise TypeError(f"metric {name!r} returned {type(res).__name__}, not MetricResult")
        if n_units and res.n_units != n_units:
            raise ValueError(
                f"metric {name!r} returned {res.n_units} units but an earlier metric in "
                f"the panel returned {n_units}; the pair vectors are not comparable"
            )
        vectors[name] = res.pairs(which)
        n_units = res.n_units

    names = list(vectors)
    ranks = np.vstack([rankdata(vectors[n]) for n in names])
    degenerate = [n for n, r in zip(names, ranks) if np.ptp(r) == 0.0]

    rho = np.full((len(names), len(names)), np.nan)
    live = [i for i, n in enumerate(names) if n not in degenerate]
    if live:
        with np.errstate(invalid="ignore", divide="ignore"):
            sub = np.corrcoef(ranks[live])
        sub = np.atleast_2d(sub)
        for a, i in enumerate(live):
            for b, j in enumerate(live):
                rho[i, j] = sub[a, b]
        np.fill_diagonal(rho, 1.0)
    for i, n in enumerate(names):
        if n in degenerate:
            rho[i, i] = np.nan

    out = pd.DataFrame(rho, index=names, columns=names)
    out.attrs.update(
        {
            "n_pairs": int(len(next(iter(vectors.values())))),
            "n_units": int(n_units),
            "which": which,
            "metrics": names,
            "degenerate": degenerate,
            "note": (
                "Spearman correlation over unordered unit pairs. "
                "Several simplex-native metrics may agree strongly because "
                "they respond to related distributional geometry, but such "
                "agreement is empirical rather than guaranteed. Robustness "
                "should therefore be evaluated from the observed agreement "
                "and downstream conclusions."
        ),
        }
    )
    return out

def aitchison_pseudocount_sweep(
    X,
    eps_grid: Sequence[float],
    scorer: Callable[[MetricResult], float | Mapping[str, float]],
    *,
    per_block: bool | Sequence[bool] = (True, False),
    tol: float = SIMPLEX_TOL,
    labels: Sequence[str] | None = None,
    unit_grain: str | None = None,
    skip_invalid: bool = False,
) -> pd.DataFrame:
    """
    Score Aitchison geometry across explicit pseudocount and centering choices.

    The pseudocount changes the log-ratios, especially near zero. This sweep makes
    that modeling choice part of the reported sensitivity analysis.

    Parameters:
    ----------
    X : array-like or pd.DataFrame - probability blocks accepted by :func:`aitchison`.
    eps_grid : sequence of float - non-empty finite, non-negative pseudocount grid.
    scorer : callable - accepts a MetricResult and returns a scalar or a mapping
    of named scores. Fix any internal randomness when comparing settings.
    per_block : bool or sequence of bool, default (True, False), keyword-only -
    centering conventions crossed with every pseudocount.
    tol : float, default SIMPLEX_TOL, keyword-only - absolute simplex tolerance.
    labels : sequence or None, default None, keyword-only - unit labels.
    unit_grain : str or None, default None, keyword-only - definition of the unit axis.
    skip_invalid : bool, default False, keyword-only - record ValueError failures
    from individual Aitchison evaluations in ``metric_error`` and continue.

    Returns:
    -------
    pd.DataFrame - one row per setting, with ``eps``, ``per_block``,
    ``n_zero_cells``, and either ``score`` or the scorer's named fields.
    Failed metric settings have ``metric_error`` and missing score entries.
    With ``skip_invalid=True``, successful rows also contain an empty ``error``
    field. Attributes record the zero-cell count and reporting note.

    Raises:
    ------
    ValueError
    - if either grid is empty, pseudocounts are invalid, or initial profile
    validation fails. Per-setting metric ValueErrors propagate unless skipped.

    Notes:
    ------
    Scorer exceptions always propagate; ``skip_invalid`` applies only to the
    individual metric evaluation. Initial input validation also always raises.
    Use score names distinct from the setting columns so reporting context is not
    overwritten. Report the score range across defensible settings, including any
    invalid settings, rather than choosing a pseudocount because it gives a desired result.
    """
    grid = [float(e) for e in eps_grid]
    if not grid:
        raise ValueError("eps_grid is empty")
    if any((not math.isfinite(e)) or e < 0.0 for e in grid):
        raise ValueError(f"eps_grid must be finite and non-negative, got {list(eps_grid)}")

    flags = [bool(per_block)] if isinstance(per_block, bool) else [bool(b) for b in per_block]
    if not flags:
        raise ValueError("per_block is empty")

    prof = _prepare(X, metric="aitchison", require_simplex=True, tol=tol, labels=labels)
    n_zero = int((prof.blocks <= 0.0).sum())

    rows: list[dict[str, object]] = []
    for eps in grid:
        for pb in flags:
            row: dict[str, object] = {"eps": eps, "per_block": pb, "n_zero_cells": n_zero}
            try:
                res = aitchison(
                    prof.blocks,
                    eps=eps,
                    per_block=pb,
                    tol=tol,
                    labels=prof.labels,
                    unit_grain=unit_grain,
                )
            except ValueError as exc:
                if not skip_invalid:
                    raise

                row["metric_error"] = f"{type(exc).__name__}: {exc}"
                rows.append(row)
                continue
            score = scorer(res)
            if isinstance(score, Mapping):
                row.update({str(k): v for k, v in score.items()})
            else:
                row["score"] = float(score)
            if skip_invalid:
                row["error"] = ""
            rows.append(row)

    out = pd.DataFrame(rows)
    if "error" in out.columns:
        out = out[[c for c in out.columns if c != "error"] + ["error"]]
    out.attrs["note"] = (
        "report the RANGE of each score over eps, not its value at a default; a score "
        "that moves materially across the grid is a pseudocount artifact"
    )
    out.attrs["n_zero_cells"] = n_zero
    return out
