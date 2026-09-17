"""
Provenance calibration for within-group similarity contrasts.

The module answers the following. Consider units that all came from a single source (one lab, one protocol, one deposit) 
and partition them at random into two groups of the published sizes. How often does a meaningless split produce
a within-group similarity gap as large as the published one?

If it is "often", the published contrast lies inside the range that provenance alone supplies. Then, it can be 
deduced that the contrast is aliased with provenance, and this module states that as a calibration result. The result 
depends on no model of the measurement. Its only inputs are a similarity matrix over units and the group sizes. 

We caution that this is not a test of whether the two published groups differ. The null hypothesis concerns
the design, NOT the biology: within one source, partitioning alone can reach the published gap this often. The procedure
is one-sided by design, because the statistic is an absolute value.

Statistic:
----------
For a set ``X`` of ``m`` units and a similarity matrix of ``S``, the within-group mean is the average off-diagonal
similarity among its members:

r_X = sum over i, j in X with i != j of S[i, j] / (m * (m - 1))

The dispersion gap of a split into sides A and B is ``r_A - r_B``. Each side needs at least 2 members, and splits 
with a smaller side gives ``nan``. The diagonal of ``S`` never contributes. ``S`` may hold similarities or distances,
but the orientataion changes how the sign reads.

Workflow:
---------
1. Build the similarity matrix over units and write down how it was built. That description is the ``basis``, and 
every result must carry it. A result with ``basis = "unspecified"`` is flagged as not reportable.
2. Score the published contrast with ``dispersion_gap(sim, idx_a, idx_b)``
3. Build the null with ``arbitrary_split_null``
4. Read the calibration from :class:`NullResult` ``p_value(gap)``, ``population_p_is_exact(gap)``, ``ratio_to(gap)``, ``summary()``
5. Check which way that p errs with ``k_diagnostic(sim, idx_a, idx_b, break_even = ...)``. The :class:`KResult` says whether
p is an upper bound, a lower bound, or undetermined
6. Answer "you would get this by splitting on anything" by scoring splits for label-defined sides (``named_splits``) and for
every tie-break of a median split on a covariate, e.g. pool size or sample count (``median_split_indices``)

Public API:
-----------
dispersion_gap(sim, idx_a, idx_b, *, signed = False)
    gap for one split. absolute by default
dispersion_gap_batch(sim, masks)
    absolute gaps for many complete partitions given as boolean masks.
n_partitions(n, n_a)
    size of the partition space, counting an equal-size partition and its
    complement once.
arbitrary_split_null(sim, n_a, ...) -> NullResult
    the null distribution of the absolute gap. enumerated exhaustively when
    the space is small enough, otherwise sampled as distinct partitions
    without replacement.
NullResult
    streaming summary of the null: exact maximum, exact registered tail
    counts, histogram quantiles, p-values and their bounds, and reporting
    helpers.
k_diagnostic(sim, idx_a, idx_b, *, break_even, basis) -> KResult
    direction-of-error diagnostic for the calibration p.
KBreakEven, KResult
    treak-even thresholds for ``k``, and the diagnostic verdict.
named_split(sim, idx_by_group, ordering)
    signed gap of a split defined by named sides.
median_split_indices(values, n_a=None, ...)
    every median split of a covariate, with all tie-breaks at the cut.

Sign conventions:
-----------------
- ``dispersion_gap`` defaults to absolute value
- ``dispersion_gap_batch`` and the null are absolute. the null is a distribution of ``|gap|`` because an 
arbitrary split has no privileged sign. exhaustive enumeration uses this, when two sides are the same size, 
unit 0 is pinned to side A, such that each partition is scored once.
- ``named_split`` is signed, because a named split does have sides.
- ``ordering`` fixes which side is A and the result is ``r_A - r_B``. 
- Compare ``abs(named_split(...))`` against the null.

Exactness:
----------
Exact is defined as reaching up to the ``_REACH_RTOL`` slack. A null statistic counts as reaching an observed value ``t`` when it is
``>= _reach_floor(t)``. The slack absorbs round-off between the single-split and the batch code paths, which differ by a few
ULPs in either direction
- tail count : exact for registered or kept values. for anything else, the histogram gives bounds.
- p over the whole partition space : exact only when the null is exhaustive and the count is exact. a sampled null gives a Monte Carlo estimate,
which can be 0, and its note then reports a rule-of-three bound
-maximum : exact among the evaluated partitions. for a sampled null that is a lower bound on the maximum over the whole space
- quantiles : accurate to one histogram bin, and never below the true value. 

Every numerical approximation errs towards a larger p. 

Direction of calibration error:
-------------------------------
The arbitrary-split null ignores the uneven similarity spread within each published group, so its p can err either way

k = SD(centrality in B)/SD(centrality in A)

where a unit's centrality is its mean similarity to other members of its own group, and A is the group the null was built around. ``k`` 
with break-even values derived for the observed group sizes:
1. ``k < k*_population`` : the null is too wide, and p is an upper bound
2. ``k > k*_conditional`` : the null is too narrow, and p is a lower bound 
3. otherwise, the direction is undetermined

Requirements:
-------------
1. ``sim`` must be square, finite, and symmetric to within 1e-9. 
2. Unit indices can be integer arrays or boolean masks of length ``n``. Duplicate indices, negative indices, and overlapping sides are rejected.
3. Each size of any scored split needs at least 2 members. the null needs at least 4 units total.
"""
from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Hashable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
import numpy as np

_VERDICTS = ("conservative", "undetermined", "anti-conservative")

#relative slack used when asking whether a null statistic reaches an observed value.
#the null's statistics and the observed one are produced by different code paths
_REACH_RTOL : float = 1e-12

def _reach_floor(t : float) -> float:
    """
    Lowest statistic that still counts as reaching the threshold ``t``. 

    Used when counting how many null statistics are at least as extreme as an observed statistic ``t``.
    A null value ``s`` counts as reaching ``t`` if ``s >= _reach_floor(t)`` instead of the exact comparison
    of ``s >= t``. The slack allows for floating point round-off. 

    The floor is 
    floor = t - _REACH_RTOL * max(1.0, |t|)

    so the tolerance is relative (``_REACH_RTOL * |t|``) when ``|t| >= 1`` and absolute ``_REACH_RTOL`` when
    ``|t| < 1``. The absolute part matters near zero. With a purely relative tolerance, a threshold of exactly
    zero would get no slack, and null statistics that are 0 up to round-off would be rejected. With the floor
    at ``-_REACH_RTOL``, every null statistic that cannot be negative is admitted. 

    Parameters:
    -----------
    t : float - ``t`` lowered by the tolerance. For finite ``t``, the result is always ``<= t``, including when
    ``t`` is negative.

    Notes:
    ------
    The slack is intentionally conservative. Lowering the floor can only add null statistics to the "at least extreme" 
    count, so a permutation-style p-value can only increase. Larger statistics are assumed to be more extreme. 
    For a lower-tail test, negative the statistics before calling this function. Non-finite input is not guarded. It is
    recommended to validate ``t`` prior to calling this function.
    """
    t = float(t)
    return t - _REACH_RTOL * max(1.0, abs(t))

def _check_sim(sim, *, min_n : int = 4) -> np.ndarray:
    """
    Validate a unit-by-unit similarity or distance matrix and return it as a float64 array. 
    
    Checks for routinese that splits the units (the rows and columns of ``sim``) into two sides and scores the split. 
    Checks the run the order below.
    1. shape: ``sim`` must be 2D and square.
    2. size: ``n >= min_n``. The default of 4 is the smallest ``n`` that allows a two-sided split with at least
    two units per side, which gives each side at least one within-side pair.
    3. finiteness: no ``nan``, ``+inf``, or ``-inf`` anywhere.
    4. symmetry: ``|S[i, j] - S[j, i]| <= 1e-9`` for all ``i, j``. 

    Parameters:
    -----------
    sim : array_like, shape (n, n) - pairwise similarity (correlation, cosine) or distance matrix over ``n`` units.
    min_n : int, default 4, keyword-only - minimum number of units required. raise if needed.

    Returns:
    --------
    S : np.ndarray float64, shape (n, n) - ``sim`` as float64.

    Raises:
    -------
    ValueError if any of the check conditions fail. Errors describe the failure. 
    """
    S = np.asarray(sim, dtype = float)
    if S.ndim != 2 or S.shape[0] != S.shape[1]:
        raise ValueError(f'sim must be square matrix over units. received shape {S.shape}')
    if S.shape[0] < min_n:
        raise ValueError(f"sim has {S.shape[0]} units, at least {min_n} are required for a two-sided split with two members per side.")
    if not np.isfinite(S).all():
        raise ValueError("sim contains non-finite entries. A NaN usually means a unit with zero variance in the feature matrix.")
    if not np.allclose(S, S.T, rtol = 0.0, atol = 1e-9):
        raise ValueError('sim must be symmetric. asymmetry above atol = 1e-9')
    return S

def _zero_diagonal(S : np.ndarray) -> np.ndarray:
    """
    Return a copy of ``S`` with the main diagonal set to 0.

    Zeroing the diagonal removes self-pairs from block sums. After this, 
    ``S0[np.ix_(idx, idx)].sum()`` adds up only the entries with ``i != j``, so within-set and between-set statistics can use 
    ``.sum()`` without masking self-pairs each time. For a similarity matrix, this removes self-similarities (1.0) which would
    push within-set means up. For a distance matrix, the diagonal is normally 0 already, and clears any floating-point noise.

    Parameters:
    -----------
    S  : np.ndarray, shape (n, n) - square symmetric similarity or distance matrix, usually the output of ``_check_sim``.

    Returns:
    --------
    S0 : np.ndarray, shape (n, n) - copy of ``S`` with the diagonal set to 0.0. 

    Notes:
    ------
    ``S`` is copied and never changed. Breaks the memory sharing that ``_check_sim`` can leave between its return value and the caller's array.
    The cost is one extra ``n x n`` array. Build ``S0`` once per matrix and reuse it for every split, not once per split.

    The diagonal is set to 0, not masked out. A sum over a block still counts every unordered pair twice, as ``(i, j)`` and ``(j, i)``. Callers
    have to use ordered-pair denominators, as ``_mean_within`` does with ``m * (m - 1)``. 
    """
    S0 = S.copy()
    np.fill_diagonal(S0, 0.0)
    return S0

def _as_index(idx, n : int, name : str) -> np.ndarray:
    """
    Normalize a unit selection to an array of integer indices into ``range(n)``. 

    Two input forms are accepted:
    1. Boolean mask : must have shape exactly ``(n,)``. returns the position of the True entries in ascending order.
    2. Integer indices : any array-like, including a scalar. it is cast to ``np.intp`` and flattened in row-major order. every
    index must satisfy ``0 <= i < n``, and no index may repeat. the input order is kept, not sorted.

    Parameters:
    -----------
    idx : array_like of bool or int - the selected units, either as boolean mask or as integer indices.
    n : int - total number of units (``S.shape[0]``). sets the valid index range and the required mask length.
    name : str - argument name shown in error messages, e.g. ``side_a``. 

    Returns: 
    --------
    np.ndarray of intp, shape (k,) - distinct integer indices into ``range(n).`` ``k`` can also be 0 (empty list; mask with no True entries)

    Raises:
    -------
    ValueError 
    - if boolean mask has wrong shape
    - if an integer index is negative or >= n
    - if an integer index appears more than once
    - passed up from ``np.asarray`` when ``idx`` cannot be cast to integers, e.g. a Python list containing ``nan`` or non-numerics.

    Notes:
    ------
    Only an array with ``dtype == bool`` takes the mask path. Everything else goes through integer casting.

    Raise an issue in github if this somehow becomes an problem. I am not fixing smth that is not broken yet...
    """
    a = np.asarray(idx)
    if a.dtype == bool:
        if a.shape != (n,):
            raise ValueError(f"{name} : bool mask has length {a.size}, expected {n}")
        return np.flatnonzero(a)
    a = np.asarray(idx, dtype=np.intp).ravel()
    if a.size:
        if a.min() < 0 or a.max() >= n:
            raise ValueError(f"{name}: index out of range for {n} units")
        if np.unique(a).size != a.size:
            raise ValueError(f"{name}: duplicate indices")
    return a

def _mean_within(S0 : np.ndarray, idx : np.ndarray) -> float:
    """
    Mean off-diagonal similarity (or distance) among the units one index set.

    For a set ``I`` with ``m = |I|``, this computes the 

    sum over i, j in I with i != j of S0[i,j] / (m * (m - 1))

    which is the average over all ordered pairs of distinct members. When ``S0`` is symmetric, this is the average over 
    unordered pairs, ``sum_{i < j} S0[i, j] / C(m, 2)``. 

    Parameters:
    -----------
    S0 : np.ndarray, shape (n, n) - similarity of distance matrix with a zero diagonal, i.e. the output of ``_zero_diagonal.`` 
    The diagonal is not checked here. with a non-zero diagonal, the block sum includes self-pairs, and the result is off by 
    tr(block) / (m * (m - 1)). for a similarity matrix, with diagonal of 1, that adds ``1 / (m - 1)``. 
    idx : np.ndarray of int, shape (m, ) - distint integer indices into ``S0``, i.e. the output of ``_as_index``. 

    Returns:
    --------
    float - within-set mean, or ``nan`` when ``m < 2``

    Notes:
    ------
    callers must handle the ``nan`` case. 

    pass integer indices, not a boolean mask, ``np.ix_`` does accept a mask, but ``idx.size`` would be ``n`` instead of the number of
    selected units. the sum would be right while the denominator is wrong, and no error would be raised. duplicate indices are also
    wrong here but not checked, since they add zereoed diagonal entries to the block and count their pairs more than once. 
    """
    m = idx.size
    if m < 2:
        return float('nan')
    block = S0[np.ix_(idx, idx)]
    return float(block.sum() / (m * (m - 1)))

def dispersion_gap(sim, idx_a , idx_b, *, signed : bool = False) -> float:
    """
    Difference in mean within-side similarity between two disjoint sets of units.

    For a side ``X`` with ``m_X`` members, the within-side mean is 

    r_X = sum over i, j in X with i != j of S[i, j] / (m_X * (m_X - 1))

    which is the average over ordered pairs of distinct members. For a symmetric ``S`` 
    it is the same as the average over unordered pairs. The function returns ``|r_A - r_B|`` or
    ``r_A - r_B`` depending on the ``signed`` argument. The result whether one side is more internally
    cohesive (similarity input) or more spread out (distance input) than the other.

    Parameters:
    -----------
    sim : array_like, shape (n, n) - similiarity or distance matrix over ``n`` units. validated by ``_check_sim``. the diagonal is ignored.
    idx_a, idx_b : array_like of int or bool - members of side A and sie B, given as integer indices or as boolean masks of length
    ``n``. normalized by ``_as_index``, so no duplicates, no negative indices, and floats are truncated. the two sides must be disjoint. 
    signed : bool, default False - if true, returns ``r_A - r_B`` instead of absolute value

    Returns:
    --------
    float - the gap, it is ``nan`` if either side has fewer than 2 members, sine that side has no within-side pairs to average. ``_check_sim``
    rejects non-finite entries, so this is the only way to get ``nan``.

    Raises:
    -------
    ValueError
    - if non-square input
    - if wrong mask shape, index out of range, duplicate  indices
    - if ``idx_a`` and ``idx_b`` share a unit

    Notes:
    ------
    with ``n = 2`` or ``n = 3`` the matrix passes validation, but the rewsult is always ``nan``. two sides with at least 2 members each need
    ``n >= 4``. 
    """
    S = _check_sim(sim, min_n=2)
    n = S.shape[0]
    a = _as_index(idx_a, n, "idx_a")
    b = _as_index(idx_b, n, "idx_b")
    if np.intersect1d(a, b).size:
        raise ValueError("idx_a and idx_b overlap; the two sides of a split are disjoint")
    S0 = _zero_diagonal(S)
    gap = _mean_within(S0, a) - _mean_within(S0, b)
    return float(gap) if signed else float(abs(gap))

def dispersion_gap_batch(sim, masks) -> np.ndarray:
    """
    Absolute dispersion gap for many complete two-sided partitions at once.

    For row ``b`` of ``masks,`` side A is the units where ``masks[b]`` is True and side B is all the others. Up to round-off, the output is the same
    ``dispersion_gap(sim, masks[b], ~masks[b])``, computed for every row. this is the complete-partition case that the null enumeration in 
    ``_enumerate_masks`` computes.

    Parameters:
    -----------
    sim : array_like, shape (n, n) - similarity or distance matrix over ``n`` units. validated by ``_check_sim``. the diagonal is ignored.
    masks : array_like of bool, shape (B, n) or (n, ) - one partition per row, with True meaning side A and False meaning side B. A 1D mask 
    is treated as a single row. the input is converted with ``np.asarray(masks, dtype = bool)``, so any non-zero value becomes True. 0/1 integer masks work, but
    integer indices are not recognized.

    Returns:
    -------- 
    np.ndarray of float64, shape (B, ) - ``|r_A - r_B|`` for each row, or ``nan`` in rows where either side has fewer than 2 members.

    Raises:
    -------
    ValueError
    - if non-square input, ``n < 2``, non-finite entries, or asymmetry
    - if ``masks``, after 1D promotion, is not 2D with ``n`` columns.

    Notes:
    ------
    the result is the absolute value, and no ``signed`` option. a row and its complement gives the same value. only complete partitions are supported, because
    every unit is either on side A or on side B. for partial splits, call :func:`dispersion_gap`.
    """
    S = _check_sim(sim, min_n=2)
    n = S.shape[0]
    M = np.asarray(masks, dtype=bool)
    if M.ndim == 1:
        M = M[None, :]
    if M.ndim != 2 or M.shape[1] != n:
        raise ValueError(f"masks must have shape (B, {n}), got {M.shape}")
    return _gap_batch(_zero_diagonal(S), M)

def _gap_batch(S0: np.ndarray, M: np.ndarray) -> np.ndarray:
    """
    Unchecked core of :func:`dispersion_gap_batch`

    Each row of ``M`` becomes a 0/1 float vector ``a``, and its complement is ``c = 1 - a``. the within-side sums are quadratic forms 
    ``a^T S0 a`` and ``c^T S0 c``. because the diagonal of ``S0`` is zero, these sums include only pairs of distinct units on the same side.

    returns,
    |a^T S0 a / (nA (nA - 1)) - c^T S0 c / (nB (nB - 1))|
    for each row, where ``nA = sum(a)`` and ``nB = n - nA``

    Parameters:
    -----------
    S0 : np.ndarray, shape (n, n) - float matrix whose diagonal is already zero
    M : np.ndarray, bool, shape (B, n) - partition masks, with True meaning side A, must be 2D with ``M.shape[1] == S0.shape[0]``

    Returns:
    --------
    np.ndarray of float64, shape (B,) - absolute gap per row, or ``nan`` where either side has fewer than 2 members.
    """
    n = S0.shape[0]
    A = M.astype(np.float64)
    Bm = 1.0 - A
    nA = A.sum(axis=1)
    nB = n - nA
    with np.errstate(invalid="ignore", divide="ignore"):
        rA = ((A @ S0) * A).sum(axis=1) / (nA * (nA - 1.0))
        rB = ((Bm @ S0) * Bm).sum(axis=1) / (nB * (nB - 1.0))
        out = np.abs(rA - rB)
    out[(nA < 2) | (nB < 2)] = np.nan
    return out

def n_partitions(n : int, n_a : int) -> int:
    """
    Number of distinct ways to split ``n`` units into two sides of sizes ``n_a`` and ``n - n_a``

    the sides are unordered. when ``n_a != n - n_a``, the sizes tell the two sides apart, so the count is ``C(n, n_a)``. 

    Parameters:
    -----------
    n : int - total number of units.
    n_a : size of one side.

    Returns:
    --------
    int : exact partition count as an int, with no overflow. it makes the total number of rows yielded by ``_enumerate_masks(n, n_a, chunk)``
    for any positive ``chunk``. it is 0 when ``n_a > n``

    Raises:
    -------
    ValueError
    - if ``n`` or ``n_a`` is negative 
    TypeError
    - if ``n`` or ``n_a`` is not an integer.

    Notes:
    ------
    treating the sides as unordered is correct only for statistics that do not change when the sides are swapped, e.g. absolute gap.
    """
    c = math.comb(n, n_a)
    return c // 2 if 2 * n_a == n else c

def _enumerate_masks(n : int, n_a : int, chunk : int) -> Iterator[np.ndarray]:
    """
    Yield every distinct partition of ``n`` units into sizes of ``n_a`` and ``n-n_a`` exactly once, as chunks of boolean masks.

    Each yielded array has the shape ``(k,n)`` and dtype bool, with True meaning side A. this matches the ``masks`` convention of 
    :func:`dispersion_gap_batch` and :func:`_gap_batch`. every chunk except possibly the lasst has ``k == chunk``. no chunk is empty, and 
    no rows are discarded. over all chunks, the total number of rows equals ``n_partitions(n, n_a)``. 

    rows follow the lexicographic order of the side-A index tuples produced by ``itertools.combinations``, so the order is deterministic. 

    pinning for equal-size checks:
    - when ``2 * n_a == n``, a mask and its complement describe the same partition and gives the same absolute statistic. unit 0 is therefore
    pinned to side A, and only the other ``n - 1`` units are enumerated. that gives, ``C(n - 1, n_a - 1) = C(n, n_a) / 2`` masks. without the pin, 
    every partition would appear twice. quantiles would not change, and numerator and denominator of every tail count would double, so ``count/total`` p-value
    would stay the same. 

    the pin is only valid for statistics that do not change when the sides are swapped, e.g. absolute gap.

    Parameters:
    -----------
    n : int - total number of units, i.e. mask width
    n_a : int - size of side A
    chunk : int - maximum number of rows per yielded array.

    Returns:
    --------
    np.ndarray of bool, shape (k, n), 1 <= k <= chunk - partition masks with True meaning side A. when pinned, column 0 is True in every row. 

    Raises:
    -------
    ValueError
    - if ``n_a < 0``
    - if ``chunk < 0``
    - if ``n = n_a = 0``
    """
    pin = 2 * n_a == n
    it = (
        itertools.combinations(range(1, n), n_a - 1)
        if pin
        else itertools.combinations(range(n), n_a)
    )
    while True:
        block = list(itertools.islice(it, chunk))
        if not block:
            return
        M = np.zeros((len(block), n), dtype=bool)
        np.put_along_axis(M, np.asarray(block, dtype=np.intp), True, axis=1)
        if pin:
            M[:, 0] = True
        yield M

def _draw_masks(n: int, n_a: int, B: int, rng: np.random.Generator) -> np.ndarray:
    """
    Draw ``B`` uniformly random side-A masks of size ``n_a``, with replacement across rows.

    Each row ranks ``n`` uniform draws with argsort and puts the ``n_a``
    lowest-ranked units on side A. That gives a uniform random subset per
    row. Different rows are independent, so duplicate rows are possible.

    Parameters:
    ----------
    n : int - mask width, i.e. the number of units.
    n_a : int - number of True entries per row. Not validated: ``n_a > n`` silently. 
    makes every entry True, and ``n_a = 0`` gives all-False rows.
    B : int - number of rows.
    rng : numpy.random.Generator - source of randomness. Consumes ``B * n`` uniform draws.

    Returns:
    -------
    np.ndarray of bool, shape (B, n) - true marks side A.

    Notes:
    -----
    Peak memory is about ``16 * B * n`` bytes for the float draws plus the
    argsort result.
    """
    M = np.zeros((B, n), dtype=bool)
    np.put_along_axis(M, np.argsort(rng.random((B, n)), axis=1)[:, :n_a], True, axis=1)
    return M

def _distinct_masks(n: int, n_a: int, B: int, rng: np.random.Generator) -> np.ndarray:
    """
    Sample ``B`` distinct partitions uniformly, without replacement. 

    Sampling with replacement would put duplicate partitions in the null, the tail fraction depends on how many
    times one lucky split happened to be drawn. masks are drawn in batches with :func:`_draw_masks`, and rows
    already seen are rejected. every accepted row is uniform over the partitions not yet taken, so returned set is
    a uniform random ``B``-subset of partition space.

    Parameters:
    ----------
    n, n_a : int - number of units and size of side A.
    B : int - number of distinct partitions to return. Must not exceed
        ``n_partitions(n, n_a)``.
    rng : numpy.random.Generator - source of randomness. The same seed and arguments give the same rows
    in the same order.

    Returns:
    -------
    np.ndarray of bool, shape (B, n) - distinct masks, in order of acceptance. For ``B <= 0`` the result is
    an empty **float64** array of shape ``(0,)``, not ``(0, n)`` bool.
    """
    seen: set[bytes] = set()
    keep: list[np.ndarray] = []
    balanced = 2 * n_a == n
    while len(keep) < B:
        want = B - len(keep)
        M = _draw_masks(n, n_a, min(200_000, 4 * want + 1_000), rng)
        if balanced:
            M = np.where(M[:, [0]], M, ~M)
        for row, packed in zip(M, np.packbits(M, axis=1)):
            key = packed.tobytes()
            if key in seen:
                continue
            seen.add(key)
            keep.append(row.copy())
            if len(keep) == B:
                break
    return np.asarray(keep)

class _Accumulator:
    """
    Streaming summary of a null : histogram, exact tail counts, exact minimum.

    A histogram is kept instead of the raw values because an exhaustive null can be large. tail counts against
    pre-registered thresholds are exact up to the ``_REACH_RTOL`` slack. quantiles are resolved to the histogram bin width.


    Parameters:
    ----------
    thresholds : sequence of float - observed statistics to count exactly. A null value ``v`` counts toward
    threshold ``t`` when ``v >= _reach_floor(t)``. A ``nan`` threshold is not rejected and always counts 0.
    bins : int - number of equal-width histogram bins on ``[0, hi]``. Must be at least 1, but this is not checked.
    hi : float - upper edge of the histogram. Values outside ``[0, hi]`` are left out
    of the histogram and counted in ``out_of_range``. They still enter the tail counts and the maximum.
    keep : bool - if True, also keep a copy of every batch in ``kept``.

    Attributes:
    ----------
    thr, floor : ndarray - thresholds and their reach floors.
    cnt : ndarray of int64 - exceedance count per threshold.
    edges, hist : ndarray - bin edges, shape ``(bins + 1,)``, and counts, shape ``(bins,)``.
    maximum : float - largest value seen so far, or ``-inf`` if nothing has been added.
    n : int - number of values added.
    out_of_range : int - number of values that fell outside ``[0, hi]``.
    kept : list of ndarray or None - copies of the batches, or None when ``keep=False``.
    """
    __slots__ = (
        "thr",
        "floor",
        "cnt",
        "edges",
        "hist",
        "maximum",
        "n",
        "out_of_range",
        "kept",
    )

    def __init__(self, thresholds: Sequence[float], bins: int, hi: float, keep: bool):
        self.thr = np.asarray(thresholds, dtype=float)
        self.floor = np.asarray([_reach_floor(t) for t in self.thr], dtype=float)
        self.cnt = np.zeros(self.thr.size, dtype=np.int64)
        self.edges = np.linspace(0.0, float(hi), int(bins) + 1)
        self.hist = np.zeros(int(bins), dtype=np.int64)
        self.maximum = -np.inf
        self.n = 0
        self.out_of_range = 0
        self.kept: list[np.ndarray] | None = [] if keep else None

    def add(self, v: np.ndarray) -> None:
        """
        Fold one batch of null statistics into summary.

        Parameters:
        -----------
        v : np.ndarray, shape (k,) - statistics for one batch. empty batch is ignored.

        Raises:
        -------
        RuntimeError 
        - if any value is non-finite. batch statistic returns ``nan`` only when a side has 
        fewer than 2 members, which the caller should have already ruled out. 

        Notes:
        ------
        counting against thresholds creates a temporary ``(k, T)`` boolean array, where ``T`` is number 
        of thresholds.
        """
        if v.size == 0:
            return
        if not np.isfinite(v).all():
            raise RuntimeError(
                "non-finite statistic in the null; a side had fewer than two members, "
                "which arbitrary_split_null should have ruled out"
            )
        self.out_of_range += int(((v < self.edges[0]) | (v > self.edges[-1])).sum())
        self.hist += np.histogram(v, bins=self.edges)[0]
        if self.thr.size:
            self.cnt += (v[:, None] >= self.floor[None, :]).sum(axis=0)
        self.maximum = max(self.maximum, float(v.max()))
        self.n += int(v.size)
        if self.kept is not None:
            self.kept.append(v.copy())


def _hist_quantile(edges: np.ndarray, cum: np.ndarray, n: int, p: float) -> float:
    """
    Estimate a quantile from a histogram as the right edge of the first bin whose cumulative count reaches ``p * n``.

    The true inverted-CDF quantile (``np.quantile(..., method="inverted_cdf")``) lies in that bin. 
    The estimate is never below it and at most one bin width above it, provided every value fell inside the histogram
    range.

    Parameters:
    ----------
    edges : np.ndarray, shape (bins + 1,) - histogram bin edges.
    cum : np.ndarray, shape (bins,) - cumulative bin counts, ``np.cumsum(hist)``.
    n : int - total number of values, normally ``cum[-1]``.
    p : float - probability in ``[0, 1]``.

    Returns:
    -------
    float - the estimate. ``nan`` if ``n <= 0``. ``p = 0`` returns the right edge of the first bin even if that bin is empty
    """
    if n <= 0:
        return float("nan")
    i = int(np.searchsorted(cum, p * n))
    return float(edges[min(i + 1, edges.size - 1)])

@dataclass(frozen = True)
class NullResult:
    """
    The arbitrary-split null for one matrix, statistic and pair of split sizes.

    Returned by :func:`arbitrary_split_null`. the maximum and the registered tail counts are computed
    while streaming and are exact. for tail counts, "exact" is up to the ``_REACH_RTOL`` slack in what counts as reaching a
    threshold. the median, 95th and 99th percentiles are histogram estimates, resolved to one bin width and biased upward 
    by at most that width. if ``n_out_of_range`` is non-zero, the histogram range was too
    narrow. In that case every quantile is ``nan``, while the maximum and the tail counts are still valid

    Attributes:
    ----------
    median, pct95, pct99 : float - histogram quantile estimates, or ``nan`` if any value fell out of range.
    maximum : float - largest statistic among the evaluated partitions. For a sampled null,
    this is a lower bound on the maximum over the full partition space.
    n_partitions : int - size of the partition space, ``n_partitions(n, n_a)``.
    n_evaluated : int - number of partitions evaluated: equal to ``n_partitions`` when
    exhaustive, ``B`` when sampled.
    exhaustive : bool - whether every partition was evaluated.
    n, n_a : int - number of units and size of side A. Side B has ``n - n_a`` units.
    basis : str - free-text description of how the similarity matrix was built.
    statistic : str - description of the statistic.
    seed : int or None - seed for the sampled path, or None when exhaustive.
    thresholds, tail_counts : tuple - registered observed statistics and their exceedance counts among the
    evaluated partitions.
    n_out_of_range : int - number of statistics outside the histogram range.
    values : np.ndarray or None - every statistic, present only when ``keep_values=True``. The order is
    the enumeration order (exhaustive) or the acceptance order (sampled).
    hist_edges, hist_counts : np.ndarray - the histogram.
    note : str - caveats that the caller should report along with the result.
    """
    median: float
    pct95: float
    pct99: float
    maximum: float
    n_partitions: int
    n_evaluated: int
    exhaustive: bool
    n: int
    n_a: int
    basis: str = "unspecified"
    statistic: str = "|dispersion gap| over off-diagonal pairs"
    seed: int | None = None
    thresholds: tuple[float, ...] = ()
    tail_counts: tuple[int, ...] = ()
    n_out_of_range: int = 0
    values: np.ndarray | None = field(default=None, repr=False)
    hist_edges: np.ndarray = field(default_factory=lambda: np.empty(0), repr=False)
    hist_counts: np.ndarray = field(default_factory=lambda: np.empty(0), repr=False)
    note: str = ""
    def _registered(self, observed: float) -> int | None:
        """
        Return the tail count for ``observed`` if it was registered, else None.
        """
        for t, c in zip(self.thresholds, self.tail_counts):
            if math.isclose(float(observed), float(t), rel_tol=1e-12, abs_tol=0.0):
                return int(c)
        return None

    def tail_count_is_exact(self, observed : float) -> bool:
        """
        Whether the exceedance count among the *evaluated* partitions is known exactly for ``observed``.

        True when values were kept or when ``observed`` was registered. This
        says nothing about the full partition space. For that, see
        :meth:`population_p_is_exact`.
        """
        return (self.values is not None or self._registered(observed) is not None)

    def population_p_is_exact(self, observed : float) -> bool:
        """
        Whether :meth:`p_value` is exact over the complete partition space.

        This requires both an exhaustive null and an exact tail count. A
        sampled null gives only a Monte Carlo estimate, even for registered
        thresholds.
        """
        return (self.exhaustive and self.tail_count_is_exact(observed))
    
    def p_value(self, observed: float) -> float:
        """
        Fraction of evaluated partitions whose |gap| reaches ``observed``.

        A partition reaches ``observed`` when its statistic is
        ``>= _reach_floor(observed)``. The result is a plain proportion,
        ``count / n_evaluated``, with no add-one correction, so a sampled null
        can return exactly 0.

        The source is chosen in this order:

        1. Kept values, if any: exact.
        2. A registered threshold: exact. Returns ``nan`` if nothing was
           evaluated.
        3. Otherwise, the upper histogram bound from
           :meth:`p_value_bounds`. This is the conservative choice, since a
           larger p makes the published effect look less distinguishable
           from arbitrary splits.

        Raises
        ------
        RuntimeError
            In case 3, if any statistic fell outside the histogram range.
        """
        if self.values is not None:
            return float(np.mean(self.values >= _reach_floor(observed)))
        
        hit = self._registered(observed)

        if hit is not None:
            if not self.n_evaluated:
                return float("nan")

            return float(hit) / self.n_evaluated

        return self.p_value_bounds(observed)[1]

    def p_value_bounds(self, observed: float) -> tuple[float, float]:
        """
        Return ``(lower, upper)`` bounds on :meth:`p_value`.

        When the count is exact (values kept, or ``observed`` registered),
        both bounds are equal. Otherwise they come from the histogram:

        - ``lower`` counts only bins that lie entirely at or above
          ``_reach_floor(observed)``.
        - ``upper`` counts every bin that could contain such a value.

        The gap between the bounds is at most the mass of the one bin that
        contains the floor.

        Returns
        -------
        tuple of float
            ``(nan, nan)`` if nothing was evaluated or there is no histogram.

        Raises
        ------
        RuntimeError
            If the bounds must come from the histogram and any statistic fell
            outside its range. Such statistics are missing from the histogram,
            so the bounds would be wrong.
        """
        if self.values is not None:
            p = float(np.mean(self.values >= _reach_floor(observed)))
            return p, p
        hit = self._registered(observed)

        if hit is not None and self.n_evaluated:
            p = float(hit) / self.n_evaluated
            return p, p

        if self.n_out_of_range:
            raise RuntimeError(f'Cannot calculate histogram-based p-value bounds because {self.n_out_of_range} null statistics fell outside hist range.')
        
        if not self.n_evaluated or self.hist_counts.size == 0:
            return float("nan"), float("nan")
        
        floor = _reach_floor(observed)

        left, right = self.hist_edges[:-1], self.hist_edges[1:]

        lo = float(self.hist_counts[left >= floor].sum()) / self.n_evaluated
        hi = float(self.hist_counts[right >= floor].sum()) / self.n_evaluated
        return lo, hi

    def quantile(self, p: float) -> float:
        """
        Null quantile at probability ``p``, estimated from the histogram.
        """
        if self.hist_counts.size == 0 or self.n_out_of_range:
            return float("nan")
        return _hist_quantile(
            self.hist_edges, np.cumsum(self.hist_counts), self.n_evaluated, p
        )

    def histogram(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Return ``(edges, counts)`` for plotting.
        """
        return self.hist_edges, self.hist_counts

    def ratio_to(self, observed: float) -> float:
        """
        Return ``maximum / observed``: how far past the published gap arbitrary splits got.
        """
        if observed == 0:
            return float("nan")
        return float(self.maximum / observed)

    def summary(self) -> str:
        """
        One-line, human-readable report.

        It covers the statistic, the split sizes, the basis, exhaustive or
        sampled (with seed), the quantiles and maximum, and ``p =
        count / n_evaluated`` for each registered threshold. Any note is
        appended at the end.
        """
        how = (
            f"exhaustive over {self.n_partitions:,} partitions"
            if self.exhaustive
            else f"{self.n_evaluated:,} distinct partitions sampled from {self.n_partitions:,} "
            f"(seed {self.seed})"
        )
        s = (
            f"{self.statistic} under {self.n_a}/{self.n - self.n_a} splits of {self.n} units, "
            f"basis: {self.basis}; {how}; median {self.median:.6g}, 95th {self.pct95:.6g}, "
            f"99th {self.pct99:.6g}, max {self.maximum:.6g}"
        )
        for t, c in zip(self.thresholds, self.tail_counts):
            p = c / self.n_evaluated if self.n_evaluated else float("nan")
            s += f"; p({t:.6g}) = {p:.6g} ({c:,}/{self.n_evaluated:,})"
        if self.note:
            s += f"; NOTE: {self.note}"
        return s

    def as_dict(self) -> dict:
        """
        Scalar fields as a dict, for logging or JSON.
        """
        return {
            "statistic": self.statistic,
            "basis": self.basis,
            "n": self.n,
            "n_a": self.n_a,
            "n_partitions": self.n_partitions,
            "n_evaluated": self.n_evaluated,
            "population_p_exact": self.exhaustive,
            "seed": self.seed,
            "median": self.median,
            "pct95": self.pct95,
            "pct99": self.pct99,
            "maximum": self.maximum,
            "thresholds": list(self.thresholds),
            "tail_counts": list(self.tail_counts),
            "n_out_of_range": self.n_out_of_range,
            "note": self.note,
        }

def arbitrary_split_null(
    sim,
    n_a: int,
    *,
    exhaustive: bool | None = None,
    B: int = 200_000,
    seed: int = 0,
    observed: float | Sequence[float] | None = None,
    units: Sequence[int] | None = None,
    basis: str = "unspecified",
    max_exhaustive: int = 100_000_000,
    keep_values: bool = False,
    max_keep: int = 5_000_000,
    chunk: int = 200_000,
    hist_bins: int = 200_000,
    hist_hi: float | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> NullResult:
    """
    Distribution of |dispersion_gap| over arbitrary partitions of units that share a single source.

    Pass the similarity matrix restricted to units from one source. Those
    units cannot differ by provenance, so any gap between arbitrary
    partitions of them arises without a provenance difference. The null
    shows how large a gap partitioning alone produces at the published split
    sizes.

    Parameters:
    ----------
    sim : array_like, shape (N, N) - symmetric similarity or distance matrix. 
    n_a : int - size of side A. it is cast with int() so floats are truncated.
    exhaustive : bool or None, default None
        - None: enumerate if the space has at most ``max_exhaustive``
          partitions, otherwise sample.
        - True: enumerate, and raise if the space is larger than
          ``max_exhaustive``.
        - False: sample, unless enumeration is no more expensive. Sampling is
          overridden when ``B >= space``, even if ``space > max_exhaustive``,
          or when ``2 * B >= space <= max_exhaustive``.
    B : int, default 200_000 - number of distinct partitions to sample. unused when enumerating.
    seed : int, default 0 - seed for sampled path. must be an int
    observed : float or sequence of float, optional - observed statistics to register so tail counts exact.
    units : sequence of int or bool mask, optional - units of ``sim`` to keep, normalized by ``_as_index``.
    basis : str, default "unspecified" - free-text description of how ``sim`` was built.
    max_exhaustive : int, default 100_000_000 - largest partition space that will be enumerated
    keep_values : bool, default False - keep every statistic in ``NullResult.values``, at 8 bytes each
    max_keep : int, default 5_000_000 - refuse ``keep_values = True`` when more statistics than this would be kept.
    chunk : int, default 200_000 - partitions scored per batch when enumerating. 
    hist_bins : int, default 200_000 - number of histogram bins.
    hist_hi : float, optional - upper edge of histogram.
    progress : callable, optional - called as ``progress(n_evaluated, n_target)`` after every batch.

    Returns:
    -------
    NullResult

    Raises:
    ------
    ValueError
    - if invalid ``sim`` or ``units``
    - if ``n_a`` out of range
    - if forced enumeration of a space above ``max_exhaustive``
    - if ``keep_values`` over ``max_kept``
    - if invalid ``hist_hi``
    RuntimeError
    - if a non-finite statistic
    - if enumeration count that does not match ``n_partitions``
    """
    S = _check_sim(sim)
    if units is not None:
        sel = _as_index(units, S.shape[0], "units")
        S = S[np.ix_(sel, sel)]
        if S.shape[0] < 4:
            raise ValueError("units selects fewer than 4 units")
    n = S.shape[0]

    n_a = int(n_a)
    if not 2 <= n_a <= n - 2:
        raise ValueError(
            f"n_a must leave at least two units on each side: got n_a={n_a} with n={n}"
        )

    space = n_partitions(n, n_a)

    if exhaustive is None:
        use_exhaustive = space <= max_exhaustive
    elif exhaustive:
        if space > max_exhaustive:
            raise ValueError(f'exhaustive = True but the space holds {space:,} partitions, above max_exhaustive={max_exhaustive:,}.')

        use_exhaustive = True
    else:
        use_exhaustive = False
    if not use_exhaustive and (B >= space or (2 * B >= space and space <= max_exhaustive)):
        use_exhaustive = True

    if observed is None:
        thresholds: tuple[float, ...] = ()
    elif np.ndim(observed) == 0:
        thresholds = (float(observed),)  #
    else:
        thresholds = tuple(float(x) for x in np.asarray(observed, dtype=float).ravel())

    n_target = space if use_exhaustive else int(B)
    if keep_values and n_target > max_keep:
        raise ValueError(f'keep_values = True would hold {n_target:,} float64 values. Raise max_keep above {max_keep:,} or set keep_values = False.')

    S0 = _zero_diagonal(S)

    #every within-group mean must lie between the minimum and the maximum off-diagonal values of S.
    #largest possible difference two within-group means is at most max(S) - min(S)

    offdiag_mask = ~np.eye(n, dtype = bool)
    offdiag = S[offdiag_mask]

    max_possible_gap = float(np.ptp(offdiag))

    if hist_hi is None:
        hist_hi = max(max_possible_gap, np.finfo(np.float64).eps, )
    else:
        hist_hi = float(hist_hi)

        if hist_hi <= 0:
            raise ValueError(f'hist_hi must be positive, received {hist_hi}')

        if hist_hi < max_possible_gap:
            raise ValueError(f'hist_hi = {hist_hi:.6g} is too small. Use hist_hi = None for auto sizing.')

    acc = _Accumulator(thresholds, hist_bins, hist_hi, keep_values, )

    if use_exhaustive:
        source: Iterator[np.ndarray] = _enumerate_masks(n, n_a, chunk)
    else:
        source = iter([_distinct_masks(n, n_a, int(B), np.random.default_rng(seed))])

    for M in source:
        acc.add(_gap_batch(S0, M))
        if progress is not None:
            progress(acc.n, n_target)

    if use_exhaustive and acc.n != space:
        raise RuntimeError(f'enumerated {acc.n:,} partitions but the space holds {space:,}. Canoncalization is incorrect.')

    notes: list[str] = []
    if acc.out_of_range:
        notes.append(f'{acc.out_of_range:,} statistics fell outside [0, {hist_hi}]. Quantiles are NaN. The max and the registered tail counts are unaffected.')

    if not use_exhaustive:
        notes.append(f'sampled {acc.n:,} distinct partitions of {space:,}. The reported tail fraction is a Monte Carlo estimate.'
                     f'If the observed exceedance count is zero, the approximate 95% rule-of-three upper bound is {3 / acc.n:.2e}')

    if basis == "unspecified":
        notes.append('basis unspecified. This p cannot be reported and cannot be paired with a k_diagnostic until how sim was built.')

    cum = np.cumsum(acc.hist)
    ok = acc.out_of_range == 0
    quant = (
        (lambda p: _hist_quantile(acc.edges, cum, acc.n, p))
        if ok
        else (lambda p: float("nan"))
    )

    return NullResult(
        median=quant(0.50),
        pct95=quant(0.95),
        pct99=quant(0.99),
        maximum=float(acc.maximum),
        n_partitions=int(space),
        n_evaluated=int(acc.n),
        exhaustive=bool(use_exhaustive),
        n=int(n),
        n_a=int(n_a),
        basis=basis,
        seed=None if use_exhaustive else int(seed),
        thresholds=thresholds,
        tail_counts=tuple(int(c) for c in acc.cnt),
        n_out_of_range=int(acc.out_of_range),
        values=(np.concatenate(acc.kept) if acc.kept else None),
        hist_edges=acc.edges,
        hist_counts=acc.hist,
        note="; ".join(notes),
    )

@dataclass(frozen = True)
class KBreakEven:
    """
    Break-even values of the k diagnostic, and the split sizes they were derived for.

    Two thresholds bound the region where the direction of calibration error
    cannot be determined: ``population <= conditional``. Which side of the
    region counts as conservative is decided by the code that builds
    :class:`KResult`, not here.

    Attributes:
    ----------
    population : float - break-even k under the population calibration. Positive and finite.
    conditional : float - break-even k under the conditional calibration. Positive and finite,
    and ``>= population``.
    null_n_a, null_n_b : int - side sizes of the arbitrary-split null. Each at least 2.
    observed_n_a, observed_n_b : int - sizes of the published groups. Each at least 2.
    derivation : str - free-text record of how the break-evens were obtained.

    Raises:
    ------
    ValueError
    - if any constraint above fails
    """
    population : float
    conditional : float
    null_n_a : int
    null_n_b : int

    observed_n_a : int
    observed_n_b : int

    derivation : str = ""

    def __post_init__(self) -> int:
        if not np.isfinite(self.population) or self.population <= 0:
            raise ValueError(
                "population k* must be positive and finite"
            )

        if not np.isfinite(self.conditional) or self.conditional <= 0:
            raise ValueError(
                "conditional k* must be positive and finite"
            )

        if self.population > self.conditional:
            raise ValueError(
                "population k* must be <= conditional k* so the "
                "undetermined region is well ordered"
            )

        for name in ("null_n_a", "null_n_b", "observed_n_a", "observed_n_b"):
            value = getattr(self, name)

            if value < 2:
                raise ValueError(f"{name} must be at least 2, got {value}")

@dataclass(frozen=True)
class KResult:
    """
    Direction-of-error diagnostic for a calibration p-value.

    It says whether the p from :func:`arbitrary_split_null` is an upper
    bound, a lower bound, or neither, compared with a calibration matched
    for size and dispersion.

    Attributes:
    ----------
    k : float - the diagnostic value.
    sd_a, sd_b : float - dispersions of groups A and B used to compute ``k``.
    verdict : str -one of ``_VERDICTS``: ``"conservative"``, ``"anti-conservative"`` or
    ``"undetermined"``.
    p_bound : str - ``"upper"``, ``"lower"`` or ``"undetermined"``. 
    n_a, n_b : int - group sizes.
    k_star_population, k_star_conditional : float -the break-even values that ``k`` is compared against (see
    :class:`KBreakEven`).
    basis : str - how the similarity matrix was built.
    note : str - free-text caveats.

    Raises:
    ------
    ValueError
    - if ``verdict`` or ``p_bound`` is not an allowed value, or if the two do not match.
    """

    k: float
    sd_a: float
    sd_b: float
    verdict: str
    p_bound : str

    n_a: int
    n_b: int

    k_star_population: float
    k_star_conditional: float

    basis: str = "unspecified"
    note: str = ""

    def __post_init__(self) -> None:
        if self.verdict not in _VERDICTS:
            raise ValueError(
                f"verdict must be one of {_VERDICTS}, "
                f"got {self.verdict!r}"
            )

        valid_bounds = (
            "upper",
            "lower",
            "undetermined",
        )

        if self.p_bound not in valid_bounds:
            raise ValueError(
                f"p_bound must be one of {valid_bounds}, "
                f"got {self.p_bound!r}"
            )

        expected_bound = {
            "conservative" : "upper",
            "anti-conservative" : "lower",
            "undetermined" : "undetermined",
        }[self.verdict]

        if self.p_bound != expected_bound:
            raise ValueError(f'verdict = {self.verdict!r} requires p_bound = {expected_bound!r}, got {self.p_bound!r} ')

    @property
    def p_relation(self) -> str:
        """
        One sentence stating how the reported p relates to a matched calibration.
        """
        if self.p_bound == "upper":
            return (
                "The reported calibration p is an upper bound : a size/dispersion-matched calibration would produce an equal or smaller tail probability."
            )
        if self.p_bound == "lower":
            return (
                "The reported calibration p is a lower bound : a size/dispersion-matched calibration would produce an equal or larger tail probability."
            )

        return (
            "The direction of calibration error cannot be determined from this diagnostic."
        )

    def summary(self) -> str:
        """
        One-line report: k, the SDs, the break-evens, the verdict, and how to read the p.
        """
        s = (
            f"k = {self.k:.4f}; "
            f"SD_A = {self.sd_a:.4f}, "
            f"SD_B = {self.sd_b:.4f}; "
            f"k*_population = {self.k_star_population:.4f}, "
            f"k*_conditional = {self.k_star_conditional:.4f}; "
            f"verdict = {self.verdict}; "
            f"p-bound = {self.p_bound}"
        )

        if self.p_bound == "upper":
            s += (
                "; the arbitrary-split null is wider than the matched "
                "reference. The reported tail probability is an UPPER "
                "BOUND and must not be interpreted as evidence that the "
                "published effect is readily attainable."
            )

        elif self.p_bound == "lower":
            s += (
                "; the arbitrary-split null is narrower than the matched "
                "reference. The reported tail probability is a LOWER "
                "BOUND and may exaggerate how exceptional the published "
                "effect appears."
            )

        else:
            s += (
                "; the diagnostic does not establish the direction of "
                "the calibration error."
            )

        return s

    def as_dict(self) -> dict:
        """
        All fields plus ``p_relation`` as a dict, for logging or JSON.
        """
        return {
            "k": self.k,
            "sd_a": self.sd_a,
            "sd_b": self.sd_b,
            "n_a": self.n_a,
            "n_b": self.n_b,
            "verdict": self.verdict,
            "p_bound": self.p_bound,
            "p_relation": self.p_relation,
            "k_star_population": self.k_star_population,
            "k_star_conditional": self.k_star_conditional,
            "basis": self.basis,
            "note": self.note,
        }

def _centrality(S0: np.ndarray, idx: np.ndarray, ) -> np.ndarray:
    """
    Each unit's mean similarity to the other members of its own group.

    For unit ``i`` in a group ``I`` with ``m`` members,

        c_i = sum over j in I with j != i of S0[i, j]  /  (m - 1)

    The mean of the ``c_i`` equals the group's within-mean from
    ``_mean_within``. Their spread shows how unevenly cohesion is distributed
    across the group's members.

    Parameters:
    ----------
    S0 : np.ndarray, shape (n, n) - similarity matrix with a zero diagonal (``_zero_diagonal``).
    idx : np.ndarray of int, shape (m,) - distinct member indices (``_as_index``)

    Returns:
    -------
    np.ndarray of float64, shape (m,) - centralities in the order of ``idx``. All ``nan`` if ``m < 2``, which
    gives an empty array when ``m = 0``.
    """
    if idx.size < 2:
        return np.full(idx.size, np.nan)

    block = S0[np.ix_(idx, idx)]

    return block.sum(axis=1) / (idx.size - 1)

def k_diagnostic(sim, 
                 idx_a, 
                 idx_b, 
                 *, 
                 break_even : KBreakEven,
                 basis : str = 'unspecified') -> KResult:
    """
    Determine whether the arbitrary-split p errs on the conservative or the anti-conservative side.

    The diagnostic is

        k = SD(centrality in B) / SD(centrality in A)

    using sample SDs (``ddof=1``), where centrality is defined as in
    :func:`_centrality`. Group A is the group the null was built around, and
    group B is the other published group. The verdict comes from comparing
    ``k`` with the break-evens in ``break_even``:

    - ``k < k*_population``: the arbitrary-split null is too wide compared
      with a matched comparison, so the reported p is an **upper bound**
      (conservative).
    - ``k > k*_conditional``: the null is too narrow, so the reported p is a
      **lower bound** (anti-conservative).
    - ``k*_population <= k <= k*_conditional``: undetermined. Both boundaries
      count as undetermined.

    Parameters:
    ----------
    sim : array_like, shape (n, n) - similarity matrix covering both groups
    idx_a, idx_b : array_like of int or bool - Members of groups A and B
    break_even : KBreakEven, keyword-only - break-even values derived for these observed group sizes
    basis : str, default "unspecified", keyword-only - how ``sim`` was built

    Returns:
    -------
    KResult - if ``sd_a`` is ``<= 0`` or either SD is non-finite, the result has
    ``k = nan``, verdict ``"undetermined"``, and a note explaining why.

    Raises:
    ------
    ValueError
    - if the groups overlap
    - if a group size does not match ``break_even``

    Notes:
    -----
    ``k`` does not change under any affine rescaling ``a * S + b`` of the
    similarities with ``a != 0``, because both SDs scale by ``|a|``.
    """
    S = _check_sim(sim, min_n=2, )

    n = S.shape[0]

    a = _as_index(idx_a, n, "idx_a", )

    b = _as_index(idx_b, n, "idx_b", )

    if np.intersect1d(a, b).size:
        raise ValueError(
            "idx_a and idx_b overlap; "
            "the two groups must be disjoint"
        )

    if a.size != break_even.observed_n_a:
        raise ValueError(
            f"k* was derived for observed group A size "
            f"{break_even.observed_n_a}, "
            f"but received {a.size}"
        )

    if b.size != break_even.observed_n_b:
        raise ValueError(
            f"k* was derived for observed group B size "
            f"{break_even.observed_n_b}, "
            f"but received {b.size}"
        )

    S0 = _zero_diagonal(S)

    notes: list[str] = []

    ca = _centrality(S0, a, )

    cb = _centrality(S0, b, )

    sd_a = float(np.std(ca, ddof = 1, ))

    sd_b = float(np.std(cb, ddof = 1))
    if (
        sd_a <= 0.0
        or not np.isfinite(sd_a)
        or not np.isfinite(sd_b)
    ):
        return KResult(
            k=float("nan"),
            sd_a=sd_a,
            sd_b=sd_b,
            verdict="undetermined",
            p_bound="undetermined",
            n_a=int(a.size),
            n_b=int(b.size),
            k_star_population=float(
                break_even.population
            ),
            k_star_conditional=float(
                break_even.conditional
            ),
            basis=basis,
            note=(
                "centrality dispersion is zero or non-finite, "
                "so k and the direction of calibration error "
                "are undefined"
            ),
        )

    k = sd_b / sd_a

    #small k -> null too wide -> p too large -> upper bound
    #large k -> null too narrow -> p too small -> lower bound

    if k < break_even.population:
        verdict = "conservative"
        p_bound = "upper"

    elif k > break_even.conditional:
        verdict = "anti-conservative"
        p_bound = "lower"

    else:
        verdict = "undetermined"
        p_bound = "undetermined"

    if basis == "unspecified":
        notes.append(
            "basis unspecified"
        )

    elif (
        "pearson" not in basis.lower()
        and "correlation" not in basis.lower()
    ):
        notes.append(
            f"the supplied break-even values were derived for a correlation statistic, under basis {basis!r}. Interpretation has not been established"
        )

    return KResult(
        k=float(k),
        sd_a=sd_a,
        sd_b=sd_b,
        verdict=verdict,
        p_bound=p_bound,
        n_a=int(a.size),
        n_b=int(b.size),
        k_star_population=float(
            break_even.population
        ),
        k_star_conditional=float(
            break_even.conditional
        ),
        basis=basis,
        note="; ".join(notes),
    )

def named_split(
    sim,
    idx_by_group: Mapping[Hashable, Sequence[int]],
    ordering: Sequence[Hashable],
) -> float:
    """
    Signed dispersion gap of one specific, named split.

    Named splits answer objections like "your null is just unequal depth" or
    "you would get this by splitting on anything." Scoring the splits a
    skeptic would name (alphabetical by unit name, by pool size, by sample
    count, by acquisition date) puts concrete, plausible alternatives next to
    the published gap and the arbitrary-split null.

    Parameters:
    ----------
    sim : array_like, shape (n, n) - similarity matrix over the units.
    idx_by_group : mapping of label -> indices - exactly two disjoint sides, keyed by the names you will print
    ordering : sequence of two labels - the two keys of ``idx_by_group``, side A first. Required, because it
    fixes the sign, and the sign is part of the result.

    Returns:
    -------
    float - ``mean within-A - mean within-B`` over off-diagonal pairs, or ``nan``
    if either side has fewer than 2 members.

    Raises:
    ------
    ValueError
    - if ``ordering`` does not have exactly two entries
    - if does not name exactly the keys of ``idx_by_group``
    - if ``idx_by_group`` does not have exactly two groups, which can only happen with a repeated label
    """
    keys = list(ordering)
    if len(keys) != 2:
        raise ValueError(f"ordering must name exactly two sides, got {len(keys)}")
    if set(keys) != set(idx_by_group):
        raise ValueError(
            f"ordering {keys!r} must name exactly the sides in idx_by_group "
            f"{sorted(map(str, idx_by_group))!r}"
        )
    if len(idx_by_group) != 2:
        raise ValueError("idx_by_group must contain exactly two groups")
    
    return dispersion_gap(
        sim, idx_by_group[keys[0]], idx_by_group[keys[1]], signed=True
    )


def median_split_indices(
    values: Sequence[float],
    n_a: int | None = None,
    *,
    max_variants: int = 2_000,
    seed: int = 0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Every median split consistent with ``values``, covering all tie-breaks at the cut.

    Units are ranked by ``values``, and side A gets the ``n_a`` smallest.
    Every unit strictly below the cut value goes to side A. When several
    units share the cut value, each way of choosing the remaining members
    from that tie block is a separate, equally valid split, and all of them
    are returned (or a sample, see ``max_variants``). This keeps a favorable
    tie-break from being picked without anyone noticing.

    Parameters
    ----------
    values : sequence of float - one ordering key per unit, in the unit order of the similarity
        matrix. Must be 1-D and finite.
    n_a : int, optional - size of side A. Defaults to ``n // 2``, which puts the extra unit on
        side B when ``n`` is odd. Must lie in ``[1, n - 1]``. 
    max_variants : int, default 2_000, keyword-only -  if the tie block admits more tie-breaks than this, exactly
    ``max_variants`` distinct ones are sampled uniformly instead of listing all of them. 
    ``max_variants <= 0`` returns an empty list.
    seed : int, default 0, keyword-only - seed for that sampling. Unused when every variant is listed.

    Returns:
    -------
    list of (np.ndarray, np.ndarray) - one ``(idx_a, idx_b)`` pair per variant. Both arrays are ascending
    integer indices and together form a complete partition. Variants are in lexicographic order 
    of the tied units chosen for side A. When the cut does not fall inside a tie block, there is a single variant.

    Raises:
    ------
    ValueError
    -  if ``values`` is not 1D or note finite
    - if ``n_a`` is out of range
        If ``values`` is not 1-D or not finite, or ``n_a`` is out of range
        (including ``n < 2``).
    """
    v = np.asarray(values, dtype=float)
    if v.ndim != 1:
        raise ValueError("values must be one-dimensional, one entry per unit")
    if not np.isfinite(v).all():
        raise ValueError("values contains non-finite entries; units cannot be ranked")
    n = v.size
    n_a = n // 2 if n_a is None else int(n_a)
    if not 1 <= n_a <= n - 1:
        raise ValueError(f"n_a must lie in [1, {n - 1}], got {n_a}")

    thr = np.sort(v)[n_a - 1]
    below = np.flatnonzero(v < thr)
    tied = np.flatnonzero(v == thr)
    need = n_a - below.size

    n_variants = math.comb(tied.size, need)
    if n_variants <= max_variants:
        picks: list[tuple[int, ...]] = list(itertools.combinations(tied.tolist(), need))
    else:
        rng = np.random.default_rng(seed)
        seen: set[tuple[int, ...]] = set()
        while len(seen) < max_variants:
            seen.add(tuple(sorted(rng.choice(tied, need, replace=False).tolist())))
        picks = sorted(seen)

    out: list[tuple[np.ndarray, np.ndarray]] = []
    for pick in picks:
        mask = np.zeros(n, dtype=bool)
        mask[below] = True
        mask[list(pick)] = True
        out.append((np.flatnonzero(mask), np.flatnonzero(~mask)))
    return out