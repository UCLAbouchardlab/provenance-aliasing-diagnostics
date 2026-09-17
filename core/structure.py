"""
The provenance structure connecting analysis units.

Which units share a source, and which would lose all support if a source were
removed? These questions describe how a resource was assembled before any
molecular feature is measured or any similarity geometry is chosen.

Computation:
-----------
Let M be the binary source x unit incidence matrix. ``M[d, u] = 1`` means at
least one retained record connects source d to unit u. Then

    C = M.T @ M

counts shared sources for every pair of units. The diagonal counts each unit's
sources. Off-diagonal entries greater than zero identify pairs sharing at least
one source; a pair is counted once even when several sources connect it.

Weighting:
---------
Connectivity, distinct-unit counts, and pair-sharing rates ignore row weights
and repeated membership. Zero-weight observations still create structural edges.
Record-count fields in the spanning-source table do retain row multiplicity,
and the design summary also reports weight in spanning sources explicitly.

Interpretation:
---------------
Shared provenance identifies common support and possible dependence. It does
not establish a measurement-level batch effect or make shared units independent.
Group-specific summaries require one group per analysis unit. All source-removal
comparisons retain the original unit universe so loss of support stays visible.

Contents:
---------
shared_source_matrix - shared-source counts over unit pairs.
pair_sharing_by_group - within-group pair-sharing counts and rates.
structural_leave_one_source_out - units left unsupported by each source removal.
spanning_source_table - sources contributing to more than one group.
concentration - dependence of pair sharing on the largest source.
design_summary - compact counts, crossing, concentration, and removal summaries.
"""
from __future__ import annotations

from typing import TypeVar

import numpy as np
import pandas as pd

from .incidence import Corpus

_F = TypeVar("_F", pd.DataFrame, pd.Series)

def _stamp(obj: _F, corpus: Corpus, note: str = "") -> _F:
    """
    Attach corpus, weighting, grain, and structural interpretation to pandas output.

    Updates ``attrs`` in place and returns the same object. ``weight_sensitive=False``
    describes the structural calculation; explicitly reported mass fields in a mixed
    design summary still depend on row weights.
    """
    obj.attrs.update(
        {
            "corpus": corpus.name,
            "weighting": corpus.weighting,
            "grain": "the corpus's `source` column",
            "weight_sensitive": False,
            "note": note,
        }
    )
    return obj


def _n_pairs(n: int) -> int:
    """
    Number of unordered pairs among n units: ``n * (n - 1) // 2``, or 0 below two.
    """
    return (n * (n - 1)) // 2 if n > 1 else 0


def _rate(numerator: float, denominator: float) -> float:
    """
    Return a count divided by its denominator, or ``nan`` for a non-positive denominator.
    """
    return float(numerator) / float(denominator) if denominator > 0 else float("nan")


def _unit_group(corpus: Corpus) -> pd.Series:
    """
    Return group labels in corpus unit order, requiring one group per unit.

    Re-raises an inconsistent unit-group mapping as a contextual ValueError for
    the group-specific structural calculations.
    """
    try:
        ug = corpus.unit_group
    except ValueError as exc:
        raise ValueError(
            f"{corpus.name!r}: the per-group structure diagnostics need exactly one group "
            "per unit, and at least one unit spans groups. When a unit spans groups the "
            "group is a property of the row, not of the unit, and 'within-group unit "
            "pairs' is not defined. Either coarsen to a grain at which units are "
            "group-pure, or restrict with Corpus.subset(groups=...)."
        ) from exc
    return ug.reindex(corpus.units)


def _group_dummies(corpus: Corpus, unit_group: pd.Series) -> pd.DataFrame:
    """
    Build integer unit x group indicators in the corpus's unit and group order.
    """
    d = pd.get_dummies(unit_group)
    return d.reindex(index=corpus.units, columns=corpus.groups, fill_value=False).astype(np.int64)

def shared_source_matrix(corpus: Corpus) -> pd.DataFrame:
    """
    Count the sources shared by every pair of analysis units.

    Parameters:
    ----------
    corpus : Corpus - binary source x unit membership supplies the calculation.

    Returns:
    -------
    pd.DataFrame, shape (n_units, n_units) - symmetric integer matrix ``M.T @ M``,
    with canonical unit keys on both axes. Off-diagonal entries count shared
    sources; diagonal entries count the sources supporting each unit.

    Notes:
    ------
    Weights and repeated rows do not alter the binary membership matrix. A count
    greater than zero records shared provenance, without measuring molecular
    similarity. The returned matrix is dense and requires quadratic storage in
    the number of units.
    """
    M = corpus.incidence
    A = M.to_numpy(dtype=np.int64)
    co = pd.DataFrame(A.T @ A, index=M.columns, columns=M.columns)
    co.index.name = "unit"
    co.columns.name = "unit"
    return _stamp(
        co,
        corpus,
        note="entry (u,v) = number of sources feeding both units; diagonal = sources per unit",
    )


def pair_sharing_by_group(corpus: Corpus) -> pd.DataFrame:
    """
    Measure the fraction of within-group unit pairs sharing at least one source.

    For a group with n units, the denominator is ``n * (n - 1) / 2``. Each
    unordered pair contributes at most once, irrespective of how many sources it shares.

    Parameters:
    ----------
    corpus : Corpus - the incidence design, with one group label per unit.

    Returns:
    -------
    pd.DataFrame - indexed by group, with ``n_units``, ``n_pairs``,
    ``n_pairs_sharing``, ``pair_sharing_rate``, ``n_sources``,
    ``sources_per_unit_mean``, ``sources_per_unit_max``, and
    ``units_with_one_source``. A group with fewer than two units has zero pairs
    and an undefined pair-sharing rate.

    Raises:
    ------
    ValueError
    - if a unit carries more than one group label.

    Notes:
    ------
    All fields use binary membership and distinct units. A high sharing rate
    describes common provenance support; it is not itself a similarity statistic
    or an estimate of the strength of a batch effect.
    """
    ug = _unit_group(corpus)
    co = shared_source_matrix(corpus).to_numpy()
    M = corpus.incidence
    units = list(M.columns)
    pos = {u: i for i, u in enumerate(units)}

    rows = []
    for g in corpus.groups:
        us = [u for u in units if ug[u] == g]
        n = len(us)
        idx = np.array([pos[u] for u in us], dtype=int)
        if n >= 2:
            block = co[np.ix_(idx, idx)]
            iu = np.triu_indices(n, 1)
            sharing = int((block[iu] > 0).sum())
        else:
            sharing = 0
        n_pairs = _n_pairs(n)

        sub = M.iloc[:, idx] if n else M.iloc[:, :0]
        spu = sub.sum(axis=0)
        rows.append(
            {
                "group": g,
                "n_units": n,
                "n_pairs": n_pairs,
                "n_pairs_sharing": sharing,
                "pair_sharing_rate": _rate(sharing, n_pairs),
                "n_sources": int((sub.sum(axis=1) > 0).sum()),
                "sources_per_unit_mean": float(spu.mean()) if n else float("nan"),
                "sources_per_unit_max": int(spu.max()) if n else 0,
                "units_with_one_source": int((spu == 1).sum()),
            }
        )

    out = pd.DataFrame(rows).set_index("group")
    return _stamp(
        out,
        corpus,
        note="within-group unit pairs sharing >=1 source; incidence arithmetic, no measurement",
    )


def structural_leave_one_source_out(corpus: Corpus) -> pd.DataFrame:
    """
    Count units that would lose every source if one source were removed.

    An emptied unit has exactly one supporting source in the original incidence
    matrix and belongs to the source being removed. The original group sizes remain
    the denominators, so a vanished unit cannot disappear from the calculation.

    Parameters:
    ----------
    corpus : Corpus - source x unit membership and one group label per unit.

    Returns:
    -------
    pd.DataFrame - source-indexed table sorted by decreasing ``emptied_total``
    and then ``units_covered``. For each group g, ``covers_g``, ``emptied_g``,
    and ``frac_emptied_g`` give coverage, lost units, and the fraction of the
    original group's units lost. ``emptied_units`` is a display string;
    ``emptied_unit_keys`` retains the canonical keys as a tuple.

    Raises:
    ------
    ValueError
    - if units do not have unique group labels.

    Notes:
    ------
    This is a structural deletion calculation, with no model refitting and no use
    of row weights. It differs from entropy leave-one-source-out, which recomputes
    R after removal and can change the marginal group distribution.
    """
    ug = _unit_group(corpus)
    M = corpus.incidence
    units = list(M.columns)
    A = M.to_numpy(dtype=np.int64)  # sources x units

    sole = (A.sum(axis=0) == 1).astype(np.int64)  # units whose only contributor is one source
    E = A * sole  # 1 where this source is the unit's sole contributor

    G = _group_dummies(corpus, ug)
    Gv = G.to_numpy()
    sizes = Gv.sum(axis=0)
    covers_g = A @ Gv
    emptied_g = E @ Gv

    data: dict[str, object] = {
        "units_covered": A.sum(axis=1),
        "emptied_total": E.sum(axis=1),
    }
    for j, g in enumerate(corpus.groups):
        size = int(sizes[j])
        data[f"covers_{g}"] = covers_g[:, j]
        data[f"emptied_{g}"] = emptied_g[:, j]
        data[f"frac_emptied_{g}"] = (
            emptied_g[:, j] / size if size > 0 else np.full(A.shape[0], np.nan)
        )
    emptied_keys = [tuple(u for u, flag in zip(units, row) if flag) for row in (E > 0)]
    data["emptied_units"] = [", ".join(map(str, keys)) for keys in emptied_keys]
    data["emptied_unit_keys"] = emptied_keys

    out = pd.DataFrame(data, index=M.index)
    out.index.name = "source"
    out = out.sort_values(
        ["emptied_total", "units_covered"], ascending=False, kind="mergesort"
    )
    return _stamp(
        out,
        corpus,
        note="a unit is 'emptied' when the removed source is its only contributor",
    )


def spanning_source_table(corpus: Corpus) -> pd.DataFrame:
    """
    List sources whose retained records include more than one group.

    Parameters:
    ----------
    corpus : Corpus - source, unit, and group membership in the analytical projection.

    Returns:
    -------
    pd.DataFrame - indexed by source and sorted by decreasing ``n_rows``. Columns
    are ``n_rows``, ``n_groups``, ``groups``, ``n_units``, and ``units``.
    The group and unit lists are comma-separated display strings. Returns an empty
    table with the same columns when no sources span groups.

    Notes:
    ------
    Spanning is structural: zero-weight observations count as group membership.
    Entropy uses positive mass and can therefore report fewer spanning sources.
    ``n_rows`` includes retained repeated observations; ``n_units`` counts distinct
    units. Use the corpus's canonical keys for computation rather than parsing the
    display lists, especially when keys are composite.
    """
    cols = [
        "n_rows",
        "n_groups",
        "groups",
        "n_units",
        "units",
    ]

    f = corpus.frame
    rows = []

    for s in corpus.spanning_sources():
        sub = f[f["source"] == s]

        rows.append(
            {
                "source": s,
                "n_rows": int(len(sub)),
                "n_groups": int(sub["group"].nunique()),
                "groups": ", ".join(
                    sorted(map(str, sub["group"].unique()))
                ),
                "n_units": int(sub["unit"].nunique()),
                "units": ", ".join(
                    sorted(map(str, sub["unit"].unique()))
                ),})

    if rows:
        out = (pd.DataFrame(rows).set_index("source")[cols].sort_values("n_rows",ascending=False,kind="mergesort",))
    else:
        out = pd.DataFrame({c: pd.Series(dtype=object) for c in cols})
        out.index.name = "source"

    return _stamp(out,corpus, note=(
            "sources contributing to >1 biological group; "
            "incidence arithmetic only"
        ),)


def concentration(corpus: Corpus) -> pd.Series:
    """
    Measure how much pair sharing is supported by the source covering the most units.

    Two quantities distinguish coverage from exclusive dependence. The largest
    source covers ``choose(k, 2)`` pairs when it contains k distinct units. Some
    of those pairs may remain connected through other sources after it is removed.

    Parameters:
    ----------
    corpus : Corpus - binary source x unit incidence; group labels do not enter
    the pair counts.

    Returns:
    -------
    pd.Series - overall ``n_units``, ``n_pairs``, ``n_pairs_sharing``, and
    ``pair_sharing_rate``; the largest source's display label, unit count, pair
    count, and ``top_source_share_of_sharing``; ``n_pairs_sharing_without_top``
    and ``share_of_sharing_lost_without_top`` after its removal; and the number
    of sources feeding multiple units. Corpus and weighting labels are included.

    Computation:
    -----------
    ``top_source_share_of_sharing`` divides the top source's pair count by all
    pairs sharing any source. ``share_of_sharing_lost_without_top`` divides the
    number of connections actually lost by that same original count. Rates with
    zero denominators are ``nan``.

    Notes:
    ------
    All unordered pairs are considered, including pairs from different groups.
    The unit universe is retained after removal. Ties for largest source select
    the first source in corpus order. The output's ``top_source`` is a display
    string, while matrix selection uses the canonical source key.
    """
    M = corpus.incidence
    A = M.to_numpy(dtype=np.int64)
    n_units = A.shape[1]
    iu = np.triu_indices(n_units, 1)

    co = A.T @ A
    n_pairs = _n_pairs(n_units)
    n_sharing = int((co[iu] > 0).sum())

    ups = M.sum(axis=1)
    # Retain the object-valued key for indexing; stringify only the reported field.
    # Composite source keys are tuples and ``Series.loc[tuple]`` is not tuple-safe.
    top_key = ups.idxmax()
    k = int(ups.at[top_key])
    top_pairs = _n_pairs(k)

    keep = [i for i, source in enumerate(M.index) if source != top_key]
    A2 = A[keep, :] if keep else A[:0, :]
    co2 = A2.T @ A2
    n_sharing_without = int((co2[iu] > 0).sum())

    if not top_pairs <= n_sharing:
        raise AssertionError(
            f"internal inconsistency: top source spans {top_pairs} pairs but only "
            f"{n_sharing} pairs share a source"
        )
    if not n_sharing_without <= n_sharing:
        raise AssertionError(
            f"internal inconsistency: removing a source raised the shared-pair count "
            f"({n_sharing_without} > {n_sharing})"
        )

    out = pd.Series(
        {
            "corpus": corpus.name,
            "weighting": corpus.weighting,
            "n_units": int(n_units),
            "n_pairs": int(n_pairs),
            "n_pairs_sharing": n_sharing,
            "pair_sharing_rate": _rate(n_sharing, n_pairs),
            "top_source": str(top_key),
            "top_source_units": k,
            "top_source_pairs": int(top_pairs),
            "top_source_share_of_sharing": _rate(top_pairs, n_sharing),
            "n_pairs_sharing_without_top": n_sharing_without,
            "share_of_sharing_lost_without_top": _rate(
                n_sharing - n_sharing_without, n_sharing
            ),
            "n_sources_feeding_multiple_units": int((ups > 1).sum()),
        }
    )
    return _stamp(out, corpus, note="pair-sharing concentrated in the single largest source")


def design_summary(corpus: Corpus) -> pd.Series:
    """
    Combine incidence counts, crossing, pair sharing, and source-removal sensitivity.

    Parameters:
    ----------
    corpus : Corpus - the declared design and its retained record weights.

    Returns:
    -------
    pd.Series - the fields from :meth:`Corpus.describe`, concentration summaries,
    source-grain context, ``rows_in_spanning_sources``, and
    ``weight_in_spanning_sources``. For group-pure units, also includes group
    sizes, within-group sharing, worst source-removal losses, extrema of the
    within-group sharing rates, and cross-group pair-sharing counts and rate.
    ``units_group_pure`` records whether the group-specific summaries were available.

    Notes:
    ------
    Pair and unit counts use binary membership, but record counts retain duplicate
    multiplicity and ``weight_in_spanning_sources`` sums actual row weights.
    That mass field is an explicit exception to the structural output's general
    weight-insensitive annotation.

    If units carry inconsistent group labels, the overall structural summaries
    remain available while group-specific fields are undefined. A within-group
    rate requires at least two units; its min-to-max gap requires at least two
    finite group rates. No feature matrix or classifier is used.
    """
    out: dict[str, object] = dict(corpus.describe())
    out["grain"] = "the corpus's `source` column"

    conc = concentration(corpus)
    for key in (
        "n_pairs",
        "n_pairs_sharing",
        "pair_sharing_rate",
        "top_source",
        "top_source_units",
        "top_source_pairs",
        "top_source_share_of_sharing",
        "share_of_sharing_lost_without_top",
        "n_sources_feeding_multiple_units",
    ):
        out[key] = conc[key]

    span = spanning_source_table(corpus)
    out["rows_in_spanning_sources"] = int(span["n_rows"].sum()) if len(span) else 0
    if len(span):
        mass = corpus.frame[corpus.frame["source"].isin(span.index)]["weight"]
        out["weight_in_spanning_sources"] = float(mass.sum())
    else:
        out["weight_in_spanning_sources"] = 0.0

    try:
        ug = _unit_group(corpus)
    except ValueError:
        out["units_group_pure"] = False
        for g in corpus.groups:
            out[f"n_units[{g}]"] = float("nan")
            out[f"n_pairs[{g}]"] = float("nan")
            out[f"n_pairs_sharing[{g}]"] = float("nan")
            out[f"pair_sharing_rate[{g}]"] = float("nan")
            out[f"worst_removal_fraction[{g}]"] = float("nan")
            out[f"worst_removal_source[{g}]"] = None
        out["pair_sharing_rate_min"] = float("nan")
        out["pair_sharing_rate_max"] = float("nan")
        out["pair_sharing_rate_gap"] = float("nan")
        out["n_cross_group_pairs"] = float("nan")
        out["n_cross_group_pairs_sharing"] = float("nan")
        out["cross_group_pair_sharing_rate"] = float("nan")
        return _stamp(pd.Series(out), corpus, note="per-group fields undefined: units span groups")

    out["units_group_pure"] = True
    ps = pair_sharing_by_group(corpus)
    lodo = structural_leave_one_source_out(corpus)
    for g in corpus.groups:
        out[f"n_units[{g}]"] = int(ps.at[g, "n_units"])
        out[f"n_pairs[{g}]"] = int(ps.at[g, "n_pairs"])
        out[f"n_pairs_sharing[{g}]"] = int(ps.at[g, "n_pairs_sharing"])
        out[f"pair_sharing_rate[{g}]"] = float(ps.at[g, "pair_sharing_rate"])
        col = lodo[f"frac_emptied_{g}"]
        if col.notna().any():
            worst = col.idxmax()
            out[f"worst_removal_fraction[{g}]"] = float(col.at[worst])
            out[f"worst_removal_source[{g}]"] = str(worst)
        else:
            out[f"worst_removal_fraction[{g}]"] = float("nan")
            out[f"worst_removal_source[{g}]"] = None

    rates = ps["pair_sharing_rate"].dropna()
    out["pair_sharing_rate_min"] = float(rates.min()) if len(rates) else float("nan")
    out["pair_sharing_rate_max"] = float(rates.max()) if len(rates) else float("nan")
    out["pair_sharing_rate_gap"] = (
        float(rates.max() - rates.min()) if len(rates) >= 2 else float("nan")
    )

    co = shared_source_matrix(corpus).to_numpy()
    same = _group_dummies(corpus, ug).to_numpy()
    same = same @ same.T 
    iu = np.triu_indices(co.shape[0], 1)
    cross = same[iu] == 0
    n_cross = int(cross.sum())
    out["n_cross_group_pairs"] = n_cross
    out["n_cross_group_pairs_sharing"] = int(((co[iu] > 0) & cross).sum())
    out["cross_group_pair_sharing_rate"] = _rate(out["n_cross_group_pairs_sharing"], n_cross)

    return _stamp(
        pd.Series(out),
        corpus,
        note="counting diagnostics only; no metric, no null model, no measurement",
    )
