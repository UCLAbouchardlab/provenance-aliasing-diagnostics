"""
Association and design checks for provenance aliasing.

Which familiar warning signs would appear in this source x group design? The
module places contingency-table association, a design-rank check, source sizes,
residual label entropy, and the binary provenance-invariant ceiling beside one
another. Each answers a different question about the same declared design.

Computation:
-----------
Cramer's V measures symmetric association between source and group. Theil's U
measures the fraction of group entropy explained by source. The design-rank
check asks whether source indicators and group indicators provide independent
columns in the specified additive design. Minimum batch size counts distinct
analysis units per source.

Weighting:
---------
V, U, R, and the ceiling use the corpus row weights. The design-rank and batch-size
checks use recorded membership and include zero-weight observations. Duplicate
records can change weighted association without creating additional distinct units.

Interpretation:
---------------
Guard thresholds describe structural extremes; they are not fitted decision rules
or significance tests. A singleton source raises a replication question without
by itself proving source-group aliasing. A non-firing guard does not validate a
biological conclusion. Binary equivalences must not be assumed for every
multigroup design, especially when numerical rank has not been computed.

Contents:
---------
cramers_v - uncorrected or finite-sample-corrected association.
theils_u - directional label uncertainty explained by source.
min_batch_size - distinct units per source.
design_rank_deficient - additive design rank and structural context.
guard_report - a common table of values, thresholds, firing states, and notes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .entropy import label_entropy
from .incidence import Corpus
from .ceiling import balanced_accuracy_ceiling

_TOL = 1e-12

def cramers_v(corpus: Corpus, *, bias_corrected: bool = False) -> float:
    """
    Cramer's V for the weighted source x group contingency table.

    Let n be total table mass, r and c the numbers of positive-mass rows and columns,
    and chi2 the Pearson contingency statistic. The uncorrected value is

        V = sqrt((chi2 / n) / (min(r, c) - 1))

    Parameters:
    ----------
    corpus : Corpus - the source x group table and its declared weighting.
    bias_corrected : bool, default False, keyword-only - apply the finite-sample
    correction to phi-squared and the effective row and column dimensions.

    Returns:
    -------
    float - association in ``[0, 1]``. Returns ``nan`` when fewer than two
    positive-mass sources or groups remain, total mass is not positive, or the
    corrected denominator is not positive.

    Computation:
    -----------
    The correction uses ``phi2_t = max(0, chi2/n - (r-1)*(c-1)/(n-1))`` and
    ``r_t = r - (r-1)^2/(n-1)``, with the analogous expression for c.
    The corrected denominator is ``min(r_t, c_t) - 1``; n must exceed 1.

    Notes:
    ------
    Zero-mass rows and columns are removed before the calculation. In a binary
    contrast with both groups and at least two sources of positive mass, V reaches
    1 when each source determines the group. That equivalence needs qualification
    for more than two groups.

    Uncorrected V is unchanged by uniform rescaling of weights. The finite-sample
    correction depends on n, so arbitrary mass weights should not automatically
    be interpreted as counts of independent observations.
    """
    table = corpus.source_group_weight.to_numpy(dtype=float)
    table = table[table.sum(axis=1) > 0, :]
    if table.size:
        table = table[:, table.sum(axis=0) > 0]
    if table.size == 0:
        return float("nan")

    r, c = table.shape
    n = float(table.sum())
    if r < 2 or c < 2 or n <= 0:
        return float("nan")

    expected = np.outer(table.sum(axis=1), table.sum(axis=0)) / n
    chi2 = float((((table - expected) ** 2) / expected).sum())
    phi2 = chi2 / n

    if not bias_corrected:
        v = np.sqrt(phi2 / (min(r, c) - 1))
        return float(np.clip(v, 0.0, 1.0))

    if n <= 1:
        return float("nan")
    phi2_t = max(0.0, phi2 - (r - 1) * (c - 1) / (n - 1))
    r_t = r - (r - 1) ** 2 / (n - 1)
    c_t = c - (c - 1) ** 2 / (n - 1)
    denom = min(r_t, c_t) - 1
    if denom <= 0:
        return float("nan")
    return float(np.clip(np.sqrt(phi2_t / denom), 0.0, 1.0))


def theils_u(corpus: Corpus) -> float:
    """
    The fraction of label entropy explained by provenance, ``U(G|D)``.

    Parameters:
    ----------
    corpus : Corpus - the weighted incidence design at the declared source grain.

    Returns:
    -------
    float - ``1 - H(G|D)/H(G)`` from :func:`label_entropy`. It is 1 when source
    fully determines group and 0 when source carries no group information.
    Returns ``nan`` when marginal label entropy is zero.

    Notes:
    ------
    The direction matters: U(G|D) need not equal U(D|G). This is an entropy
    fraction, not a classification accuracy. Weight-validation errors propagate
    from the entropy calculation.
    """
    return float(label_entropy(corpus).uncertainty_coefficient)


def min_batch_size(corpus: Corpus) -> pd.Series:
    """
    Count distinct analysis units in each source and list the smallest sources first.

    Parameters:
    ----------
    corpus : Corpus - binary source x unit membership defines each batch size.

    Returns:
    -------
    pd.Series - integer counts named ``batch_size``, indexed by source and sorted
    stably in increasing order. Attributes state the definition, corpus, and weighting.

    Notes:
    ------
    Repeated rows do not increase a source's unit count, and a zero-weight row still
    records membership. A source with one unit provides no within-source replication
    at this unit grain; it does not alone establish that source determines group.
    """
    s = corpus.units_per_source().astype(int).rename("batch_size")
    s = s.sort_values(ascending=True, kind="mergesort")
    s.attrs.update(
        {
            "definition": "distinct analysis units contributed by each source",
            "weighting": corpus.weighting,
            "corpus": corpus.name,
        }
    )
    return s


def design_rank_deficient(corpus: Corpus, *, max_cells: int = 50_000_000) -> dict[str, object]:
    """
    Check the rank of an additive source-plus-group indicator design.

    The matrix contains one indicator for every source and all but one group
    indicator, with one row per retained observation. The source indicators already
    span an intercept, so the expected full column count is

        n_columns = n_sources + max(n_groups - 1, 0)

    Parameters:
    ----------
    corpus : Corpus - the source and group memberships defining the design.
    max_cells : int, default 50000000, keyword-only - maximum number of matrix
    cells before dense rank computation is skipped.

    Returns:
    -------
    dict - ``rank``, ``n_columns``, ``deficient``, the structural shortcut
    ``structural_deficient``, and ``rank_test_agrees``. Also includes design counts,
    minimum source size, singleton-source counts and flag, ``failure_mode``,
    ``note``, corpus name, and weighting. ``rank`` is ``nan`` and agreement is
    None when numerical rank is skipped.

    Computation:
    -----------
    For a matrix within the size limit, ``np.linalg.matrix_rank`` determines
    whether rank is below the column count. A separate structural shortcut asks
    whether there are no sources spanning groups. Above the size limit,
    ``deficient`` uses that shortcut; no numerical rank has been established.

    Notes:
    ------
    The no-spanning-source shortcut characterizes complete aliasing for a binary
    contrast. With multiple groups, rank deficiency can also arise from disconnected
    sets of groups even when some sources span groups. In that case the shortcut
    can miss a deficiency, so inspect ``rank`` and ``rank_test_agrees``.

    The matrix is unweighted and includes zero-weight rows. Singleton batches and
    rank deficiency are reported separately because they concern different failures.
    With fewer than two groups, there is no group contrast to assess.
    """
    frame = corpus.frame
    n_rows = corpus.n_rows
    n_sources = corpus.n_sources
    n_groups = corpus.n_groups
    n_columns = n_sources + max(n_groups - 1, 0)

    sizes = min_batch_size(corpus)
    n_singleton = int((sizes <= 1).sum())
    smallest = int(sizes.min())
    singleton_block = smallest <= 1

    if n_groups < 2:
        structural: bool | None = None
    else:
        structural = len(corpus.spanning_sources()) == 0

    rank: float | int
    if n_rows * max(n_columns, 1) > max_cells:
        rank = float("nan")
        deficient = bool(structural) if structural is not None else False
        agrees: bool | None = None
        size_note = (
            f" The design would have {n_rows} x {n_columns} entries, above max_cells="
            f"{max_cells}, so the numerical rank was not computed and `deficient` is the "
            "exact structural test."
        )
    else:
        src = pd.Categorical(frame["source"], categories=corpus.sources)
        grp = pd.Categorical(frame["group"], categories=corpus.groups)
        batch = pd.get_dummies(src).to_numpy(dtype=float)
        cond = pd.get_dummies(grp, drop_first=True).to_numpy(dtype=float)
        design = np.hstack([batch, cond]) if cond.size else batch
        rank = int(np.linalg.matrix_rank(design))
        deficient = bool(rank < n_columns)
        agrees = None if structural is None else bool(deficient == structural)
        size_note = ""

    modes: list[str] = []
    if n_groups < 2:
        modes.append("undefined: one group, so there is no contrast to confound")
    elif deficient:
        modes.append("rank deficiency: the group label is determined by the source")
    if singleton_block:
        modes.append(f'singleton batch: {n_singleton} of {n_sources} sources contribute one analysis unit;'
                     f'in the current ComBat implementation, a one-sample batch fources mean.only = True.'
        )
    failure_mode = "; ".join(modes) if modes else "none"

    note = (
        "ComBat builds design = [full source indicators | group indicators minus "
        "one] and stops when qr(design)$rank < ncol(design). On this design that test is "
        "equivalent to `corpus.spanning_sources() == []`, i.e. to H(G|D) = 0 exactly, so "
        "it fires only at the boundary and says nothing about how far inside it a corpus "
        "sits. The singleton-batch condition is a separate, non-confounding reason the "
        "same model refuses to run and must not be reported as evidence of aliasing."
    )
    if agrees is False:
        note += (
            " WARNING: the numerical rank disagrees with the exact structural test; the "
            "structural test is the one to trust (numerical rank tolerance)."
        )
    note += size_note

    return {
        "rank": rank,
        "n_columns": int(n_columns),
        "deficient": bool(deficient),
        "structural_deficient": structural,
        "rank_test_agrees": agrees,
        "n_rows": int(n_rows),
        "n_sources": int(n_sources),
        "n_groups": int(n_groups),
        "min_batch_size": smallest,
        "n_singleton_batches": n_singleton,
        "singleton_batch_block": bool(singleton_block),
        "failure_mode": failure_mode,
        "note": note,
        "corpus": corpus.name,
        "weighting": f"{corpus.weighting} (rank is unweighted; weights do not affect it)",
    }

def _fires(value: float | int | None, condition: bool) -> object:
    """
    Evaluate a guard condition only when its statistic is finite and numeric.

    Returns ``pd.NA`` for missing, non-convertible, or non-finite values, otherwise
    returns the Boolean condition. Undefined statistics must remain distinguishable
    from a guard that was evaluated and did not fire.
    """
    if value is None:
        return pd.NA
    try:
        if not np.isfinite(float(value)):
            return pd.NA
    except (TypeError, ValueError):
        return pd.NA
    return bool(condition)


def guard_report(
    corpus: Corpus,
    *,
    grain: str | None = None,
    ceiling: float | None = None,
) -> pd.DataFrame:
    """
    Collect seven provenance and design checks in one reporting table.

    Parameters:
    ----------
    corpus : Corpus - the design supplied to every guard.
    grain : str or None, default None, keyword-only - human-readable source
    definition. An omitted grain receives the module's undeclared-grain label.
    ceiling : float or None, default None, keyword-only - optional precomputed
    balanced-accuracy ceiling. None computes the exact binary ceiling when there
    are two groups; otherwise the ceiling value is undefined.

    Returns:
    -------
    pd.DataFrame - one row each for V, corrected V, Theil's U, design rank,
    minimum batch size, R, and ceiling. Columns are ``name``, ``value``,
    ``threshold``, ``fires``, ``what_it_means``, ``citation``, ``grain``,
    ``weighting``, and ``corpus``. ``fires`` uses pandas' nullable Boolean dtype.
    Attributes include the names and count of fired guards and design failure context.

    Thresholds:
    -----------
    V and U fire near 1; R fires near 0; the binary ceiling fires at chance, 0.5.
    These boundary comparisons use a tolerance of ``1e-12``. The batch-size guard
    fires at one or fewer units, and the rank guard fires when computed rank is
    below the required column count. An unavailable statistic gives an undefined
    firing state, including when numerical rank was skipped.

    Notes:
    ------
    The checks are descriptive comparisons implemented here; the table does not
    run external batch-correction software. Read each statistic's assumptions,
    particularly corrected V under mass weighting and rank in multigroup designs.
    A supplied ceiling is used as given, so its contrast and weighting must match
    the corpus. The ceiling threshold in this table is specifically binary.
    """
    grain_label = grain or "UNDECLARED"
    ent = label_entropy(corpus)
    v = cramers_v(corpus)
    v_bc = cramers_v(corpus, bias_corrected=True)
    u = ent.uncertainty_coefficient
    design = design_rank_deficient(corpus)
    smallest = int(design["min_batch_size"])
    n_singleton = int(design["n_singleton_batches"])
    rank = design["rank"]
    n_columns = int(design["n_columns"])

    if ceiling is None:
        if corpus.n_groups == 2:
            result = balanced_accuracy_ceiling(corpus)
            ceiling_value = float(result.exact)
            q_star = float(result.q_star)
            ceiling_note = (f'exact frontier at q* = {q_star:.4f}')
        else:
            ceiling_value = float("nan")
            ceiling_note = ("undefined here: balanced-accuracy ceiling is currently defined for a binary contrast")

    else:
        ceiling_value = float(ceiling)
        ceiling_note = "supplied by the caller"

    rows: list[dict[str, object]] = [
        {
            "name": "Cramer's V (source x group)",
            "value": v,
            "threshold": 1.0,
            "fires": _fires(v, bool(np.isfinite(v) and abs(v - 1.0) <= _TOL)),
            "what_it_means": (
                "1 exactly means the source determines the group. BatchQC's "
                "confound_metrics computes this from metadata alone and auto-excludes the "
                "variable only at 1; a corpus one shared source short of degenerate scores "
                "below 1 and passes."
            ),
            "citation": "Manimaran et al. 2016, Bioinformatics 32(24):3836 (BatchQC)",
        },
        {
            "name": "Cramer's V, Bergsma bias-corrected",
            "value": v_bc,
            "threshold": 1.0,
            "fires": _fires(v_bc, bool(np.isfinite(v_bc) and abs(v_bc - 1.0) <= _TOL)),
            "what_it_means": (
                "Same statistic with the small-sample bias removed; V is upward biased on "
                "tables with many single-row sources. The published threshold is defined "
                "on the uncorrected V, and the corrected form does not reach 1 even under "
                "perfect association, so this row will essentially never fire -- shown for "
                "the magnitude, not for the switch."
            ),
            "citation": "Bergsma 2013, J. Korean Statist. Soc. 42(3):323",
        },
        {
            "name": "Theil's U = U(G|D)",
            "value": u,
            "threshold": 1.0,
            "fires": _fires(u, bool(np.isfinite(u) and abs(u - 1.0) <= _TOL)),
            "what_it_means": (
                "Fraction of the label's entropy explained by provenance; 1 exactly means "
                "the source determines the label. Not novel -- Theil's uncertainty "
                "coefficient, and 1 - ratio below. Not a proportion of cases."
            ),
            "citation": "Theil 1970 (uncertainty coefficient)",
        },
        {
            "name": "ComBat design rank vs columns",
            "value": rank,
            "threshold": float(n_columns),
            "fires": _fires(rank, bool(design["deficient"])),
            "what_it_means": (
                "sva::ComBat stops when qr(design)$rank < ncol(design) with 'The covariate "
                "is confounded with batch!'. On this design that is equivalent to no source "
                "spanning groups, i.e. to H(G|D) = 0 exactly. Right at the boundary, silent "
                "everywhere inside it. Failure mode: " + str(design["failure_mode"])
            ),
            "citation": "Johnson, Li & Rabinovich 2007; sva::ComBat",
        },
        {
            "name": "minimum batch size (units per source)",
            "value": float(smallest),
            "threshold": 1.0,
            "fires": _fires(smallest, bool(smallest <= 1)),
            "what_it_means": (
                f"{n_singleton} of {corpus.n_sources} sources contribute a single unit, so "
                "a within-batch variance is not estimable. In the current ComBat implementation,"
                "the presence of a one-sample batch forces mean.only = True. This is a "
                "separability estimability limitation and is not itself evidence of source-group"
                "confounding."
            ),
            "citation": "ComBat precondition (version-dependent: mean.only or stop)",
        },
        {
            "name": "residual label entropy ratio H(G|D)/H(G)  [this package]",
            "value": ent.ratio,
            "threshold": 0.0,
            "fires": _fires(
                ent.ratio, bool(np.isfinite(ent.ratio) and abs(ent.ratio) <= _TOL)
            ),
            "what_it_means": (
                "Fraction of the label's entropy that survives conditioning on provenance. "
                "0 is exact aliasing -- the point where every guard above fires. Unlike "
                "them it is informative away from 0, which is the only claim made for it. "
                f"{ent.n_spanning_sources} of {ent.n_sources} sources "
                "span biological groups; "
                f"{ent.n_nested_sources} positive-weight sources are "
                "nested within a single group and therefore contribute "
                "zero conditional label entropy."
            ),
            "citation": "this package; see provenance.entropy",
        },
        {
            "name": "balanced-accuracy ceiling",
            "value": ceiling_value,
            "threshold": 0.5,
            "fires": _fires(
                ceiling_value,
                bool(np.isfinite(ceiling_value) and ceiling_value <= 0.5 + _TOL),
            ),
            "what_it_means": (
                "Best balanced accuracy any predictor built on an exactly source-invariant "
                "representation could reach on this contrast (" + ceiling_note + "). 0.5 is "
                "chance. It holds in the exact-invariance limit: a method that only "
                "partially removes provenance can score higher, precisely by keeping the "
                "signal the correction was meant to remove. It is a design-level number "
                "and moves with the data mass assumed per row."
            ),
            "citation": "this package; source-invariant positive-rate argument",
        },
    ]

    out = pd.DataFrame(rows)
    out["fires"] = pd.array(out["fires"].tolist(), dtype="boolean")
    out["grain"] = grain_label
    out["weighting"] = corpus.weighting
    out["corpus"] = corpus.name
    out = out[
        [
            "name",
            "value",
            "threshold",
            "fires",
            "what_it_means",
            "citation",
            "grain",
            "weighting",
            "corpus",
        ]
    ]

    fired = out.loc[out["fires"].fillna(False).astype(bool), "name"].tolist()
    out.attrs.update(
        {
            "corpus": corpus.name,
            "weighting": corpus.weighting,
            "grain": grain_label,
            "fired": fired,
            "n_fired": len(fired),
            "failure_mode": design["failure_mode"],
        }
    )
    return out