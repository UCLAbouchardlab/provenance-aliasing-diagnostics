"""
Balanced-accuracy ceiling under provenance invariance.

The module measures the most any provenance-invariant method could achieve. Suppose a correction
method worked, and the representation it produced carried no information about where each unit came from.
How well could the group contrast still be classified? The answer depends only on how groups are 
distributed across sources, so it is a property of the data-integration design itself, not of any feature set
or classifier. On a fully aliased corpus, the ceiling equals chance. On a partially aliased corpus, it is above
chance only by what the crossing between sources and groups can support. 

Computation:
-----------
``BA(q)`` is concave and piecewise linear in ``q``, with kinks only at the
per-source compositions ``p(A | d)``. Evaluating it at ``{0, 1}`` and every
``p(A | d)`` therefore gives the exact maximum. No grid or optimizer is
involved. The cost is O(S^2) for ``S`` sources.

Properties
----------
- The ceiling lies in ``[0.5, 1]``. ``BA(0) = 0.5`` always, so it can never
  fall below chance.
- It equals 0.5 exactly when no source with positive weight contains both
  groups (full aliasing). Then every ``q`` is optimal, and the reported
  ``q_star`` is 0.
- It equals 1 exactly when every source has the same group composition, so
  that sources carry no group information.
- It does not depend on which group is called positive. Swapping the labels
  maps ``q`` to ``1 - q``.
- It is defined at the grain of the corpus ``source`` column and under the
  corpus weighting. Changing either changes the number.

Crossing indices and the loose relaxation:
-----------------------------------------
For each group,

    kappa_g = E_{D | G=g} [ 1 - p(g | D) ]

is the expected share of the other group in the source of a randomly drawn
unit from group ``g``. It is 0 when group ``g`` never shares a source with
the other group. The closed form ``1/2 + 1/2 (kappa_A + kappa_B)`` is never
below the exact ceiling and equals it at both extremes above. In between it
overstates the excess over chance, and ``CeilingResult.relaxation_slack``
reports by how much. Report the exact ceiling; the relaxation is there for
intuition.

Weighting:
---------
Under incidence weighting every ``(source, unit)`` row carries equal mass.
That is rarely true when one database source supplies most of one group.
The metadata cannot settle the real masses, so :func:`ceiling_sensitivity`
recomputes the ceiling across a range of hypothetical masses for one source
instead of estimating them.

Contents
--------
CHANCE - 0.5, the balanced accuracy of any provenance-invariant rule on a corpus
with no crossing.
CeilingResult - exact ceiling, maximizing ``q``, both arms, relaxation, crossing indices,
priors and provenance fields, with ``summary()`` and ``as_dict()``.
balanced_accuracy_ceiling(corpus, positive_group=None) - the ceiling for a two-group corpus.
ceiling_sensitivity(corpus, source, masses, positive_group=None) - the ceiling as one source's total 
mass sweeps over hypothetical values.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .incidence import Corpus

CHANCE: float = 0.5
#Balanced accuracy of any provenance-invariant rule on a design with no crossing

_GRID = 2001
_TIE_TOL = 1e-12


@dataclass(frozen=True)
class CeilingResult:
    """
    The exact balanced-accuracy ceiling for one corpus, one weighting, and one choice of positive group.

    Returned by :func:`balanced_accuracy_ceiling`. Every field is a scalar, so instances can be compared and hashed.

    Attributes:
    -----------
    exact : float - ``max_q BA(q)`` over the provenance-invariance positive rate ``q``. Always in ``[0.5, 1.0]``. 
    ``nan`` if contrast is degenerate.
    q_star : float - maximizing ``q``. when several ``q`` tie, the smallest candidate wins. on a fully aliased corpus, 
    every ``q`` gives 0.5, so ``q_star`` is 0, with sensitivity 0 and specificity 1
    sensitivity, specificity : float - the two arms at ``q_star``. they are computed by the same expression as ``exact``
    chance : float - 0.5, the balanced accuracy of any rule on a corpus with no crossing
    n_groups : int - number of groups
    weighting : str - the corpus that defines mass behind every number
    corpus : str - corpus name
    positive_group, negative_group : str - group labels, converted with ``str()``
    prior_positive : float - weighted share of positive group, ``P(G = positive)``
    kappa_positive, kappa_negative : float - crossing index of each group 
    n_sources : int - rows in the source table, including sources with zero weight
    n_zero_weight_sources : int - sources whose total weight is not positive. nan totals are counted here.
    """

    exact: float
    q_star: float
    sensitivity: float
    specificity: float
    relaxation: float
    chance: float = CHANCE
    n_groups: int = 2
    weighting: str = "incidence rows"
    corpus: str = "corpus"
    positive_group: str = ""
    negative_group: str = ""
    prior_positive: float = float("nan")
    kappa_positive: float = float("nan")
    kappa_negative: float = float("nan")
    n_sources: int = 0
    n_zero_weight_sources: int = 0

    @property
    def excess_over_chance(self) -> float:
        """
        ``exact - 0.5``, the only part of the ceiling that carries information.

        It is 0 on a fully aliased corpus and ``nan`` when ``exact`` is
        ``nan``.
        """
        return self.exact - self.chance

    @property
    def relaxation_slack(self) -> float:
        """
        Fractional overstatement of the excess over chance by the loose relaxation.

        Computed as ``(relaxation - 0.5) / (exact - 0.5) - 1``. A value of 0
        means the relaxation is tight. A value of 1 means it claims twice the
        real excess. ``nan`` when the excess is zero or not finite, since the
        ratio is undefined when ``exact`` is at chance.
        """
        excess = self.excess_over_chance
        if not np.isfinite(excess) or excess <= 0:
            return float("nan")
        return (self.relaxation - self.chance) / excess - 1.0

    def summary(self) -> str:
        """
        One-line report of the ceiling.
        """
        return (
            f"{self.corpus}: balanced-accuracy ceiling = {self.exact:.4f} at q* = "
            f"{self.q_star:.4f} (sensitivity {self.sensitivity:.4f}, specificity "
            f"{self.specificity:.4f}; chance {self.chance:.2f}), positive group "
            f"{self.positive_group!r} against {self.negative_group!r}; loose relaxation "
            f"{self.relaxation:.4f}; {self.n_sources} sources; weighting: "
            f"{self.weighting}; grain: whatever the corpus 'source' column encodes"
        )

    def as_dict(self) -> dict:
        """
        All fields plus ``excess_over_chance`` and ``relaxation_slack`` as a dict, for tables or JSON.

        Values can be ``nan``, which strict JSON does not allow.
        """
        return {
            "corpus": self.corpus,
            "weighting": self.weighting,
            "exact": self.exact,
            "q_star": self.q_star,
            "sensitivity": self.sensitivity,
            "specificity": self.specificity,
            "relaxation": self.relaxation,
            "chance": self.chance,
            "excess_over_chance": self.excess_over_chance,
            "relaxation_slack": self.relaxation_slack,
            "n_groups": self.n_groups,
            "positive_group": self.positive_group,
            "negative_group": self.negative_group,
            "prior_positive": self.prior_positive,
            "kappa_positive": self.kappa_positive,
            "kappa_negative": self.kappa_negative,
            "n_sources": self.n_sources,
            "n_zero_weight_sources": self.n_zero_weight_sources,
        }

def _binary_table(
    corpus: Corpus, positive_group: object | None
) -> tuple[pd.DataFrame, object, object]:
    """
    Source x group weight table for a two-group corpus, with the positive group as the first column.

    Parameters:
    ----------
    corpus : Corpus - must have exactly two groups.
    positive_group : object or None - label of the positive group. None selects ``corpus.groups[0]``.

    Returns:
    -------
    table : pd.DataFrame, shape (n_sources, 2) - ``corpus.source_group_weight`` with columns ``[positive, negative]``.
    positive_group, negative_group : object - the resolved labels, not converted to strings.

    Raises:
    ------
    ValueError
    - If the corpus does not have two groups 
    - If ``positive_group`` is not one of them
    """
    groups = corpus.groups
    if len(groups) != 2:
        raise ValueError(
            f"balanced accuracy is a two-group contrast, but corpus {corpus.name!r} has "
            f"{len(groups)} groups: {groups}. Collapse them with Corpus.regroup(...) or "
            f"restrict with Corpus.subset(groups=[...]) and say in the caption which "
            f"contrast the ceiling refers to."
        )
    
    if positive_group is None:
        positive_group = groups[0]

    elif positive_group not in groups:
        raise ValueError(
            f"positive_group {positive_group!r} is not a group of {corpus.name!r}; "
            f"groups are {groups}"
        )
    
    negative_group = groups[0] if groups[1] == positive_group else groups[1]
    return corpus.source_group_weight[[positive_group, negative_group]], positive_group, negative_group


def _kappas(table: pd.DataFrame) -> tuple[float, float]:
    """
    Crossing index of each group, positive group first.

        kappa_g = E_{D | G=g}[1 - p(G=g | D)]

    ``kappa_g`` is the expected share of the other group in the source of a
    randomly drawn unit from group ``g``. The expectation uses the weight
    distribution of group ``g`` across sources, so a source contributing a
    lot of group ``g`` counts proportionally. A source whose rows all belong
    to one group has ``1 - p(g | d) = 0`` and drops out, exactly as it does
    from ``H(G | D)``. ``kappa_g = 0`` means group ``g`` never shares a source
    with the other group.

    Parameters:
    ----------
    table : pd.DataFrame, shape (n_sources, 2) - non-negative weights, positive group first. 
    sources whose total weight is not positive are ignored.

    Returns:
    -------
    (float, float) - ``(kappa_positive, kappa_negative)``, each in ``[0, 1]``. a group with
    no weight gets ``nan``.

    Notes:
    -----
    Negative weights are not rejected and give meaningless values.
    """
    w = table.to_numpy(dtype=float)
    n_d = w.sum(axis=1)
    live = n_d > 0
    out = []
    for j in range(2):
        w_g = w[live, j]
        denom = w_g.sum()
        if denom <= 0:
            out.append(float("nan"))
            continue
        p_g = w_g / n_d[live]
        out.append(float((w_g * (1.0 - p_g)).sum() / denom))
    return out[0], out[1]


def _exact_ceiling(table: pd.DataFrame, grid: int = _GRID) -> dict:
    """
    Compute ``max_q BA(q)`` exactly over the source-invariant positive rate ``q``.

    Let ``w_d = P(D = d)`` and ``p_d = p(positive | d)``. The class priors are
    ``pi_pos = sum_d w_d p_d`` and ``pi_neg = 1 - pi_pos``. The two arms are

        sens(q) = sum_d w_d min(q, p_d) / pi_pos
        spec(q) = sum_d w_d min(1 - q, 1 - p_d) / pi_neg

    and ``BA(q)`` is their average. Both arms are concave and piecewise linear
    in ``q``, with kinks only at the ``p_d``. The maximum is attained at some ``p_d`` 
    or at an endpoint, and evaluating ``{0, 1} ∪ {p_d}`` gives the exact value. 
    No grid search is involved.

    Parameters:
    ----------
    table : pd.DataFrame, shape (n_sources, 2) - weights, positive group first. sources
    with non-positive or NaN total weight are dropped without a warning. negative entries are 
    not rejected
    grid : int - ignored. the breakpoint evaluation is exact.

    Returns:
    -------
    dict - keys ``exact``, ``q_star``, ``sensitivity``, ``specificity``, ``prior_positive``, 
    ``n_sources``, and ``n_zero_weight_sources``. the first four are ``nan`` if no source 
    carries weight or either group has zero prior. 
    """
    w = table.to_numpy(dtype=float)
    n_d = w.sum(axis=1)
    live = n_d > 0
    n_zero = int((~live).sum())

    blank = {
        "exact": float("nan"), "q_star": float("nan"),
        "sensitivity": float("nan"), "specificity": float("nan"),
        "prior_positive": float("nan"),
        "n_sources": int(len(n_d)), "n_zero_weight_sources": n_zero,
    }
    if not live.any():
        return blank

    n_live = n_d[live]
    p_pos = w[live, 0] / n_live                      # p(positive | d)
    weight = n_live / n_live.sum()                   # P(D = d)
    pi_pos = float((weight * p_pos).sum())
    
    pi_neg = float((weight * (1.0 - p_pos)).sum())
    if not (pi_pos > 0.0 and pi_neg > 0.0):
        # one group carries no weight - undef
        return {**blank, "prior_positive": pi_pos}

    #maximizers. objective is concave and piecewise linear. at p(positive | d), so {0, 1} together 
    #gives the exact max. 
    qs = np.unique(
        np.concatenate([[0.0, 1.0], p_pos])
    )

    def arms(q: float) -> tuple[float, float]:
            sens = float((weight * np.minimum(q, p_pos)).sum()) / pi_pos
            spec = float((weight * np.minimum(1.0 - q, 1.0 - p_pos)).sum()) / pi_neg
            return sens, spec
    
    vals = np.array(
        [0.5*sum(arms(q)) for q in qs], dtype = float,
    )

    i_best = int(np.argmax(vals))

    q_star = float(qs[i_best])
    sens, spec = arms(q_star)

    return {
        "exact": float(vals[i_best]),
        "q_star": q_star,
        "sensitivity": sens,
        "specificity": spec,
        "prior_positive": pi_pos,
        "n_sources": int(len(n_d)),
        "n_zero_weight_sources": n_zero,
    }

def balanced_accuracy_ceiling(
    corpus: Corpus,
    positive_group: object | None = None,
    *,
    grid: int = _GRID,
) -> CeilingResult:
    """
    Highest balanced accuracy any provenance-invariant classifier can reach on this corpus.

    The bound holds for every prediction ``Ghat = f(Z)`` whose representation
    ``Z`` is independent of the source ``D``. Any randomness in ``f`` must
    also be independent of ``D``. Under that constraint, the positive rate
    ``q = P(Ghat = positive | D = d)`` is the same for every source. Within
    source ``d``, a classifier that flags a fraction ``q`` as positive can
    cover at most ``min(q, p(positive | d))`` of the true-positive mass and
    at most ``min(1 - q, p(negative | d))`` of the true-negative mass. It
    reaches both at once only by ranking every true positive above every true
    negative within each source. Maximizing the resulting ``BA(q)`` over
    ``q`` gives the ceiling, which is tight within the constraint. See
    :func:`_exact_ceiling` for why evaluating the breakpoints is exact.

    Parameters:
    ----------
    corpus : Corpus - a two-group corpus. all masses come from its weighting
    (``corpus.weighting``), at the grain of its source column.
    positive_group : object, optional - group counted as positive. defaults to 
    ``corpus.groups[0]``. the ceiling is the same whichever group is positive; 
    only the labels, ``q_star`` and the arm order change.
    grid : int, keyword-only - passed through but ignored. the computation is exact without a grid.

    Returns
    -------
    CeilingResult - ``exact`` is ``nan`` when no source carries weight or one group has
    no weight.

    Raises:
    ------
    ValueError
    - If the corpus does not have exactly two groups
    - ``positive_group`` is not one of the groups

    Notes:
    ------
    The bound is the limit under exact invariance. ComBat and domain-adversarial training 
    approach provenance invariance but do not reach it. A method that partly removes source
    information is not covered by this bound and can score above it. A reported accuracy above the
    ceiling, by more than its sampling uncertainty, is evidence of provenance leakage, not a 
    better method. 
    """
    table, pos, neg = _binary_table(corpus, positive_group)
    num = _exact_ceiling(table, grid=grid)
    k_pos, k_neg = _kappas(table)
    relaxation = (
        CHANCE + 0.5 * (k_pos + k_neg)
        if np.isfinite(k_pos) and np.isfinite(k_neg)
        else float("nan")
    )
    return CeilingResult(
        exact=num["exact"],
        q_star=num["q_star"],
        sensitivity=num["sensitivity"],
        specificity=num["specificity"],
        relaxation=relaxation,
        chance=CHANCE,
        n_groups=2,
        weighting=corpus.weighting,
        corpus=corpus.name,
        positive_group=str(pos),
        negative_group=str(neg),
        prior_positive=num["prior_positive"],
        kappa_positive=k_pos,
        kappa_negative=k_neg,
        n_sources=num["n_sources"],
        n_zero_weight_sources=num["n_zero_weight_sources"],
    )


def ceiling_sensitivity(
    corpus: Corpus,
    source: object,
    masses: Iterable[float],
    *,
    positive_group: object | None = None,
    grid: int = _GRID,
) -> pd.DataFrame:
    """
    Recompute the ceiling with one source's total data mass set to each of several hypothetical values.

    The ceiling uses the corpus weighting. Incidence weighting assumes every
    ``(source, unit)`` row carries equal data mass, which is almost never true
    when one database source supplies most of one group. This function makes
    that assumption visible. For each value in ``masses``, it rescales
    ``source`` so its total weight equals that value, leaves every other
    source unchanged, and recomputes. The source's own group split
    ``p(positive | source)`` stays fixed; what changes is its share
    ``P(D = source)``, and with it the class priors. Nothing is estimated from
    data. It is a sweep over an assumption the metadata cannot pin down.

    Parameters:
    ----------
    corpus : Corpus - a two-group corpus.
    source : object - tndex label of the source to re-weight. Tuple-valued composite labels
    are supported.
    masses : iterable of float - hypothetical total weights, each finite and ``>= 0``. 
    mass 0 removes the source. include the observed mass to anchor the sweep: that row
    reproduces :func:`balanced_accuracy_ceiling`. it must be an iterable,
    so pass ``[x]`` rather than a bare scalar. An empty iterable raises ``KeyError``.
    positive_group, grid : optional, keyword-only - as in :func:`balanced_accuracy_ceiling`. ``grid`` is ignored.

    Returns:
    -------
    pd.DataFrame - one row per mass, indexed by ``mass``. Repeated masses give repeated
    index values. Columns:
    - ``share_of_total_weight``: the source's share after rescaling.
    - ``exact``, ``q_star``, ``sensitivity``, ``specificity``,
        ``relaxation``, ``prior_positive``: as in :class:`CeilingResult`.
    - ``observed_mass``, ``weighting``: the same in every row.

    ``DataFrame.attrs`` records the corpus, source, groups, observed
    mass and weighting. Many pandas operations drop ``attrs``, so copy
    them out before transforming the frame.

    Raises:
    ------
    KeyError
    - If ``source`` is not in the corpus, or ``masses`` is empty.
    ValueError
    - If the corpus does not have two groups, ``positive_group`` is
    invalid, the source has zero observed weight, or any mass is
    negative or non-finite.
    TypeError
    - If the weight table has an integer dtype and a
    rescaled weight is not a whole number.
    """
    table, pos, neg = _binary_table(corpus, positive_group)
    if source not in table.index:
        known = list(table.index[:8])
        raise KeyError(
            f"source {source!r} is not in corpus {corpus.name!r} "
            f"({len(table.index)} sources, e.g. {known})"
        )
    # A tuple-valued composite source is one object label, but ``.loc[tuple]`` is
    # parsed by pandas as a multi-axis indexer.  List wrapping selects that one row.
    row = table.loc[[source]].iloc[0].to_numpy(dtype=float)
    observed = float(row.sum())
    if observed <= 0:
        raise ValueError(
            f"source {source!r} carries zero weight in {corpus.name!r}, so its group "
            f"split is undefined and cannot be rescaled"
        )

    masses = [float(m) for m in masses]
    bad = [m for m in masses if not np.isfinite(m) or m < 0]
    if bad:
        raise ValueError(f"masses must be finite and non-negative; got {bad}")

    rows = []
    for m in masses:
        scaled = table.copy()
        scaled_row = row * (m / observed)
        for column, value in zip(scaled.columns, scaled_row):
            scaled.at[source, column] = value
        num = _exact_ceiling(scaled, grid=grid)
        k_pos, k_neg = _kappas(scaled)
        total = float(scaled.to_numpy(dtype=float).sum())
        rows.append(
            {
                "mass": m,
                "share_of_total_weight": (m / total) if total > 0 else float("nan"),
                "exact": num["exact"],
                "q_star": num["q_star"],
                "sensitivity": num["sensitivity"],
                "specificity": num["specificity"],
                "relaxation": (
                    CHANCE + 0.5 * (k_pos + k_neg)
                    if np.isfinite(k_pos) and np.isfinite(k_neg)
                    else float("nan")
                ),
                "prior_positive": num["prior_positive"],
                "observed_mass": observed,
                "weighting": corpus.weighting,
            }
        )

    out = pd.DataFrame(rows).set_index("mass")
    out.attrs.update(
        corpus=corpus.name,
        source=str(source),
        positive_group=str(pos),
        negative_group=str(neg),
        observed_mass=observed,
        weighting=corpus.weighting,
    )
    return out
