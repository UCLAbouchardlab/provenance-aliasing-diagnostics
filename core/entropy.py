"""
Residual label entropy after conditioning on provenance.

How much of the group contrast remains once the source of each observation is known?
The answer is computed from the source x group weight table. No molecular measurement,
feature model, or classifier is required, so the same calculation applies to an atlas
of peptides, metabolites, transcripts, or cells once its analytical roles are declared.

Computation:
-----------
Let ``w[d, g]`` be the retained mass for source ``d`` and group ``g``, and let
``W = sum(w)``. Normalizing this table defines the empirical distribution of ``D``
and ``G``. All entropies use base-2 logarithms and are reported in bits.

    H(G|D) = sum_d p(d) H(G|D=d)
    R = H(G|D) / H(G)
    U(G|D) = 1 - R

Properties:
-----------
- For positive marginal label entropy, ``R`` lies in ``[0, 1]``. It is 0 when
  every positive-weight source contains only one group and 1 when the weighted
  group distribution is the same in every positive-weight source.
- ``U(G|D)`` is the fraction of label entropy explained by source. It is not
  classification accuracy or a proportion of correctly classified observations.
- When ``H(G) = 0``, the normalized quantities are undefined and return ``nan``.
- Counts of spanning and nested sources use positive group mass. A zero-weight
  observation can create a structural connection without contributing entropy.

Weighting and interpretation:
-----------------------------
Every retained row has mass 1 unless explicit weights were supplied to the corpus.
Repeated retained records therefore affect the result. Neither equal row weighting
nor a large number of rows establishes independent biological replication.
Report the source grain, unit definition, duplicate policy, and weighting beside R.

Contents:
---------
shannon - entropy of a non-negative mass vector.
EntropyResult, label_entropy - the full diagnostic and its source contributions.
leave_one_source_out - change in the diagnostic after removing each source.
BootstrapResult, bootstrap_ratio - sensitivity to resampling whole sources.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .incidence import Corpus

def shannon(p) -> float:
    """
    Shannon entropy of a finite, non-negative mass vector, in bits.

    The input is flattened and normalized internally. Zero entries contribute zero,
    so callers may supply counts, weights, or an already normalized distribution.

    Parameters:
    ----------
    p : array-like - non-negative masses. All entries must be finite.

    Returns:
    -------
    float - ``-sum(p_i * log2(p_i))`` after normalization. Returns ``nan`` for
    an empty vector or a vector with zero total mass.

    Raises:
    ------
    ValueError
    - if an entry is negative or non-finite.
    """
    p = np.asarray(p, dtype=float)

    if p.ndim != 1:
        p = p.ravel()

    if not np.isfinite(p).all(): 
        raise ValueError('entropy argument contains non-finite mass')

    if np.any(p < 0):
        raise ValueError("negative mass in entropy argument")
    
    total = p.sum()

    if total <= 0:
        return float("nan")
    
    p = p[p > 0] / total

    return float(-(p * np.log2(p)).sum())


@dataclass(frozen=True)
class EntropyResult:
    """
    The residual-entropy diagnostic for one corpus and one weighting.

    Returned by :func:`label_entropy`. The source contributions explain where the
    remaining label entropy comes from, which matters when one source supports most
    of the crossing in an otherwise nested design.

    Attributes:
    -----------
    H_G : float - marginal label entropy, in bits.
    H_G_given_D : float - source-weighted conditional label entropy, in bits.
    ratio : float - ``H_G_given_D / H_G``; ``nan`` when ``H_G`` is zero.
    uncertainty_coefficient : float - ``1 - ratio``, also written ``U(G|D)``.
    per_source : pd.DataFrame - source-indexed table of mass, source probability,
    positive-mass group count, spanning flag, conditional entropy, and contribution
    to ``H_G_given_D``. Sorted by decreasing contribution.
    n_sources : int - all sources, including sources with zero total weight.
    n_spanning_sources : int - positive-weight sources with more than one group
    of positive mass.
    n_nested_sources : int - positive-weight sources with only one such group.
    n_zero_weight_sources : int - sources with zero total mass.
    weighting, corpus : str - the weighting definition and corpus name.

    Notes:
    ------
    ``mutual_information`` is ``H_G - H_G_given_D`` in bits, and
    ``explained_fraction`` aliases ``uncertainty_coefficient``. The dataclass is
    frozen, but its contained DataFrame is still mutable.
    """

    H_G: float
    H_G_given_D: float
    ratio: float
    uncertainty_coefficient: float
    per_source: pd.DataFrame = field(repr=False)
    n_sources: int = 0

    n_spanning_sources : int = 0
    n_nested_sources : int = 0
    n_zero_weight_sources : int = 0
    weighting : str = "incidence rows"
    corpus : str = "corpus"

    @property
    def mutual_information(self) -> float:
        """
        ``I(G;D) = H(G) - H(G|D)``, the label information carried by source, in bits.
        """
        return self.H_G - self.H_G_given_D

    @property
    def explained_fraction(self) -> float:
        """
        The fraction of label entropy explained by source, identical to ``U(G|D)``.

        Returns ``nan`` when the marginal label entropy is zero.
        """
        return self.uncertainty_coefficient
    
    def summary(self) -> str:
        """
        One-line report of the entropies, ratio, source counts, and weighting.

        The source grain is whatever the corpus source column encodes; describe that
        grain explicitly when quoting the returned text.
        """
        return (
            f"{self.corpus}: "
            f"H(G) = {self.H_G:.4f} bits, "
            f"H(G|D) = {self.H_G_given_D:.4f} bits, "
            f"residual ratio = {self.ratio:.4f}, "
            f"provenance-explained fraction = "
            f"{self.uncertainty_coefficient:.4f}; "
            f"{self.n_spanning_sources} spanning, "
            f"{self.n_nested_sources} nested, "
            f"{self.n_zero_weight_sources} zero-weight "
            f"sources; weighting: {self.weighting}"
        )


    def as_dict(self) -> dict:
        """
        Scalar diagnostic fields and provenance labels as a dictionary.

        Includes mutual information and the source counts, but excludes the per-source
        table. Values may be ``nan``, which strict JSON does not allow.
        """
        return {
            "corpus": self.corpus,
            "weighting": self.weighting,
            "H_G": self.H_G,
            "H_G_given_D": self.H_G_given_D,
            "mutual_information": self.mutual_information,
            "ratio": self.ratio,
            "uncertainty_coefficient": self.uncertainty_coefficient,
            "n_sources": self.n_sources,
            "n_spanning_sources": self.n_spanning_sources,
            "n_nested_sources": self.n_nested_sources,
            "n_zero_weight_sources": self.n_zero_weight_sources, }


def label_entropy(corpus: Corpus) -> EntropyResult:
    """
    Measure how much group-label entropy survives conditioning on source.

    The weighted source x group table is the sufficient input. Each source contributes
    its conditional label entropy multiplied by its share of the total corpus mass:

        contribution[d] = p(d) * H(G|D=d)
        H(G|D) = sum_d contribution[d]

    Parameters:
    ----------
    corpus : Corpus - normalized observations with declared source, unit, group,
    and row weights. Both scalar and composite source keys are supported.

    Returns:
    -------
    EntropyResult - marginal and conditional entropy in bits, residual ratio R,
    uncertainty coefficient U, source counts, and the source contribution table.
    The table columns are ``weight``, ``source_probability``, ``n_groups_present``,
    ``spanning``, ``H_G_given_d``, and ``contribution``.

    Raises:
    ------
    ValueError
    - if the source table has negative or non-finite weights, or no positive mass.
    RuntimeError
    - if marginal entropy is non-finite or conditional entropy exceeds marginal
    entropy beyond numerical tolerance.

    Notes:
    ------
    Zero-mass sources have stored entropy and contribution 0 for bookkeeping;
    this does not define a conditional distribution for an unobserved source.
    ``ratio`` and ``uncertainty_coefficient`` are ``nan`` when ``H_G = 0``.
    Small floating-point excursions are clipped before the ratio is formed.

    The result describes the empirical incidence design under its declared weighting.
    It neither estimates a classifier's accuracy nor establishes that units are
    independent or that an observed biological effect is valid.
    """
    ct = (corpus.source_group_weight.astype(float))

    values = ct.to_numpy(dtype = float)

    if not np.isfinite(values).all():
        raise ValueError("source_group_weight contains non-finite groups")

    if np.any(values < 0):
        raise ValueError("source_group_weight contains negative weights")

    n_d = ct.sum(axis = 1)
    total = float(n_d.sum())

    if total <= 0:
        raise ValueError("corpus has zero total weight")

    # Iterate over the numerical rows positionally.  Tuple-valued composite source
    # keys are valid labels, but pandas interprets ``frame.loc[tuple_key]`` as a
    # multi-axis indexer rather than as one object-valued row label.
    H_d = pd.Series(
        [
            shannon(row) if source_weight > 0 else 0.0
            for row, source_weight in zip(values, n_d.to_numpy(dtype=float))
        ],
        index=ct.index,
        dtype=float,
    )

    source_probability = (n_d / total)

    contribution = (source_probability * H_d)

    groups_present = ((ct > 0).sum(axis = 1))

    live = (n_d > 0)

    spanning = (live & (groups_present > 1))

    nested = (live & (groups_present <= 1))

    zero_weight = (~live)

    per_source = pd.DataFrame(
        {
            "weight": n_d,
            "source_probability": source_probability,
            "n_groups_present": groups_present,
            "spanning": spanning,
            "H_G_given_d": H_d,
            "contribution":contribution,}
    ).sort_values("contribution", ascending=False, )

    H_cond = float(contribution.sum())
    H_marg = shannon(ct.sum(axis = 0).to_numpy(dtype = float))

    if not np.isfinite(H_marg):
        raise RuntimeError("marginal group entropy is non-finite")

    if H_marg <= 0:
        ratio = float("nan")
        uncertainty = float("nan")

    else:
        tol = (1e-12 * max(1.0,abs(H_marg),))

        if H_cond > H_marg + tol:
            raise RuntimeError(
                "conditional entropy exceeds marginal "
                f"entropy: H(G|D)={H_cond}, "
                f"H(G)={H_marg}. "
                "Check corpus weighting."
            )

        H_cond = min(max(H_cond, 0.0), H_marg)
        ratio = (H_cond / H_marg)
        uncertainty = (1.0 - ratio)

    return EntropyResult(
        H_G=H_marg,
        H_G_given_D=H_cond,
        ratio=ratio,
        uncertainty_coefficient=(uncertainty),
        per_source=per_source,
        n_sources=int(len(ct)),
        n_spanning_sources=int(spanning.sum()),
        n_nested_sources=int(nested.sum()),
        n_zero_weight_sources=int(zero_weight.sum()),
        weighting=corpus.weighting,
        corpus=corpus.name,
    )


def leave_one_source_out(corpus: Corpus) -> pd.DataFrame:
    """
    Recompute the entropy ratio after removing each source in turn.

    This asks whether the diagnostic depends on one provenance stratum. Removing a
    source changes both the marginal label distribution and the conditional entropy;
    the resulting ratio can therefore move in either direction.

    Parameters:
    ----------
    corpus : Corpus - the full design used for the baseline and source removals.

    Returns:
    -------
    pd.DataFrame - indexed by removed source and sorted by decreasing absolute
    change in R. Successful rows report ``ratio_without``, ``delta_ratio``,
    ``abs_delta_ratio``, ``pct_change``, ``H_G_without``,
    ``H_G_given_D_without``, and ``original_contribution``.
    An empty table is returned when no source can be removed from a one-source corpus.

    Notes:
    ------
    ``pct_change = 100 * delta_ratio / baseline_ratio`` is ``nan`` when the
    baseline is non-finite or within ``1e-12`` of zero. A remaining single-group
    design has undefined R. A ValueError during subset construction or scoring produces a legacy
    fallback row with undefined scores and a ``contribution`` field; successful
    rows use ``original_contribution`` instead.

    This is deletion sensitivity over whole sources. It is separate from structural
    leave-one-source-out, which counts units losing all provenance support.
    Errors computing the full-corpus baseline propagate to the caller.
    """
    base = label_entropy(corpus)
    rows = []
    for s in corpus.sources:
        keep = [x for x in corpus.sources if x != s]
        if not keep:
            continue
        try:
            sub = label_entropy(corpus.subset(sources=keep))
        except ValueError:
            rows.append({"source": s, "ratio_without": np.nan, "pct_change": np.nan,
                         "H_G_without": np.nan, "contribution": np.nan})
            continue
        delta = (sub.ratio - base.ratio if (np.isfinite(sub.ratio) and np.isfinite(base.ratio)) else float("nan"))

        if np.isfinite(base.ratio) and abs(base.ratio) > 1e-12 and np.isfinite(delta):
            pct_change = (100.0 * delta / base.ratio)
        else:
            pct_change = float("nan")
        
        rows.append(
            {
                "source": s,
                "ratio_without": sub.ratio,
                "delta_ratio": delta,
                "abs_delta_ratio": (abs(delta) if np.isfinite(delta) else float("nan")),
                "pct_change": pct_change,
                "H_G_without": sub.H_G,
                "H_G_given_D_without": sub.H_G_given_D,
                "original_contribution": float(base.per_source.at[s, "contribution"])
            })
    if not rows:
        return pd.DataFrame(
        columns=[
            "ratio_without",
            "delta_ratio",
            "abs_delta_ratio",
            "pct_change",
            "H_G_without",
            "H_G_given_D_without",
            "original_contribution",
        ]
    )

    out = (pd.DataFrame(rows).set_index("source").sort_values("abs_delta_ratio",ascending=False,))

    out.attrs.update(corpus=corpus.name,base_ratio=base.ratio,
            note=(
            "Leave-one-source-out values are influence "
            "diagnostics. Removing a source changes both "
            "conditional and marginal label distributions."), )

    return out


@dataclass(frozen=True)
class BootstrapResult:
    """
    Source-resampling distribution of the residual-entropy ratio.

    Attributes:
    -----------
    values : np.ndarray, shape (B,) - one R per source resample. Degenerate draws
    remain ``nan`` in their original positions.
    observed : float - R for the original corpus.
    lo, hi : float - 2.5th and 97.5th percentiles of finite draws; ``nan`` when none
    are finite. These are descriptive resampling limits.
    n_degenerate, n_effective : int - counts of undefined and finite draws.
    frac_exactly_zero : float - fraction of finite draws satisfying
    ``np.isclose(value, 0)`` with NumPy's default tolerance, despite the field name.
    note : str - qualification of the interval's interpretation.

    Notes:
    ------
    A narrow interval does not establish calibrated uncertainty when crossing or
    one group depends on a small number of sources. The source-resampling unit
    also need not coincide with the unit of independent biological replication.
    """

    values: np.ndarray = field(repr=False)
    observed: float = float("nan")
    lo: float = float("nan")
    hi: float = float("nan")
    n_degenerate: int = 0
    n_effective: int = 0
    frac_exactly_zero: float = float("nan")
    note: str = ""

    def as_dict(self) -> dict:
        """
        The observed ratio, percentile limits, draw counts, zero fraction, and note.

        Excludes the full ``values`` array. Undefined scalars remain ``nan``.
        """
        return {
            "observed": self.observed, "lo": self.lo, "hi": self.hi,
            "n_effective": self.n_effective, "n_degenerate": self.n_degenerate,
            "frac_exactly_zero": self.frac_exactly_zero, "note": self.note,
        }


def bootstrap_ratio(corpus: Corpus, B: int = 10_000, seed: int = 0) -> BootstrapResult:
    """
    Resample whole source weight vectors and recompute R.

    For a corpus with S sources, each replicate draws S source rows uniformly with
    replacement from the source x group table. A selected source retains its original
    group weights and total mass; sources are not reweighted to equal mass.

    Parameters:
    ----------
    corpus : Corpus - the design whose source table defines the resampling population.
    B : int, default 10000 - number of resamples. Zero returns an empty distribution.
    seed : int, default 0 - seed for the local NumPy random generator.

    Returns:
    -------
    BootstrapResult - original R, all resampled ratios, finite-draw percentile limits,
    degeneracy counts, and the fraction of finite ratios numerically close to zero.

    Notes:
    ------
    Draws with zero total mass or zero marginal label entropy receive ``nan`` and
    are excluded from percentile calculations. The limits summarize sensitivity
    to the observed sources; they are not automatically a valid confidence interval,
    especially when one source dominates a group. This procedure does not resample
    units, donors, or assay records independently.

    Invalid sample counts are rejected by NumPy; errors in the observed entropy
    calculation propagate from :func:`label_entropy`.
    """
    rng = np.random.default_rng(seed)
    ct = corpus.source_group_weight
    counts = ct.to_numpy(dtype=float)
    n_d = counts.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        H_d = np.array([shannon(row) for row in counts])
    H_d = np.nan_to_num(H_d, nan=0.0)

    K = len(n_d)
    out = np.full(B, np.nan)
    for b in range(B):
        idx = rng.integers(0, K, K)
        tot = n_d[idx].sum()
        if tot <= 0:
            continue
        H_marg = shannon(counts[idx].sum(axis=0))
        if not np.isfinite(H_marg) or H_marg <= 0:
            continue
        out[b] = ((n_d[idx] * H_d[idx]).sum() / tot) / H_marg

    good = out[np.isfinite(out)]
    n_deg = int(B - good.size)
    lo, hi = (np.percentile(good, [2.5, 97.5]) if good.size else (np.nan, np.nan))
    return BootstrapResult(
        values=out,
        observed=label_entropy(corpus).ratio,
        lo=float(lo),
        hi=float(hi),
        n_degenerate=n_deg,
        n_effective=int(good.size),
        frac_exactly_zero=float(np.isclose(good, 0.0).mean()) if good.size else float("nan"),
        note="not a confidence interval when one source dominates a group; see docstring",
    )
