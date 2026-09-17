"""
Sensitivity to the grain at which provenance is declared.

Does source mean a deposit, a laboratory, a study, or an instrument run? These
definitions partition the same observations differently and can produce different
residual-entropy ratios. This module makes that choice visible by scoring several
declared source keys and carrying their definitions beside the results.

Computation:
-----------
Every grain defines a new source column for :func:`label_entropy`. The unit and
group definitions remain those supplied by the caller. ``score_grains`` starts
from raw records and selects a shared complete-case row set; ``grain_sweep``
starts from an existing corpus and requires metadata aligned to its analytical rows.

Interpretation:
---------------
A single source gives ``R = 1`` when the group entropy is positive. A separate
source for every group-pure row gives ``R = 0`` under the same condition. Neither
extreme selects the scientifically appropriate provenance grain. Declare that
grain from how the resource was assembled, then report sensitivity to alternatives.

Composite keys remain tuples. A local run or sample identifier can therefore be
qualified by its study without relying on a separator embedded in a text label.
Repeated observations and explicit weights are handled by :class:`Corpus` at
each grain; they are part of the diagnostic's definition.

Reporting:
----------
Predeclare a primary grain and report a sweep across alternatives supported
by the metadata. The submission or deposit, where available, or the originating
publication provides the starting reporting convention. The caller must supply
the corresponding columns; the code does not infer this grain automatically.

Contents:
---------
score_grains - compare source definitions from one input table.
grain_sweep - compare aligned source metadata for an existing corpus.
declare_grain - format one scored grain as a reporting sentence.
"""
from __future__ import annotations

from typing import Mapping, Sequence, Union

import numpy as np
import pandas as pd

from .entropy import label_entropy
from .incidence import Corpus

_ColumnSpec = Union[str, Sequence[str]]

GRAIN_REPORTING_RULE = (
    "A residual-entropy ratio is not reproducibly interpretable "
    "without a declared provenance grain and weighting. The same "
    "corpus may appear strongly aliased under one defensible source "
    "definition and well crossed under another. Report a sensitivity "
    "sweep across the provenance columns supported by the metadata, "
    "alongside a predeclared primary grain."
)

SWEEP_COLUMNS: tuple[str, ...] = (
    "grain",
    "column",

    "n_sources",
    "n_spanning_sources",
    "n_nested_sources",
    "n_zero_weight_sources",

    "H_G",
    "H_G_given_D",
    "ratio",
    "U",

    "n_rows",
    "n_units",
    "n_groups",

    "degeneracy",
    "weighting",
    "corpus",
)

def _as_grain_map(
    grains: Mapping[str, _ColumnSpec] | Sequence[str],
) -> dict[str, _ColumnSpec]:
    """
    Resolve a named mapping or sequence of columns into the grain panel.

    Mapping keys become display strings; mapping values retain scalar or composite
    column specifications. A sequence uses each column name as its display label.
    An empty panel raises ValueError, and a bare string raises TypeError.
    """
    if isinstance(grains, str):
        raise TypeError(
            "grains must be a mapping {label: column} or a sequence of column names, "
            f"not the single string {grains!r} -- a sweep of one grain is not a sweep"
        )
    if isinstance(grains, Mapping):
        out = {str(k): v for k, v in grains.items()}
    else:
        out = {str(c): str(c) for c in grains}
    if not out:
        raise ValueError("no grains given; a sweep needs at least one provenance column")
    return out


def _spec_columns(spec: _ColumnSpec) -> list[str]:
    """
    List the physical columns used by one scalar or composite grain specification.
    """
    return [spec] if isinstance(spec, str) else [str(column) for column in spec]


def _reported_spec(spec: _ColumnSpec) -> str | tuple[str, ...]:
    """
    Keep a scalar column name or return a tuple for a composite reporting field.
    """
    return spec if isinstance(spec, str) else tuple(str(column) for column in spec)


def _degeneracy(corpus: Corpus) -> str:
    """
    Describe count-based extremes that can make a grain uninformative.

    Checks for fewer than two groups, one source, or one source per analytical row,
    in that order. An empty string means none of these checks fired; it does not
    exclude degeneracy caused by the weighting.
    """
    if corpus.n_groups < 2:
        return ("one biological group: H(G) = 0, so normalized residual entropy is undefined")

    if corpus.n_sources == 1:
        return ("one provenance source for the whole corpus: ratio = 1 by construction")

    if corpus.n_sources == corpus.n_rows:
        return ("source is unique to every row : ratio = 0 by construction")
    return ""


def _score_one(
    corpus: Corpus,
    grain: str,
    column: _ColumnSpec,
    corpus_name: str,
) -> dict[str, object]:
    """
    Combine one grain's entropy result with its column specification and design counts.

    The returned dictionary follows ``SWEEP_COLUMNS`` and includes the source mass
    categories, R, U, weighting, corpus name, and any count-based degeneracy note.
    """
    res = label_entropy(corpus)
    return {
        "grain":
            grain,
        "column":
            _reported_spec(column),
        "n_sources":
            res.n_sources,
        "n_spanning_sources":
            res.n_spanning_sources,
        "n_nested_sources":
            res.n_nested_sources,
        "n_zero_weight_sources":
            res.n_zero_weight_sources,
        "H_G":
          res.H_G,
        "H_G_given_D":
            res.H_G_given_D,
        "ratio":
            res.ratio,
        "U":
            res.uncertainty_coefficient,
        "n_rows":
            corpus.n_rows,
        "n_units":
            corpus.n_units,
        "n_groups":
            corpus.n_groups,
        "degeneracy":
            _degeneracy(corpus),
        "weighting":
            corpus.weighting,
        "corpus":
            corpus_name,}


def _finish(rows: list[dict[str, object]], *, ascending: bool, attrs: dict[str, object]) -> pd.DataFrame:
    """
    Assemble and stably sort a grain panel by R, with undefined values last.

    Resets the row index and attaches the supplied metadata and reporting rule to
    ``DataFrame.attrs``. Equal ratios retain their original grain order.
    """
    out = pd.DataFrame(rows, columns=list(SWEEP_COLUMNS))
    out = out.sort_values("ratio", ascending=ascending, na_position="last", kind="mergesort")
    out = out.reset_index(drop=True)
    out.attrs.update(attrs)
    out.attrs["reporting_rule"] = GRAIN_REPORTING_RULE
    return out

def score_grains(
    df: pd.DataFrame,
    grains: Mapping[str, _ColumnSpec] | Sequence[str],
    *,
    unit: _ColumnSpec,
    group: _ColumnSpec,
    weight: str | None = None,
    name: str = "corpus",
    dropna: bool = True,
    ascending: bool = True,
    duplicates: str = "warn",
    hierarchy: Mapping[str, _ColumnSpec] | None = None,
) -> pd.DataFrame:
    """
    Score alternative provenance grains from a shared input table.

    The comparison begins with the same eligible observations for every grain. Rows
    missing a value in any required unit, group, source-grain, or weight column are
    removed together when ``dropna=True``. Each resulting source definition is then
    constructed and validated through :meth:`Corpus.from_frame`.

    Parameters:
    ----------
    df : pd.DataFrame - input records, including every requested grain column.
    grains : mapping or sequence - display name to column specification, or a
    sequence of column names. A specification can be one column or several columns
    forming a composite source key.
    unit, group : str or sequence of str, keyword-only - columns defining analysis
    units and group labels. Each normalized unit must carry only one group.
    weight : str or None, default None, keyword-only - row-mass column. None gives
    every retained analytical row mass 1.
    name : str, default "corpus", keyword-only - label carried into the results.
    dropna : bool, default True, keyword-only - remove shared incomplete rows;
    False raises when the required columns contain missing values.
    ascending : bool, default True, keyword-only - sort from low to high R.
    duplicates : str, default "warn", keyword-only - repeated-key policy passed
    to Corpus: ``"warn"``, ``"keep"``, ``"drop"``, or ``"raise"``.
    hierarchy : mapping or None, default None, keyword-only - explicitly declared
    parent-key columns passed to Corpus for hierarchy validation.

    Returns:
    -------
    pd.DataFrame - one row per grain, with its column specification, source and
    unit counts, positive-mass spanning/nested/zero-weight source counts, entropies
    in bits, R, U, degeneracy note, weighting, and corpus name. Attributes record
    input, scored, and dropped row counts and the shared-row selection rule.

    Raises:
    ------
    KeyError
    - if a required column is absent.
    ValueError
    - if no rows remain, missing values are disallowed, or Corpus validation fails.
    TypeError
    - if ``grains`` is supplied as a bare string.

    Notes:
    ------
    The shared filter uses pandas missing-value detection. Further normalization
    can reject identifiers such as blank strings. Hierarchy metadata follows the
    Corpus missing-parent rules rather than this complete-case filter.

    The shared row set is defined before duplicate handling. With ``duplicates="drop"``,
    different grains can collapse different numbers of observations. Compare retained
    row counts and weighting as well as R before interpreting the sweep.
    """
    gmap = _as_grain_map(grains)

    specs = [unit, group, *gmap.values()]
    required = [column for spec in specs for column in _spec_columns(spec)]

    if weight is not None:
        required.append(weight)

    required = list(dict.fromkeys(required))

    missing_columns = [c for c in required if c not in df.columns]

    if missing_columns:
        raise KeyError(
            f"required columns not found: {missing_columns}; "
            f"have {list(df.columns)}"
        )

    if dropna:
        work = (df.dropna(subset=required).copy())

    else:
        bad = df[required].isna().any(axis=1)

        if bad.any():
            raise ValueError(
                f"{int(bad.sum())} rows contain missing values "
                "in unit/group/weight/provenance columns. "
                "Either clean them first or use dropna=True."
            )

        work = df.copy()

    if work.empty:
        raise ValueError(
            "no rows remain after applying the shared "
            "complete-case filter"
        )

    rows: list[dict[str, object]] = []

    for label, column in gmap.items():

        corpus = Corpus.from_frame(
            work,
            source=column,
            unit=unit,
            group=group,
            weight=weight,
            name=f"{name}@{label}",

            dropna=False,
            duplicates=duplicates,
            hierarchy=hierarchy,
        )

        rows.append(_score_one(corpus,label,column,name,))

    return _finish(
        rows,
        ascending=ascending,
        attrs={
            "corpus": name,
            "unit": unit,
            "group": group,
            "weight": weight,
            "duplicates": duplicates,
            "hierarchy": None if hierarchy is None else dict(hierarchy),
            "n_input_rows": int(len(df)),
            "n_scored_rows": int(len(work)),
            "n_dropped_rows": int(len(df) - len(work)),
            "shared_row_universe": True,
        },
    )


def grain_sweep(
    corpus: Corpus,
    df: pd.DataFrame,
    columns: Mapping[str, _ColumnSpec] | Sequence[str],
    *,
    ascending: bool = True,
) -> pd.DataFrame:
    """
    Rescore an existing corpus using alternative, positionally aligned source keys.

    Each grain is applied through :meth:`Corpus.with_source`, which preserves the
    corpus lineage and reruns collision checks under its existing duplicate policy.

    Parameters:
    ----------
    corpus : Corpus - existing analytical projection and provenance records.
    df : pd.DataFrame - source metadata with exactly ``corpus.n_rows`` rows, in
    the same order as ``corpus.frame``. Raw-record order is insufficient if rows
    were excluded or collapsed during construction.
    columns : mapping or sequence - named grain specifications, or column names
    used as both the specification and display label.
    ascending : bool, default True, keyword-only - sort by increasing R; undefined
    ratios are always placed last.

    Returns:
    -------
    pd.DataFrame - the grain panel with the same diagnostic columns as
    :func:`score_grains`. Attributes retain corpus, weighting, duplicate policy,
    hierarchy, and the reporting rule.

    Raises:
    ------
    ValueError
    - if row counts differ, a new key is missing, or reprojection is invalid.
    KeyError
    - if a requested grain column is absent.

    Notes:
    ------
    The length check cannot establish correct row order. The caller must align
    metadata to the analytical projection before calling this function. Use
    :func:`score_grains` when starting directly from a raw input table.
    """
    gmap = _as_grain_map(columns)
    if len(df) != corpus.n_rows:
        raise ValueError(
            f"frame has {len(df)} rows but corpus {corpus.name!r} has {corpus.n_rows}; "
            "grain_sweep needs a row-aligned frame. A corpus built with dropna=True is "
            "shorter than its source frame -- rebuild with dropna=False, pass the cleaned "
            "frame, or use score_grains()."
        )
    requested = [column for spec in gmap.values() for column in _spec_columns(spec)]
    missing = [column for column in requested if column not in df.columns]
    if missing:
        raise KeyError(f"grain columns not found in frame: {missing}; have {list(df.columns)}")

    rows: list[dict[str, object]] = []
    for label, column in gmap.items():
        rows.append(
            _score_one(
                corpus.with_source(df, column, name=f"{corpus.name}@{label}"),
                label,
                column,
                corpus.name,
            )
        )

    return _finish(
        rows,
        ascending=ascending,
        attrs={
            "corpus": corpus.name,
            "weighting": corpus.weighting,
            "duplicates": getattr(corpus, "duplicate_policy", "unknown"),
            "hierarchy": getattr(corpus, "hierarchy", None),
        },
    )


def declare_grain(result_row: pd.Series | Mapping[str, object] | pd.DataFrame) -> str:
    """
    Format one grain result with the information needed to quote its ratio.

    Parameters:
    ----------
    result_row : pd.Series, mapping, or one-row pd.DataFrame - a result from
    :func:`score_grains` or :func:`grain_sweep`.

    Returns:
    -------
    str - grain, source counts, entropies, R, U, and weighting, followed by the
    degeneracy note when present. Missing fields use placeholders; non-finite
    numbers are displayed as ``NaN``.

    Raises:
    ------
    TypeError
    - if the supplied DataFrame does not contain exactly one row.

    Notes:
    ------
    This is a reporting helper. It does not validate whether the declared grain
    corresponds to the resource's actual provenance or independent sampling unit.
    """
    if isinstance(result_row, pd.DataFrame):
        if len(result_row) != 1:
            raise TypeError(
                f"declare_grain takes one row, got a DataFrame with {len(result_row)} rows; "
                "use df.apply(declare_grain, axis=1) or df.iloc[i]"
            )
        result_row = result_row.iloc[0]

    row: Mapping[str, object] = (
        result_row.to_dict() if isinstance(result_row, pd.Series) else dict(result_row)
    )

    def num(key: str, nd: int = 4) -> str:
        """
        Format a numeric diagnostic, retaining an explicit placeholder when unavailable.
        """
        v = row.get(key, None)
        if v is None:
            return "?"
        try:
            f = float(v)
        except (TypeError, ValueError):
            return str(v)
        return "NaN" if not np.isfinite(f) else f"{f:.{nd}f}"

    def cnt(key: str) -> str:
        """
        Format an integer count, or a placeholder when the field is absent or undefined.
        """
        v = row.get(key, None)
        if v is None:
            return "?"
        try:
            return str(int(v))
        except (TypeError, ValueError):
            return str(v)

    corpus = row.get("corpus", "corpus")
    grain = row.get("grain", "UNDECLARED")
    column = row.get("column", None)
    weighting = row.get("weighting", "UNDECLARED")

    head = f"corpus {corpus!r} at grain {grain!r}"
    if column is not None and str(column) != str(grain):
        head += f" (column {str(column)!r})"

    line = (
        f"{head}, weighting: {weighting} -- "
        f"{cnt('n_sources')} sources over {cnt('n_rows')} rows, "
        f"{cnt('n_units')} units, {cnt('n_groups')} groups; "
        f"{cnt('n_spanning_sources')} sources span biological groups, "
        f"{cnt('n_nested_sources')} are nested within one group, and "
        f"{cnt('n_zero_weight_sources')} carry zero weight; "
        f"{cnt('n_nested_sources')} of {cnt('n_sources')} contribute exactly zero; "
        f"H(G) = {num('H_G')} bits, H(G|D) = {num('H_G_given_D')} bits, "
        f"ratio = {num('ratio')}, U(G|D) = {num('U')}."
    )

    note = str(row.get("degeneracy", "") or "").strip()
    if note:
        line += f" DEGENERATE GRAIN: {note}."
    return line
