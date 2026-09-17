"""The shape most published corpora arrive in: an aggregated supplementary table.

One row per (source, unit) incidence, plus a group column. That is enough to build a
:class:`~provenance.incidence.Corpus` and score it -- but only if the table really does
carry those three fields at a usable grain, and most published tables do not.

So this module has two jobs, and the second one is a *result*, not a utility:

``from_incidence_table``
    Build a corpus from a table that is already in incidence shape.

``check_recoverability``
    Audit whether a table is sufficient to build a corpus at all, and say what is
    missing. Run this on a corpus you did not assemble yourself before you quote any
    number from it. The observation that most published supplements fail this audit is
    the argument for a provenance reporting standard; it is not an aside.

The recurring failure is a column that *looks* like data mass and is not. In the worked
example's Table S1 (84 incidence rows), the column named "Peptides extracted" held
methods prose: 0 of 84 cells parsed directly as a number, 77 contained some number of
three or more digits, 74 contained more than one such number, and naively extracting the
largest match per row gave "counts" spanning 100 to 186,002,321. A weighting built on
that column would have been fiction. :func:`looks_like_prose` exists to catch it.
"""

from __future__ import annotations

import re
from typing import Mapping, Sequence, Union

import numpy as np
import pandas as pd

from ..incidence import Corpus

__all__ = [
    "from_incidence_table",
    "check_recoverability",
    "looks_like_prose",
    "numeric_audit",
]

_ColumnSpec = Union[str, Sequence[str]]

# --------------------------------------------------------------------------- helpers


def _clean(s: pd.Series) -> pd.Series:
    """Match Corpus.from_frame's key normalisation so audits agree with construction."""
    out = s.astype("string").str.strip()
    return out.mask(out.isna() | out.str.lower().isin(["", "nan", "none", "<na>"]))


def _spec_columns(spec: _ColumnSpec) -> list[str]:
    """Return the physical columns named by a scalar or composite key specification."""
    return [spec] if isinstance(spec, str) else [str(column) for column in spec]


def _clean_spec(df: pd.DataFrame, spec: _ColumnSpec) -> pd.Series:
    """Normalize a scalar/composite audit key without joining its components."""
    columns = _spec_columns(spec)
    components = [_clean(df[column]).reset_index(drop=True) for column in columns]
    if len(components) == 1:
        return components[0]
    values: list[object] = []
    for row in zip(*(component.tolist() for component in components)):
        if any(pd.isna(value) for value in row):
            values.append(pd.NA)
        else:
            values.append(tuple(str(value) for value in row))
    return pd.Series(values, dtype=object)


def _unsafe_automatic_weight_name(column: object) -> bool:
    """True when a numeric column's name describes non-row-level information.

    Weight suggestions are deliberately conservative.  A dataset/study total repeated
    over sample rows is not sample-level mass, and a replicate token is an identifier or
    design descriptor rather than independent biological evidence.  Callers may still
    pass a genuinely row-level numeric column explicitly as ``weight=``; this helper only
    prevents unsafe automatic suggestions.
    """
    tokens = set(re.findall(r"[a-z0-9]+", str(column).lower()))
    replicate = bool(tokens & {"replicate", "replicates", "rep", "technicalreplicate"})
    aggregate = bool(tokens & {"total", "overall", "aggregate", "aggregated"})
    dataset_scope = bool(tokens & {"dataset", "study", "project", "deposit", "cohort", "atlas"})
    count_like = bool(tokens & {"count", "counts", "number", "num", "n", "runs", "samples"})
    return replicate or aggregate or (dataset_scope and count_like)


def _number_pattern(min_digits: int) -> re.Pattern[str]:
    if min_digits < 1:
        raise ValueError("min_digits must be >= 1")
    return re.compile(r"\d[\d,]{%d,}" % (min_digits - 1))


def _ascii(text: str, width: int = 80) -> str:
    """Console-safe, truncated preview.

    Free-text supplement cells routinely contain micro signs and dashes; printing them
    raw raises UnicodeEncodeError on a cp1252 Windows console, which turns an audit into
    a traceback. Previews are transliterated, never the stored values.
    """
    return text[:width].encode("ascii", "replace").decode("ascii")


def numeric_audit(series: pd.Series, *, min_digits: int = 3) -> pd.Series:
    """Count how a column *would* be misread as data mass, without misreading it.

    Returns the five numbers worth printing beside any claimed weighting:

    ``n``, ``n_missing``
        size of the column.
    ``n_direct_numeric``, ``frac_direct_numeric``
        cells that parse as a bare number with :func:`pandas.to_numeric`. This is the
        only parse that is safe to use as a weight.
    ``n_with_number``, ``n_multi_number``, ``frac_multi_number``
        cells containing at least one, and more than one, run of ``min_digits`` or more
        digits. More than one means any single extracted "count" is a choice among
        several, i.e. an invention.
    ``extracted_min``, ``extracted_max``
        range of the largest match per cell, the value a naive regex extractor would
        have handed you. NaN when nothing matched -- not 0, because nothing was found
        rather than zero found.
    ``example``
        an ASCII-transliterated preview of the first non-missing cell.

    Anchor (worked-example corpus, Table S1, 84 rows, the column "Peptides extracted"):
    ``n_direct_numeric`` 0, ``n_with_number`` 77, ``n_multi_number`` 74,
    ``extracted_min`` 100, ``extracted_max`` 186,002,321.
    """
    s = pd.Series(series)
    n = int(len(s))
    text = s.astype(str)
    missing = s.isna() | text.str.strip().isin(["", "nan", "None", "NaN", "<NA>"])
    n_missing = int(missing.sum())

    direct = pd.to_numeric(s, errors="coerce")
    n_direct = int(direct.notna().sum())

    pat = _number_pattern(min_digits)
    matches = text.where(~missing, "").map(lambda t: pat.findall(t))
    counts = matches.map(len)

    def _largest(found: Sequence[str]) -> float:
        vals: list[int] = []
        for m in found:
            digits = m.replace(",", "")
            if digits:
                try:
                    vals.append(int(digits))
                except ValueError:  # pragma: no cover - regex guarantees digits
                    continue
        return float(max(vals)) if vals else float("nan")

    any_num = text.where(~missing, "").map(lambda t: len(re.findall(r"\d[\d,]*", t)))
    extracted = matches.map(_largest)
    n_present = n - n_missing
    example = ""
    if n_present:
        example = _ascii(text.loc[~missing].iloc[0])

    return pd.Series(
        {
            "n": n,
            "n_missing": n_missing,
            "n_direct_numeric": n_direct,
            "frac_direct_numeric": (n_direct / n_present) if n_present else float("nan"),
            "n_with_number": int((counts > 0).sum()),
            "n_multi_number": int((any_num > 1).sum()),
            "frac_multi_number": (
                float((any_num > 1).sum() / n_present) if n_present else float("nan")
            ),
            "extracted_min": float(np.nanmin(extracted)) if extracted.notna().any() else float("nan"),
            "extracted_max": float(np.nanmax(extracted)) if extracted.notna().any() else float("nan"),
            "example": example,
        },
        name=getattr(series, "name", None),
    )


def looks_like_prose(
    series: pd.Series,
    *,
    max_direct_numeric_frac: float = 0.5,
    min_multi_number_frac: float = 0.2,
    min_digits: int = 3,
) -> bool:
    """True when a column reads like counts but is free text.

    The rule, stated so it can be argued with: a column is prose when **few or no cells
    parse as a bare number** (``frac_direct_numeric <= max_direct_numeric_frac``) **and
    many cells contain several numbers** (``frac_multi_number >= min_multi_number_frac``).
    The second clause is what distinguishes methods prose from a column of clean counts
    stored as text -- clean counts have exactly one number per cell.

    This is deliberately narrow. A column of pure text with no digits at all returns
    False: nothing about it invites being read as data mass. An all-missing column also
    returns False, for the same reason; use :func:`numeric_audit` when you need the
    evidence rather than the verdict.

    Anchor (worked-example corpus, Table S1, 84 rows, "Peptides extracted"): 0 of 84
    directly numeric and 74 of 84 with more than one multi-digit number, so this returns
    True and the column must not be used as a weight. Under incidence weighting instead,
    that corpus scores ratio 0.0832 -- a number whose weighting has to be quoted with it.
    """
    audit = numeric_audit(series, min_digits=min_digits)
    frac_direct = audit["frac_direct_numeric"]
    frac_multi = audit["frac_multi_number"]
    if not np.isfinite(frac_direct) or not np.isfinite(frac_multi):
        return False
    return bool(frac_direct <= max_direct_numeric_frac and frac_multi >= min_multi_number_frac)


# ----------------------------------------------------------------------- construction


def from_incidence_table(
    df: pd.DataFrame,
    *,
    source: _ColumnSpec,
    unit: _ColumnSpec,
    group: _ColumnSpec,
    weight: str | None = None,
    name: str = "supplement",
    duplicates: str = "raise",
    dropna: bool = True,
    hierarchy: Mapping[str, _ColumnSpec] | None = None,
) -> Corpus:
    """Build a corpus from a published table in (source, unit, group) incidence shape.

    Parameters
    ----------
    source, unit, group
        Column names. ``source`` fixes the **grain** -- deposit, laboratory, instrument
        or acquisition batch are all defensible and they do not agree; see ``grain.py``.
        Whatever you choose has to be declared beside every number you report.
    weight
        Optional column of per-row data mass. Most published supplements have no usable
        one, which means incidence weighting, which means every result is a statement
        about *which* rows exist rather than *how much* data each carries. That is a
        limitation to declare, not a default to leave implicit. Run
        :func:`check_recoverability` first: a column that looks like counts is often
        prose (see :func:`looks_like_prose`).
    duplicates
        What to do about repeated (source, unit) pairs, which violate the declared shape.
        ``"raise"`` (default) reports them, ``"drop"`` keeps the first of each, ``"keep"``
        leaves them after an explicit choice, and ``"warn"`` retains them while emitting
        a warning. Enforcement lives in :meth:`Corpus.from_frame`, so this adapter and
        every other construction path use the same normalized keys. This is not cosmetic:
        H(G|D) weights each source by its total row mass, so a duplicated row silently
        up-weights its source.
    hierarchy
        Optional named parent-unit columns, forwarded to :meth:`Corpus.from_frame` and
        preserved as metadata. Missing hierarchy values remain missing; this adapter does
        not infer a donor from a sample name or a sample count from a dataset total.

    Notes
    -----
    Anchor (worked-example corpus, incidence weighting): the published table is 84
    incidence rows over 43 sources, 44 units and 2 groups, and yields ratio
    0.08319240360050288 with 40 of 43 sources contributing exactly zero.
    """
    if duplicates not in ("raise", "drop", "keep", "warn"):
        raise ValueError("duplicates must be one of 'raise', 'drop', 'keep', 'warn'")

    required = [column for spec in (source, unit, group) for column in _spec_columns(spec)]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError(
            f"columns not found: {missing}; have {list(df.columns)}. "
            "Run check_recoverability() to see what this table can and cannot support."
        )

    if weight is not None and weight in df.columns and looks_like_prose(df[weight]):
        audit = numeric_audit(df[weight])
        raise ValueError(
            f"weight column {weight!r} looks like free text, not counts: "
            f"{int(audit['n_direct_numeric'])} of {int(audit['n'])} cells parse as a bare "
            f"number and {int(audit['n_multi_number'])} contain more than one number. "
            "Use incidence weighting (weight=None) and say so, or supply real counts."
        )

    return Corpus.from_frame(
        df,
        source=source,
        unit=unit,
        group=group,
        weight=weight,
        name=name,
        dropna=dropna,
        duplicates=duplicates,
        hierarchy=hierarchy,
    )


# ---------------------------------------------------------------------------- audit


def check_recoverability(
    df: pd.DataFrame,
    *,
    source: _ColumnSpec,
    unit: _ColumnSpec,
    group: _ColumnSpec,
    weight: str | None = None,
    min_digits: int = 3,
) -> pd.DataFrame:
    """Can this published table be scored at all, and if not, what is missing?

    Returns one row per check, indexed by check name, with columns ``status``, ``value``
    and ``detail``. Statuses are:

    ``ok``
        the check passed.
    ``warn``
        the corpus builds, but a reported number will need a caveat.
    ``blocked``
        the corpus does not build, or builds into something with no contrast to score.
    ``unknown``
        the check could not run because a required column is absent.

    Nothing here raises: the point is to characterise tables that *cannot* be turned
    into a corpus, and an audit that dies on the first missing column cannot do that.
    Read ``corpus_constructible`` first, then the blocked rows.

    Two results deserve to be read as findings rather than diagnostics:

    ``sources_spanning_groups = 0``
        every source sits in exactly one group. The contrast is fully aliased with
        provenance: H(G|D) is 0 by construction, no correction method recovers anything,
        and no accuracy above chance is attainable. Anchor (second corpus in the worked
        example, incidence weighting): 28 studies, 2 groups, no study contributing to
        both, ratio exactly 0.0000.
    ``prose_columns``
        columns that a reader would take for counts and that are free text. This is the
        usual reason a published corpus cannot be weighted by data mass.

    Numeric dataset/study totals and technical-replicate fields are not offered as weight
    candidates: they describe a different level of the hierarchy, not mass measured for
    each row. They remain untouched metadata, and missing sample counts are never inferred
    from them.

    A note on grain, because it decides the answer: ``source`` is a choice, not a fact.
    The same table can score near 0 at one grain and near 1 at another, so this audit
    describes the table *at the grain you passed* and nothing more.
    """
    rows: list[dict[str, object]] = []

    def add(check: str, status: str, value: object, detail: str = "") -> None:
        rows.append({"check": check, "status": status, "value": value, "detail": detail})

    role_specs = {"source": source, "unit": unit, "group": group}
    absent = {
        role: [column for column in _spec_columns(spec) if column not in df.columns]
        for role, spec in role_specs.items()
    }
    absent = {role: columns for role, columns in absent.items() if columns}
    present = {role: spec for role, spec in role_specs.items() if role not in absent}

    add(
        "required_columns_present",
        "ok" if not absent else "blocked",
        f"{len(present)} of 3",
        "" if not absent else
        f"missing {sorted({column for columns in absent.values() for column in columns})}; "
        f"available {list(df.columns)[:12]}",
    )
    add("rows", "ok" if len(df) else "blocked", int(len(df)),
        "" if len(df) else "empty table")

    cleaned = {role: _clean_spec(df, spec) for role, spec in present.items()}

    for role in ("source", "unit", "group"):
        if role not in cleaned:
            add(f"{role}_missing_cells", "unknown", float("nan"),
                f"column(s) {absent[role]!r} not in table")
            continue
        n_missing = int(cleaned[role].isna().sum())
        add(
            f"{role}_missing_cells",
            "ok" if n_missing == 0 else ("blocked" if n_missing == len(df) else "warn"),
            n_missing,
            "" if n_missing == 0 else f"{n_missing} of {len(df)} rows would be dropped",
        )

    complete = None
    if len(cleaned) == 3:
        complete = pd.DataFrame(cleaned).dropna()

    for role in ("source", "unit", "group"):
        if complete is None or role not in complete:
            add(f"n_{role}s", "unknown", float("nan"), "required column absent")
        else:
            add(f"n_{role}s", "ok", int(complete[role].nunique()), f"at grain {present[role]!r}"
                if role == "source" else "")

    if complete is None or complete.empty:
        add("n_groups_at_least_2", "unknown", float("nan"), "cannot count groups")
        add("unit_group_unique", "unknown", float("nan"), "cannot check")
        add("duplicate_source_unit_rows", "unknown", float("nan"), "cannot check")
        add("sources_spanning_groups", "unknown", float("nan"), "cannot check")
        add("units_with_one_source", "unknown", float("nan"), "cannot check")
    else:
        n_groups = int(complete["group"].nunique())
        add(
            "n_groups_at_least_2",
            "ok" if n_groups >= 2 else "blocked",
            n_groups,
            "" if n_groups >= 2 else "one group means there is no contrast to score",
        )

        spans = complete.groupby("unit")["group"].nunique()
        n_span_units = int((spans > 1).sum())
        add(
            "unit_group_unique",
            "ok" if n_span_units == 0 else "blocked",
            n_span_units,
            "" if n_span_units == 0 else
            f"{n_span_units} units carry more than one group, e.g. "
            f"{sorted(spans[spans > 1].index, key=repr)[:3]}; "
            "unit-level statistics are undefined",
        )

        dup = int(complete.duplicated(subset=["source", "unit"]).sum())
        add(
            "duplicate_source_unit_rows",
            "ok" if dup == 0 else "warn",
            dup,
            "" if dup == 0 else
            "repeats up-weight their source in H(G|D); decide via duplicates= in "
            "from_incidence_table",
        )

        per_source = complete.groupby("source")["group"].nunique()
        n_spanning = int((per_source > 1).sum())
        add(
            "sources_spanning_groups",
            "ok" if n_spanning > 0 else "warn",
            n_spanning,
            "" if n_spanning > 0 else
            "fully aliased: no source contributes to more than one group, so H(G|D) = 0 "
            "structurally and no correction can recover the contrast",
        )

        spu = complete.drop_duplicates(["source", "unit"]).groupby("unit")["source"].nunique()
        n_single = int((spu == 1).sum())
        add(
            "units_with_one_source",
            "ok" if n_single < len(spu) else "warn",
            n_single,
            f"of {len(spu)} units"
            + ("" if n_single < len(spu) else "; every unit rests on a single source"),
        )

    # ------------------------------------------------------------------ weighting
    reserved = {
        column for spec in present.values() for column in _spec_columns(spec)
    }
    numeric_cols: list[str] = []
    prose_cols: list[str] = []
    for col in df.columns:
        if col in reserved:
            continue
        if _unsafe_automatic_weight_name(col):
            continue
        audit = numeric_audit(df[col], min_digits=min_digits)
        frac = audit["frac_direct_numeric"]
        if np.isfinite(frac) and frac >= 0.99:
            numeric_cols.append(str(col))
        elif looks_like_prose(df[col], min_digits=min_digits):
            prose_cols.append(str(col))

    add(
        "numeric_weight_candidates",
        "ok" if numeric_cols else "warn",
        len(numeric_cols),
        ", ".join(numeric_cols[:8]) if numeric_cols else
        "no fully numeric column: only incidence weighting is available, and that must "
        "be declared with every number",
    )
    add(
        "prose_columns",
        "ok" if not prose_cols else "warn",
        len(prose_cols),
        "" if not prose_cols else
        "read like counts, are free text: " + ", ".join(prose_cols[:8]),
    )

    if weight is None:
        add("weight_column_usable", "unknown", float("nan"),
            "no weight column requested; incidence weighting assumed")
    elif weight not in df.columns:
        add("weight_column_usable", "blocked", False,
            f"weight column {weight!r} not in table")
    else:
        audit = numeric_audit(df[weight], min_digits=min_digits)
        prose = looks_like_prose(df[weight], min_digits=min_digits)
        usable = bool(np.isfinite(audit["frac_direct_numeric"])
                      and audit["frac_direct_numeric"] >= 0.99 and not prose)
        if usable:
            detail = ""
        else:
            detail = (
                f"{int(audit['n_direct_numeric'])} of {int(audit['n'])} cells parse as a "
                f"bare number, {int(audit['n_multi_number'])} contain more than one number"
            )
            if np.isfinite(audit["extracted_max"]):
                detail += (
                    f"; naive extraction would span {audit['extracted_min']:,.0f} to "
                    f"{audit['extracted_max']:,.0f}"
                )
        add("weight_column_usable", "ok" if usable else "blocked", usable, detail)

    out = pd.DataFrame(rows).set_index("check")
    blocked = out.index[out["status"] == "blocked"].tolist()
    detail = "" if not blocked else "blocked by: " + ", ".join(blocked)
    if blocked == ["weight_column_usable"]:
        # Distinguish "this table cannot be scored" from "this table cannot be scored
        # the way you asked". The second is a weighting result, not a dead end.
        detail += (
            "; the corpus does build under incidence weighting (weight=None), and that "
            "weighting then has to be declared with every number"
        )
    verdict = pd.DataFrame(
        [{
            "status": "ok" if not blocked else "blocked",
            "value": not blocked,
            "detail": detail,
        }],
        index=pd.Index(["corpus_constructible"], name="check"),
    )
    return pd.concat([verdict, out])
