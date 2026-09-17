"""Read a tidy metadata table off disk and hand back a :class:`Corpus`.

This is the general-purpose adapter: the caller names the source, unit and group
columns and gets a corpus. Everything here is about surviving real files -- separators
that are not what the extension claims, byte-order marks, identifiers that pandas would
helpfully convert to numbers.

Three habits are load-bearing and are the reason this module exists rather than a bare
``pd.read_csv``:

**Files named ``.csv`` are frequently tab separated.** That has cost this project time
twice. When ``sep`` is left as None the reader sniffs, and if the result is a single
column it retries with a tab and then with the python engine's own sniffing, so a
mislabelled file loads instead of collapsing into one column with a header like
``"source\\tunit\\tgroup"``. The tab is always built with ``chr(9)``; a literal escape
sequence in pasted code has been mangled here before.

**Identifiers are read as text.** Deposit accessions, donor numbers and run indices are
identifiers, not quantities. Left to itself pandas turns ``007`` into ``7`` and a long
digit string into a float, which silently merges or splits sources and changes every
number this package reports. ``dtype_str=True`` is the default for that reason.

**Column guessing is never applied.** :func:`infer_columns` returns suggestions. Which
column is the *source* is a grain decision -- the same table scores near 0 at one grain
and near 1 at another -- so it belongs to the analyst, not to a heuristic.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any, Mapping, Sequence, Union

import numpy as np
import pandas as pd

from ..incidence import Corpus
from .aggregated_supplement import _unsafe_automatic_weight_name, looks_like_prose

__all__ = [
    "TAB",
    "sniff_separator",
    "read_table",
    "from_csv",
    "from_excel",
    "from_long",
    "column_profile",
    "infer_columns",
]

_ColumnSpec = Union[str, Sequence[str]]

#: Built with chr(9) on purpose: a literal escape in pasted code has been mangled here.
TAB = chr(9)

#: Tab first, so that ties break towards the mislabelled-.csv case.
_CANDIDATE_SEPS: tuple[str, ...] = (TAB, ",", ";", "|")

#: Byte-order mark. Built with chr() for the same reason TAB is: an invisible character
#: sitting in source code is a trap, and this one has survived a copy-paste before.
_BOM = chr(0xFEFF)

# Accession-shaped tokens, in the two shapes public repositories actually use:
# a letter prefix followed by digits, and a hyphenated multi-part identifier.
_ACCESSION_RE = re.compile(r"^[A-Za-z]{1,8}[-_]?\d{3,}(?:\.\d+)?$")
_HYPHEN_ACCESSION_RE = re.compile(r"^[A-Za-z]{1,8}(?:-[A-Za-z0-9]{1,8})+-\d{2,}$")

# Name hints. Substring matches on the column name only, deliberately domain-neutral:
# these are prompts for a human, never a decision.
_ROLE_HINTS: dict[str, tuple[str, ...]] = {
    "source": (
        "source", "accession", "deposit", "dataset", "study", "project", "submission",
        "batch", "lab", "laboratory", "instrument", "run", "experiment", "series",
        "site", "centre", "center", "platform", "pipeline", "repository", "cohort_id",
    ),
    "unit": (
        "unit", "sample", "specimen", "subject", "donor", "entity", "item", "record",
        "observation", "case", "patient", "individual",
    ),
    "group": (
        "group", "class", "label", "condition", "arm", "status", "phenotype",
        "category", "cohort", "treatment", "outcome", "disease", "state",
    ),
    "weight": (
        "weight", "count", "n_", "num", "size", "depth", "mass", "quantity",
        "frequency", "reads",
    ),
}


# ------------------------------------------------------------------------- reading


def sniff_separator(
    path: str | Path,
    *,
    encoding: str = "utf-8-sig",
    n_chars: int = 65_536,
) -> str | None:
    """Guess a delimiter from the head of a file, or None when nothing is convincing.

    Tries :class:`csv.Sniffer` first, then falls back to the candidate whose count is
    non-zero and *constant* across the first few non-empty lines -- constancy is what
    distinguishes a real delimiter from a comma that happens to sit inside a title.
    Returns None rather than a default, so the caller can decide what None means;
    :func:`read_table` treats it as "try a comma, then escalate".
    """
    p = Path(path)
    with open(p, "r", encoding=encoding, errors="replace", newline="") as fh:
        sample = fh.read(n_chars)
    if not sample.strip():
        return None
    cut = sample.rfind("\n")
    if cut > 0:
        sample = sample[:cut]

    try:
        dialect = csv.Sniffer().sniff(sample, delimiters="".join(_CANDIDATE_SEPS))
        if dialect.delimiter in _CANDIDATE_SEPS:
            return str(dialect.delimiter)
    except csv.Error:
        pass

    lines = [ln for ln in sample.splitlines() if ln.strip()][:5]
    if not lines:
        return None
    best, best_count = None, 0
    for sep in _CANDIDATE_SEPS:
        counts = {ln.count(sep) for ln in lines}
        if len(counts) == 1 and counts != {0}:
            count = counts.pop()
            if count > best_count:
                best, best_count = sep, count
    return best


def _first_line(path: Path, encoding: str) -> str:
    with open(path, "r", encoding=encoding, errors="replace", newline="") as fh:
        for line in fh:
            if line.strip():
                return line.rstrip("\r\n")
    return ""


def _delimiter_is_plausible(header: str, n_cols: int) -> bool:
    """Could a real delimiter have produced ``n_cols`` columns from this header line?

    The python engine's own sniffer considers *any* character a candidate delimiter, so
    on a genuinely single-column file it will happily decide that a letter is the
    separator and return a frame of shredded nonsense. That frame has more than one
    column, so a column count alone cannot reject it. This asks the weaker, sufficient
    question instead: does some ordinary delimiter occur often enough in the header to
    explain the split? Quoted fields make the count an upper bound, hence ``>=`` rather
    than ``==``.
    """
    if n_cols <= 1:
        return False
    need = n_cols - 1
    counts = [header.count(s) for s in _CANDIDATE_SEPS]
    counts.append(len(re.findall(r"\s+", header.strip())))
    return max(counts) >= need


def read_table(
    path: str | Path,
    *,
    sep: str | None = None,
    encoding: str = "utf-8-sig",
    dtype_str: bool = True,
    **read_csv_kw: Any,
) -> pd.DataFrame:
    """Read a delimited text file, escalating through separators until it parses.

    With ``sep=None`` the order is: the sniffed separator (or a comma), then a tab, then
    the python engine's own sniffing. The first attempt yielding a plausible split wins.
    A file that produces one column under all three raises, quoting the single column
    name -- which almost always contains the real delimiter in plain sight.

    The last attempt is gated. The python engine's sniffer treats any character as a
    candidate delimiter, so on a real single-column file it will pick a letter and return
    shredded nonsense with several columns and no error. Its result is accepted only if
    an ordinary delimiter in the header line could have produced that many columns.

    The separator actually used is recorded on ``df.attrs["separator"]`` so a notebook
    can print what it got rather than assume. Column names are stripped of surrounding
    whitespace and of a leading byte-order mark.

    Values are read as text by default (``dtype_str``); see the module docstring for why
    that is not a stylistic preference.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"no such file: {p}")

    kw: dict[str, Any] = dict(read_csv_kw)
    kw.setdefault("encoding", encoding)
    if dtype_str:
        kw.setdefault("dtype", str)

    def _attempt(candidate: str | None, engine: str | None) -> pd.DataFrame:
        extra: dict[str, Any] = {} if engine is None else {"engine": engine}
        try:
            return pd.read_csv(p, sep=candidate, **extra, **kw)
        except UnicodeDecodeError as exc:
            raise ValueError(
                f"{p.name} is not decodable as {kw['encoding']!r}; pass an explicit "
                "encoding= (files from spreadsheet exports are often 'latin-1' or "
                "'cp1252'). Guessing an encoding here would corrupt identifiers."
            ) from exc

    if sep is not None:
        df = _attempt(sep, None)
        used: str = repr(sep)
        if df.shape[1] == 1:
            raise ValueError(
                f"{p.name} parsed into a single column {df.columns[0]!r} with the "
                f"separator you passed ({sep!r}). Pass sep=None to let the reader "
                "escalate, or check the file."
            )
    else:
        first = sniff_separator(p, encoding=encoding) or ","
        plan: list[tuple[str | None, str | None, str]] = [(first, None, repr(first))]
        if first != TAB:
            plan.append((TAB, None, "chr(9)"))
        plan.append((None, "python", "python-engine sniffing"))

        header = _first_line(p, encoding)
        df, used, failures = None, "", []
        for candidate, engine, label in plan:
            try:
                got = _attempt(candidate, engine)
            except (pd.errors.ParserError, pd.errors.EmptyDataError, csv.Error) as exc:
                # A ragged parse is evidence against this separator, so try the next.
                # An encoding failure is not, and _attempt raises that straight through.
                failures.append(f"{label}: {type(exc).__name__}")
                continue
            if got.shape[1] == 1:
                failures.append(f"{label}: 1 column ({str(got.columns[0])[:60]!r})")
                continue
            if engine == "python" and not _delimiter_is_plausible(header, got.shape[1]):
                failures.append(
                    f"{label}: rejected, {got.shape[1]} columns not explained by any "
                    "ordinary delimiter in the header line"
                )
                continue
            df, used = got, label
            break
        if df is None:
            raise ValueError(
                f"{p.name} did not parse into more than one column. Attempts -- "
                + "; ".join(failures)
                + ". The single column name usually shows the real delimiter; pass it "
                "as sep=."
            )

    df.columns = [str(c).strip().lstrip(_BOM).strip() for c in df.columns]
    df.attrs["separator"] = used
    df.attrs["path"] = str(p)
    return df


# -------------------------------------------------------------------- construction


def from_csv(
    path: str | Path,
    *,
    source: _ColumnSpec,
    unit: _ColumnSpec,
    group: _ColumnSpec,
    weight: str | None = None,
    name: str | None = None,
    sep: str | None = None,
    encoding: str = "utf-8-sig",
    dtype_str: bool = True,
    dropna: bool = True,
    duplicates: str = "warn",
    hierarchy: Mapping[str, _ColumnSpec] | None = None,
    **read_csv_kw: Any,
) -> Corpus:
    """Build a corpus from a delimited text file.

    Parameters
    ----------
    source, unit, group
        Column names in the file. ``source`` fixes the **grain**; see ``grain.py``. The
        grain has to be declared beside every number computed from the corpus, because
        the same table can score near 0 at one grain and near 1 at another.
    weight
        Optional column of per-row data mass. Omitted means incidence weighting, which
        is a choice with consequences and is recorded on the returned corpus. A dataset-
        level total or a technical-replicate identifier is not row-level mass and is never
        inferred or filled here; pass only a measured value scoped to each analysis row.
    duplicates
        Forwarded to :meth:`Corpus.from_frame`; ``"warn"`` (default) makes repeated
        normalized biological observations visible without choosing deduplication for the
        caller.
    hierarchy
        Optional named parent-unit column specifications. They are retained with the
        input metadata, including missing values; no hierarchy is parsed from identifiers.
    sep
        Leave as None unless you know better; see :func:`read_table` for the escalation.
        A ``.csv`` that is really tab separated is common enough to be the default
        assumption of the retry path.

    Notes
    -----
    Anchor (worked-example corpus, incidence weighting): a table of 84 incidence rows
    over 43 sources, 44 units and 2 groups gives ratio 0.08319240360050288, with 40 of
    the 43 sources contributing exactly zero to H(G|D).
    """
    df = read_table(path, sep=sep, encoding=encoding, dtype_str=dtype_str, **read_csv_kw)
    return Corpus.from_frame(
        df,
        source=source,
        unit=unit,
        group=group,
        weight=weight,
        name=name or Path(path).stem,
        dropna=dropna,
        duplicates=duplicates,
        hierarchy=hierarchy,
    )


def from_excel(
    path: str | Path,
    *,
    sheet: str | int = 0,
    header: int = 0,
    source: _ColumnSpec,
    unit: _ColumnSpec,
    group: _ColumnSpec,
    weight: str | None = None,
    name: str | None = None,
    dtype_str: bool = True,
    dropna: bool = True,
    duplicates: str = "warn",
    hierarchy: Mapping[str, _ColumnSpec] | None = None,
    **read_excel_kw: Any,
) -> Corpus:
    """Build a corpus from one sheet of a workbook.

    ``header`` is explicit because published supplements routinely carry a title row, a
    legend row, or both above the real header, and a workbook read with ``header=0``
    then yields a frame whose column names are prose and whose first data row is the
    header. Open the sheet and look before trusting the default.

    ``sheet`` selects by name or position. Supplements often split a corpus across one
    sheet per group; read each and concatenate before calling this, or call it per sheet
    and combine the frames -- do not assume the sheets share a column layout.

    Requires an Excel engine (``pip install openpyxl`` for ``.xlsx``).
    """
    kw: dict[str, Any] = dict(read_excel_kw)
    if dtype_str:
        kw.setdefault("dtype", str)
    try:
        df = pd.read_excel(path, sheet_name=sheet, header=header, **kw)
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "reading Excel needs an engine: pip install openpyxl (for .xlsx) or xlrd "
            "(for legacy .xls)"
        ) from exc
    if isinstance(df, dict):  # pragma: no cover - only when sheet=None is forced through
        raise TypeError(
            "sheet must name a single sheet; got a dict of sheets. Read them "
            "individually so the column layout of each is checked."
        )
    df.columns = [str(c).strip().lstrip(_BOM).strip() for c in df.columns]
    return Corpus.from_frame(
        df,
        source=source,
        unit=unit,
        group=group,
        weight=weight,
        name=name or f"{Path(path).stem}[{sheet}]",
        dropna=dropna,
        duplicates=duplicates,
        hierarchy=hierarchy,
    )


def from_long(
    df: pd.DataFrame,
    *,
    source: _ColumnSpec = "source",
    unit: _ColumnSpec = "unit",
    group: _ColumnSpec = "group",
    weight: str | None = None,
    name: str = "corpus",
    dropna: bool = True,
    duplicates: str = "warn",
    hierarchy: Mapping[str, _ColumnSpec] | None = None,
) -> Corpus:
    """Build a corpus from a dataframe already in memory.

    A thin pass-through to :meth:`Corpus.from_frame`, provided so that code which took
    a path can be switched to a frame without changing shape. All validation lives in
    ``Corpus``; nothing is added here.
    """
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


# ---------------------------------------------------------------------- inspection


def _accession_fraction(values: pd.Series) -> float:
    text = values.dropna().astype(str).str.strip()
    if text.empty:
        return float("nan")
    hit = text.map(
        lambda v: bool(_ACCESSION_RE.match(v)) or bool(_HYPHEN_ACCESSION_RE.match(v))
    )
    return float(hit.mean())


def column_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Per-column evidence for the column-role question, as a table you can read.

    Columns: ``dtype``, ``n_missing``, ``n_unique``, ``frac_unique`` (unique values per
    non-missing row), ``frac_numeric`` (cells parsing as a bare number),
    ``frac_accession_like``, ``median_len``, ``looks_like_prose``.

    This is the evidence :func:`infer_columns` scores. Print it when a suggestion looks
    wrong -- it usually shows why in one line.
    """
    rows = []
    for col in df.columns:
        s = df[col]
        text = s.astype(str).str.strip()
        missing = s.isna() | text.isin(["", "nan", "None", "NaN", "<NA>"])
        present = s[~missing]
        n_present = int(len(present))
        numeric = pd.to_numeric(present, errors="coerce") if n_present else pd.Series(dtype=float)
        rows.append(
            {
                "column": str(col),
                "dtype": str(s.dtype),
                "n_missing": int(missing.sum()),
                "n_unique": int(present.nunique()) if n_present else 0,
                "frac_unique": (present.nunique() / n_present) if n_present else float("nan"),
                "frac_numeric": float(numeric.notna().mean()) if n_present else float("nan"),
                "frac_accession_like": _accession_fraction(present),
                "median_len": (
                    float(text[~missing].str.len().median()) if n_present else float("nan")
                ),
                "looks_like_prose": looks_like_prose(s),
            }
        )
    return pd.DataFrame(rows).set_index("column")


def _hit(name: str, hints: Sequence[str]) -> bool:
    low = str(name).strip().lower()
    return any(h in low for h in hints)


def infer_columns(
    df: pd.DataFrame,
    *,
    hints: Mapping[str, Sequence[str]] | None = None,
    top: int = 3,
) -> dict[str, list[str]]:
    """Suggest which columns might be source, unit, group and weight.

    Returns ``{"source": [...], "unit": [...], "group": [...], "weight": [...]}``, each
    a list of column names ranked best-guess first and capped at ``top``. A role with no
    plausible candidate gets an empty list rather than a bad guess.

    **These are suggestions and this package will never apply them.** The source column
    *is* the grain, and the grain is an argument the analyst has to make and defend: the
    same table can score a ratio near 0 with one column as source and near 1 with
    another, and no heuristic can adjudicate that. Likewise a two-level column is not
    automatically the group -- it may be a batch that happens to have two levels.

    The scoring is transparent on purpose: name-substring hints (extend or replace them
    with ``hints``), accession-shaped values for source, low cardinality for group,
    moderate-to-high cardinality for unit, short values for all three, and fully numeric
    non-prose columns for weight. Columns named as dataset/study totals or technical
    replicates are excluded from weight suggestions because their scope does not match an
    analysis row. Call :func:`column_profile` to see the evidence behind any suggestion.

    An empty ``weight`` list is itself informative: it means no column of this table
    parses cleanly as data mass, so only incidence weighting is available and that has to
    be declared. Published supplements often carry a column whose *name* promises counts
    and whose contents are methods prose; :func:`looks_like_prose` keeps those out. No
    suggestion is ever applied and no missing sample value is filled from an aggregate.
    """
    merged = {role: tuple(_ROLE_HINTS[role]) for role in _ROLE_HINTS}
    if hints:
        for role, extra in hints.items():
            if role not in merged:
                raise KeyError(f"unknown role {role!r}; roles are {sorted(merged)}")
            merged[role] = tuple(extra)

    prof = column_profile(df)
    n_rows = max(int(len(df)), 1)
    scores: dict[str, dict[str, float]] = {role: {} for role in merged}

    for col, row in prof.iterrows():
        n_unique = float(row["n_unique"])
        frac_unique = float(row["frac_unique"]) if np.isfinite(row["frac_unique"]) else 0.0
        frac_acc = float(row["frac_accession_like"]) if np.isfinite(row["frac_accession_like"]) else 0.0
        frac_num = float(row["frac_numeric"]) if np.isfinite(row["frac_numeric"]) else 0.0
        prose = bool(row["looks_like_prose"])
        if n_unique <= 1:
            continue  # a constant column can play no role in a contrast

        # Identifiers and labels are short. A column of paragraphs is a description of
        # a source, unit or group -- never the key itself.
        med_len = float(row["median_len"]) if np.isfinite(row["median_len"]) else 0.0
        length = 0.5 if med_len <= 25 else (-2.0 if med_len > 60 else 0.0)

        s = 3.0 * _hit(col, merged["source"]) + 4.0 * frac_acc + length
        s += 1.0 if 1 < n_unique < n_rows else 0.0
        s -= 2.0 if prose else 0.0
        scores["source"][str(col)] = s

        low_card = 2 <= n_unique <= max(10.0, 0.05 * n_rows)
        g = 3.0 * _hit(col, merged["group"]) + (3.0 if n_unique <= 10 else 0.0) + length
        g += 1.5 if low_card else 0.0
        g -= 2.0 * frac_acc + (4.0 if prose else 0.0)
        g -= 2.0 if frac_unique > 0.5 else 0.0
        scores["group"][str(col)] = g

        u = 3.0 * _hit(col, merged["unit"]) + length
        u += 2.0 if n_unique >= max(3.0, 0.05 * n_rows) else 0.0
        u -= 1.0 if frac_unique >= 1.0 else 0.0  # more likely a row id than a unit
        u -= 2.0 if prose else 0.0
        scores["unit"][str(col)] = u

        if not _unsafe_automatic_weight_name(col):
            w = 3.0 * _hit(col, merged["weight"]) + 3.0 * frac_num
            w -= 5.0 if prose else 0.0
            w -= 3.0 if frac_acc > 0.5 else 0.0
            scores["weight"][str(col)] = w

    # A candidate has to clear more than the one point every non-constant column earns
    # for being non-constant; otherwise the list fills with columns there is no evidence
    # for, and a suggestion nobody believes is worse than no suggestion.
    floor = 2.0
    out: dict[str, list[str]] = {}
    for role, table in scores.items():
        ranked = sorted((c for c, v in table.items() if v >= floor),
                        key=lambda c: (-table[c], c))
        out[role] = ranked[:top]

    # Do not offer the same column as both source and unit at the top of both lists.
    if out["source"] and out["unit"] and out["source"][0] == out["unit"][0]:
        if len(out["unit"]) > 1:
            out["unit"] = out["unit"][1:] + out["unit"][:1]
    return out
