"""Adapters for repository-style corpora, where provenance lives in names.

Public data repositories encode provenance in two places: the accession of the deposit,
and the structure of the run or file names inside it. Neither is a *given* grain -- both
are conventions, and this module's job is to turn a convention the caller declares into
a :class:`~provenance.incidence.Corpus`.

**This module is offline.** It performs no network calls, by design: a reproducible
diagnostic cannot depend on what an API returned on a Tuesday. Fetch your manifest with
whatever client you like, save it, and pass it here.

**If you do fetch a manifest, paginate.** A repository API that accepts a ``pageSize``
parameter may cap it silently -- PRIDE ignores ``pageSize`` above 100 and returns the
first page without saying so. This project made that mistake three times. A manifest
truncated at 100 records does not fail loudly; it produces a smaller, tidier corpus that
scores differently on every quantity in this package. If your file count is exactly 100,
or a round multiple of the cap, assume truncation until you have checked.

**Grain.** An accession is a defensible grain and it is not the only one. A single
deposit can hold several laboratories, instruments and acquisition batches; conversely
one laboratory may spread one experiment over several deposits, in which case scoring at
accession grain reports crossing that does not physically exist. Whichever you choose,
declare it with the number, and use ``grain.py`` to score several and show the spread --
the same corpus can score near 0 at one grain and near 1 at another, so a ratio with no
declared grain is not a result.

**Filenames are metadata that was never validated by anyone.** Parsing them is fine;
guessing at them is not. :func:`parse_run_names` therefore takes an explicit token schema
and raises on any name that does not fit it, rather than defaulting a missing field. A
parser that quietly substitutes a default for an absent replicate token merges replicates
into one unit, which deflates the within-group variance the null is built from and turns
a negative result positive. That failure is silent in every output except this one.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, Union

import numpy as np
import pandas as pd

from ..incidence import Corpus
from .tidy import read_table

__all__ = [
    "parse_run_names",
    "from_filenames",
    "from_sdrf",
]

_ColumnSpec = Union[str, Sequence[str]]

_RESERVED_FIELDS = ("run", "n_tokens")
_EXTENSION_RE = re.compile(r"\.[A-Za-z0-9]{1,8}$")


# ---------------------------------------------------------------------- filenames


def _basename(name: str) -> str:
    return str(name).replace("\\", "/").rsplit("/", 1)[-1]


def _clean_name(name: str, *, strip_directory: bool, strip_extension: bool) -> str:
    out = str(name).strip()
    if strip_directory:
        out = _basename(out)
    if strip_extension:
        out = _EXTENSION_RE.sub("", out)
    return out


def parse_run_names(
    names: Iterable[str],
    schema: Mapping[str, int],
    *,
    sep: str = "_",
    strip_directory: bool = True,
    strip_extension: bool = True,
    expected_tokens: int | None = None,
    allow_duplicates: bool = False,
    max_report: int = 10,
) -> pd.DataFrame:
    """Split structured run or file names into declared provenance fields.

    Parameters
    ----------
    names
        The run or file names, exactly as the repository lists them. Directory prefixes
        and a single trailing extension are removed by default.
    schema
        ``{field: token index}`` for the tokens of ``sep``-separated names, for example
        ``{"donor": 2, "tissue": 3, "antibody": 4, "replicate": 7}``. Negative indices
        count from the end. Field names may not be ``run`` or ``n_tokens``.
    expected_tokens
        The token count every name must have. When None it is taken from the most common
        count observed, and any name disagreeing with it is an offender. Pass it
        explicitly when you know the convention: then a manifest in which *every* name is
        wrong is caught too, which the modal rule cannot do.

    Returns
    -------
    DataFrame indexed by the original names, with one column per schema field plus
    ``run`` (the cleaned name) and ``n_tokens``. The index is the input string so the
    result can be joined straight back onto a manifest.

    Raises
    ------
    ValueError
        If any name has the wrong token count, if any token a field points at is empty,
        or if names repeat (unless ``allow_duplicates``). Offenders are listed.

    Notes
    -----
    The raising is the feature. A heuristic parser that fills a missing field with a
    default -- an absent replicate token becoming ``"1"``, say -- merges distinct runs
    into one unit. That collapses within-group variation, which is exactly the quantity a
    within-group null is estimated from, so the design looks better than it is and
    nothing in the output says why. Better to stop and make the caller state the schema.

    Repeated names raise for a related reason: in practice they mean a manifest was
    concatenated twice, or paginated wrongly (see the module docstring). Pass
    ``allow_duplicates=True`` when the repeats are genuinely distinct records.
    """
    listed = [str(n) for n in names]
    if not listed:
        raise ValueError("no names given")
    if not schema:
        raise ValueError("schema is empty: name at least one field -> token index")
    if not sep:
        raise ValueError("sep must be a non-empty string")

    fields: dict[str, int] = {}
    for field, idx in schema.items():
        if not isinstance(field, str) or not field.strip():
            raise ValueError(f"field names must be non-empty strings; got {field!r}")
        if field in _RESERVED_FIELDS:
            raise ValueError(f"field name {field!r} is reserved; rename it")
        if isinstance(idx, bool) or not isinstance(idx, (int, np.integer)):
            raise TypeError(f"schema[{field!r}] must be an int token index; got {idx!r}")
        fields[field] = int(idx)

    need = max(
        max((i + 1 for i in fields.values() if i >= 0), default=0),
        max((-i for i in fields.values() if i < 0), default=0),
    )

    cleaned = [
        _clean_name(n, strip_directory=strip_directory, strip_extension=strip_extension)
        for n in listed
    ]
    tokens = [c.split(sep) for c in cleaned]
    counts = pd.Series([len(t) for t in tokens], index=pd.Index(listed, name="name"))

    if expected_tokens is None:
        expected = int(counts.mode().iloc[0])
    else:
        expected = int(expected_tokens)

    if expected < need:
        raise ValueError(
            f"schema needs at least {need} tokens but names have {expected}: "
            f"schema {dict(fields)}, example {cleaned[0]!r} splits into {len(tokens[0])} "
            f"tokens on {sep!r}"
        )

    offenders = counts[counts != expected]
    if len(offenders):
        distribution = counts.value_counts().sort_index().to_dict()
        listing = "; ".join(
            f"{name!r} -> {n} tokens" for name, n in offenders.head(max_report).items()
        )
        raise ValueError(
            f"{len(offenders)} of {len(listed)} names do not have {expected} "
            f"{sep!r}-separated tokens. Token-count distribution: {distribution}. "
            f"First offenders: {listing}"
            + ("" if len(offenders) <= max_report else " ...")
            + ". Fix the schema or the manifest -- do not let a parser default the "
            "missing field, because that silently merges records."
        )

    out = pd.DataFrame(index=pd.Index(listed, name="name"))
    out["run"] = cleaned
    empty_report: list[str] = []
    for field, idx in fields.items():
        pos = idx if idx >= 0 else expected + idx
        values = [t[pos].strip() for t in tokens]
        out[field] = values
        blank = [listed[i] for i, v in enumerate(values) if not v]
        if blank:
            empty_report.append(
                f"{field!r} (token {idx}) is empty in {len(blank)} names, e.g. "
                f"{blank[:3]}"
            )
    if empty_report:
        raise ValueError(
            "empty tokens where a field was expected -- usually a doubled separator: "
            + "; ".join(empty_report)
        )

    out["n_tokens"] = counts.to_numpy()

    if not allow_duplicates:
        dup = out.index[out.index.duplicated(keep="first")]
        if len(dup):
            raise ValueError(
                f"{len(dup)} repeated names, e.g. {list(dict.fromkeys(dup))[:max_report]}. "
                "Repeats usually mean a manifest was concatenated or paginated twice; "
                "pass allow_duplicates=True if they are genuinely distinct records."
            )
    return out


def from_filenames(
    names: Iterable[str],
    schema: Mapping[str, int],
    *,
    source_field: _ColumnSpec,
    unit_field: _ColumnSpec,
    group_field: _ColumnSpec,
    weight_field: str | None = None,
    name: str = "runs",
    sep: str = "_",
    strip_directory: bool = True,
    strip_extension: bool = True,
    expected_tokens: int | None = None,
    allow_duplicates: bool = False,
    dropna: bool = True,
    duplicates: str = "warn",
    hierarchy: Mapping[str, _ColumnSpec] | None = None,
) -> Corpus:
    """Build a corpus directly from structured run or file names.

    The three ``*_field`` arguments must name fields of ``schema``, and choosing
    ``source_field`` is choosing the grain: donor, laboratory, acquisition batch and
    deposit are all reachable from a filename convention and they do not agree. Declare
    which one you used with every number you report, and see ``grain.py`` for scoring
    several at once.

    ``weight_field`` is offered only for a measured value scoped to the resulting analysis
    row. A replicate token is an identifier or design descriptor, not data mass, and a
    dataset-level total must never be copied across its sample rows. Leaving ``weight_field``
    as None gives incidence weighting, which is recorded on the corpus.

    ``allow_duplicates`` controls identical input names in the manifest. ``duplicates``
    is a separate biological-observation policy applied after parsing: distinct HLA-I and
    HLA-II filenames may map to the same source/unit key and must reach
    :meth:`Corpus.from_frame` for that decision. ``hierarchy`` names explicit parent-unit
    fields such as donor; parents are never inferred from filename prefixes.

    Every validation in :func:`parse_run_names` applies, which means this raises rather
    than parses partially.
    """
    def fields(spec: _ColumnSpec) -> list[str]:
        return [spec] if isinstance(spec, str) else [str(field) for field in spec]

    for role, spec in (("source_field", source_field), ("unit_field", unit_field),
                       ("group_field", group_field)):
        missing = [field for field in fields(spec) if field not in schema]
        if missing:
            raise KeyError(
                f"{role} contains fields not in the schema: {missing}; declared fields are "
                f"{sorted(schema)}"
            )
    if weight_field is not None and weight_field not in schema:
        raise KeyError(
            f"weight_field={weight_field!r} is not in the schema; declared fields are "
            f"{sorted(schema)}"
        )
    if hierarchy is not None:
        for level, spec in hierarchy.items():
            missing = [field for field in fields(spec) if field not in schema]
            if missing:
                raise KeyError(
                    f"hierarchy[{level!r}] contains fields not in the schema: {missing}; "
                    f"declared fields are {sorted(schema)}"
                )

    parsed = parse_run_names(
        names,
        schema,
        sep=sep,
        strip_directory=strip_directory,
        strip_extension=strip_extension,
        expected_tokens=expected_tokens,
        allow_duplicates=allow_duplicates,
    )
    return Corpus.from_frame(
        parsed,
        source=source_field,
        unit=unit_field,
        group=group_field,
        weight=weight_field,
        name=name,
        dropna=dropna,
        duplicates=duplicates,
        hierarchy=hierarchy,
    )


# --------------------------------------------------------------------------- SDRF


def _resolve_column(df: pd.DataFrame, requested: str, *, role: str) -> str:
    """Match a column name exactly, then case-insensitively, then loosely.

    Sample-metadata headers are written by hand and vary in case and spacing across
    submissions, so an exact-match-only lookup fails on files that are otherwise fine.
    The match is reported by returning the real column name; nothing is renamed.
    """
    if requested in df.columns:
        return requested

    def norm(text: object) -> str:
        return re.sub(r"\s+", " ", str(text).strip().lower())

    wanted = norm(requested)
    hits = [c for c in df.columns if norm(c) == wanted]
    if len(hits) == 1:
        return str(hits[0])
    if len(hits) > 1:
        raise KeyError(
            f"{role}={requested!r} matches {len(hits)} columns after case folding: "
            f"{hits}. Pass the exact column name."
        )
    raise KeyError(
        f"{role}={requested!r} not found. Available columns: {list(df.columns)[:20]}"
        + (" ..." if len(df.columns) > 20 else "")
    )


def _resolve_spec(df: pd.DataFrame, requested: _ColumnSpec, *, role: str) -> _ColumnSpec:
    """Resolve every physical column in a scalar or composite key specification."""
    if isinstance(requested, str):
        return _resolve_column(df, requested, role=role)
    resolved = [
        _resolve_column(df, str(column), role=f"{role}[{index}]")
        for index, column in enumerate(requested)
    ]
    return tuple(resolved) if isinstance(requested, tuple) else resolved


def from_sdrf(
    path_or_df: str | Path | pd.DataFrame,
    *,
    source_col: _ColumnSpec,
    unit_col: _ColumnSpec,
    group_col: _ColumnSpec,
    weight_col: str | None = None,
    name: str | None = None,
    dropna: bool = True,
    duplicates: str = "warn",
    hierarchy: Mapping[str, _ColumnSpec] | None = None,
    **read_kw: Any,
) -> Corpus:
    """Build a corpus from a sample-and-data-relationship (SDRF) style metadata file.

    Accepts a path or an in-memory frame. Files are read with
    :func:`provenance.adapters.tidy.read_table`, which escalates separators: SDRF is
    nominally tab separated, real deposits are not always. Column names are matched
    exactly first, then case-insensitively, because these headers are hand-written.

    Caveats worth knowing before you pass column names:

    * **"source name" in SDRF is not provenance.** In that vocabulary it denotes the
      biological sample the material came from, which is usually a *unit* here, not a
      source. The provenance grain is more often carried by an assay, file, instrument
      or batch column. Mapping the two "source" words onto each other produces a corpus
      whose sources are units, and it will score as though the design were crossed.
    * Only a minority of repository projects deposit a sample-metadata file at all, so
      this adapter covers the well-annotated tail of a corpus rather than its bulk.
      Whatever you do for the remainder has to be stated, because "the projects with
      metadata" is not a random subset of projects.
    * Characteristic columns are often named with brackets, e.g.
      ``characteristics[disease]``. Pass them verbatim.
    * ``duplicates`` is applied after column resolution, so separate assay/run rows that
      resolve to one biological source/unit cannot bypass Corpus duplicate detection.
      ``hierarchy`` records explicit parent columns; missing parents remain missing and
      are never guessed from a sample name or dataset total.
    """
    if isinstance(path_or_df, pd.DataFrame):
        df = path_or_df.copy()
        df.columns = [str(c).strip() for c in df.columns]
        default_name = "sdrf"
    else:
        df = read_table(path_or_df, **read_kw)
        default_name = Path(path_or_df).stem

    source = _resolve_spec(df, source_col, role="source_col")
    unit = _resolve_spec(df, unit_col, role="unit_col")
    group = _resolve_spec(df, group_col, role="group_col")
    weight = None if weight_col is None else _resolve_column(df, weight_col, role="weight_col")
    resolved_hierarchy = None
    if hierarchy is not None:
        resolved_hierarchy = {
            str(level): _resolve_spec(df, spec, role=f"hierarchy[{level!r}]")
            for level, spec in hierarchy.items()
        }

    return Corpus.from_frame(
        df,
        source=source,
        unit=unit,
        group=group,
        weight=weight,
        name=name or default_name,
        dropna=dropna,
        duplicates=duplicates,
        hierarchy=resolved_hierarchy,
    )
