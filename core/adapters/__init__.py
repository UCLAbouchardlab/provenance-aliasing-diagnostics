"""Adapters: messy real metadata in, a :class:`~provenance.incidence.Corpus` out.

This is the only layer of the package allowed to know about file formats, repository
conventions or the habits of a particular corpus. Everything above it takes objects and
returns diagnostics, which is what lets the same diagnostics run on any study whose
metadata can be written as ``(source, unit, group[, weight])``.

Three entry points, by the shape the metadata arrives in:

``tidy``
    A delimited file or a workbook with one row per observation. Start here.
    ``from_csv``, ``from_excel``, ``from_long``, and ``infer_columns`` /
    ``column_profile`` for working out which column is which.

``pride``
    Repository-style corpora where provenance is encoded in accessions and run names.
    ``parse_run_names``, ``from_filenames``, ``from_sdrf``. Offline: no network calls.

``aggregated_supplement``
    The published-supplement shape, one row per (source, unit) incidence.
    ``from_incidence_table``, and ``check_recoverability`` -- which answers whether a
    published table can be scored at all, and is a result in its own right.

Writing a new adapter
---------------------
1. Add a module here. Take a path or a raw object; return a ``Corpus``. Never return a
   half-cleaned dataframe -- if it cannot become a corpus, raise and say what is missing.
2. Keep the format knowledge inside it. No core module may import a format, and no
   adapter may put a corpus-specific accession, tissue or identifier in its code. Those
   are arguments the caller supplies.
3. Choose nothing silently. The source column is the grain and the weight column is the
   weighting, and both change the answer; make the caller name them. Where a convention
   must be assumed, validate it and raise on anything that does not fit, listing
   offenders. A parser that defaults a missing field destroys the within-group structure
   that later nulls are built from, and says nothing.
4. Pass the provenance of the numbers along: give the corpus a ``name`` that says where
   it came from and at what grain, and let ``Corpus.from_frame`` set the weighting. Pass
   through its duplicate policy and any explicit unit hierarchy; do not deduplicate inside
   an adapter or infer a donor from a sample name.
5. Read identifiers as text. ``007`` is a donor, not the number seven.
6. Dataset/study totals and technical-replicate identifiers are metadata at a different
   scope, not per-row weights. Preserve missing values and never fill sample counts from
   those totals.
7. Re-export the public functions here, and note the new module in the list above.
"""

from __future__ import annotations

from .aggregated_supplement import (
    check_recoverability,
    from_incidence_table,
    looks_like_prose,
    numeric_audit,
)
from .pride import from_filenames, from_sdrf, parse_run_names
from .tidy import (
    TAB,
    column_profile,
    from_csv,
    from_excel,
    from_long,
    infer_columns,
    read_table,
    sniff_separator,
)

__all__ = [
    # tidy
    "TAB",
    "sniff_separator",
    "read_table",
    "from_csv",
    "from_excel",
    "from_long",
    "column_profile",
    "infer_columns",
    # pride
    "parse_run_names",
    "from_filenames",
    "from_sdrf",
    # aggregated_supplement
    "from_incidence_table",
    "check_recoverability",
    "looks_like_prose",
    "numeric_audit",
]
