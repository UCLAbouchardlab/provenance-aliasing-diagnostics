"""Strict CSV/TSV ingestion that preserves declared metadata identifiers."""
from __future__ import annotations

import csv
from contextlib import contextmanager
import hashlib
from io import BufferedReader, RawIOBase, TextIOWrapper
from os import PathLike
from pathlib import Path
from typing import BinaryIO, Iterator, TextIO

import pandas as pd


class TableReadError(ValueError):
    """A metadata file cannot be read as an unambiguous rectangular text table."""


class _FingerprintReader(RawIOBase):
    """Hash exactly the bytes delivered to the parser's decoding stream."""

    def __init__(self, stream: BinaryIO) -> None:
        super().__init__()
        self._stream = stream
        self.digest = hashlib.sha256()
        self.size_bytes = 0

    def readable(self) -> bool:
        return True

    def readinto(self, buffer) -> int | None:
        count = self._stream.readinto(buffer)
        if count:
            self.digest.update(memoryview(buffer)[:count])
            self.size_bytes += count
        return count

    def close(self) -> None:
        try:
            self._stream.close()
        finally:
            super().close()


@contextmanager
def _fingerprinted_text(source: Path) -> Iterator[tuple[TextIO, _FingerprintReader]]:
    with source.open("rb", buffering=0) as binary:
        captured = _FingerprintReader(binary)
        with TextIOWrapper(BufferedReader(captured), encoding="utf-8-sig", errors="strict", newline="") as text:
            yield text, captured


def _checked_lines(stream: TextIO, source: Path) -> Iterator[str]:
    """Yield unchanged physical lines, rejecting NUL and decoding errors early."""
    line_number = 0
    while True:
        try:
            line = next(stream)
        except StopIteration:
            return
        except UnicodeDecodeError as exc:
            # TextIO may decode ahead before returning a physical line. Count
            # line breaks in the valid part of that decoder buffer for useful
            # error context without reading or copying the entire file.
            prefix = exc.object[:exc.start]
            breaks = prefix.count(b"\n") + prefix.count(b"\r") - prefix.count(b"\r\n")
            approximate_line = line_number + breaks + 1
            raise TableReadError(
                f"{source}: invalid UTF-8 near file line {approximate_line}: {exc.reason}."
            ) from exc
        line_number += 1
        if "\x00" in line:
            raise TableReadError(f"{source}: embedded NUL character at file line {line_number}.")
        yield line


def read_metadata(path: str | PathLike[str], *, format: str | None = None) -> pd.DataFrame:
    """Read UTF-8 CSV/TSV without changing headers, values, or record boundaries.

    An explicit ``format`` is ``"csv"`` or ``"tsv"``. Otherwise the file must
    have a case-insensitive ``.csv`` or ``.tsv`` suffix. There is no delimiter
    sniffing or decompression. A leading UTF-8 BOM is accepted; all cell values,
    including empty strings and missing-looking identifiers, remain strings.

    Every logical data record must match the nonblank, unique header. Blank
    records, empty files, and header-only files raise :class:`TableReadError`.
    File-system errors propagate as :class:`OSError` subclasses.
    """
    return read_metadata_with_fingerprint(path, format=format)[0]


def read_metadata_with_fingerprint(
    path: str | PathLike[str], *, format: str | None = None,
) -> tuple[pd.DataFrame, dict[str, str | int]]:
    """Read metadata and fingerprint the same byte stream consumed by parsing.

    The fingerprint covers the original file bytes, including its BOM and line
    endings. No second file read, delimiter inference, or content copy is used.
    The returned file identity contains the basename rather than an absolute path.
    """
    source = Path(path)
    selected = format
    if selected is None:
        selected = {".csv": "csv", ".tsv": "tsv"}.get(source.suffix.lower())
        if selected is None:
            raise TableReadError(
                f"{source}: cannot infer table format from suffix {source.suffix!r}; "
                "use a .csv or .tsv file, or explicitly set format='csv' or format='tsv'."
            )
    if not isinstance(selected, str) or selected not in ("csv", "tsv"):
        raise TableReadError(f"{source}: format must be 'csv' or 'tsv'.")

    delimiter = "," if selected == "csv" else "\t"
    with _fingerprinted_text(source) as (stream, captured):
        reader = csv.reader(_checked_lines(stream, source), delimiter=delimiter, strict=True)

        def next_record(record_number: int) -> list[str] | None:
            try:
                return next(reader)
            except StopIteration:
                return None
            except csv.Error as exc:
                location = "header" if record_number == 0 else f"data record {record_number}"
                raise TableReadError(
                    f"{source}: cannot parse {selected.upper()} {location} near file line "
                    f"{max(reader.line_num, 1)}: {exc}."
                ) from exc

        header = next_record(0)
        if header is None:
            raise TableReadError(f"{source}: the metadata file is empty; a header and data records are required.")
        if not header:
            raise TableReadError(f"{source}: blank header at file line {reader.line_num}.")
        blank_columns = [number for number, name in enumerate(header, start=1) if not name.strip()]
        if blank_columns:
            raise TableReadError(
                f"{source}: blank header name in column(s) {', '.join(map(str, blank_columns))} "
                f"at file line {reader.line_num}."
            )
        seen: set[str] = set()
        duplicates: list[str] = []
        for name in header:
            if name in seen and name not in duplicates:
                duplicates.append(name)
            seen.add(name)
        if duplicates:
            raise TableReadError(
                f"{source}: duplicate header name(s) {duplicates!r} at file line {reader.line_num}."
            )

        rows: list[list[str]] = []
        record_number = 1
        while (record := next_record(record_number)) is not None:
            if len(record) != len(header):
                raise TableReadError(
                    f"{source}: data record {record_number}, ending at file line {reader.line_num}, "
                    f"has {len(record)} field(s); expected {len(header)}. "
                    "Blank records and ragged rows are not permitted."
                )
            rows.append(record)
            record_number += 1
        if not rows:
            raise TableReadError(f"{source}: the file contains only a header; at least one data record is required.")

    fingerprint = {
        "kind": "file", "name": source.name, "sha256": captured.digest.hexdigest(),
        "size_bytes": captured.size_bytes, "format": selected,
        "encoding": "utf-8-sig", "delimiter": delimiter,
    }
    return pd.DataFrame(rows, columns=header, dtype=object), fingerprint
