"""Strict text ingestion keeps metadata identity and rejects ambiguous records."""
from __future__ import annotations

import hashlib
from io import BytesIO
from pathlib import Path

import pandas as pd
import pytest

from provenance_aliasing.api._tabular import TableReadError, read_metadata, read_metadata_with_fingerprint


def write_bytes(tmp_path: Path, content: bytes, name: str = "metadata.csv") -> Path:
    path = tmp_path / name
    path.write_bytes(content)
    return path


def test_values_remain_exact_strings_without_numeric_or_missing_inference(tmp_path) -> None:
    path = write_bytes(tmp_path, b"source,unit,group,note\nNA,001,nan,\nnone,01,<NA>,  keep spaces  \n")
    original = path.read_bytes()
    actual = read_metadata(path)
    expected = pd.DataFrame({"source": ["NA", "none"], "unit": ["001", "01"],
                             "group": ["nan", "<NA>"], "note": ["", "  keep spaces  "]}, dtype=object)
    pd.testing.assert_frame_equal(actual, expected)
    assert isinstance(actual.index, pd.RangeIndex)
    assert all(isinstance(value, str) for row in actual.itertuples(index=False, name=None) for value in row)
    assert path.read_bytes() == original


def test_nonblank_header_whitespace_is_preserved_without_normalized_duplicate_detection(tmp_path) -> None:
    path = write_bytes(tmp_path, b"source, source ,unit\nS1,S2,001\n")
    actual = read_metadata(path)
    assert list(actual.columns) == ["source", " source ", "unit"]
    assert actual.iloc[0].tolist() == ["S1", "S2", "001"]


def test_utf8_bom_unicode_and_string_path_are_supported(tmp_path) -> None:
    path = write_bytes(tmp_path, "source,unit,group\nétude,001,α\n".encode("utf-8-sig"), "META.CSV")
    actual = read_metadata(str(path))
    assert actual.to_dict("list") == {"source": ["étude"], "unit": ["001"], "group": ["α"]}


def test_only_the_initial_bom_is_removed(tmp_path) -> None:
    path = write_bytes(tmp_path, "source,unit\nS1,\ufeff001\n".encode("utf-8-sig"))
    assert read_metadata(path).at[0, "unit"] == "\ufeff001"


def test_quoted_delimiters_escaped_quotes_and_multiline_fields_are_lossless(tmp_path) -> None:
    path = write_bytes(tmp_path, b'source,unit,note\r\n"S,1",001,"first\r\n\r\nsecond ""quoted"" line"\r\nS2,002,"last"\r\n')
    actual = read_metadata(path)
    assert actual.to_dict("list") == {
        "source": ["S,1", "S2"], "unit": ["001", "002"],
        "note": ['first\r\n\r\nsecond "quoted" line', "last"],
    }
    assert actual.index.tolist() == [0, 1]


@pytest.mark.parametrize("newline", [b"\n", b"\r\n", b"\r"])
def test_record_newline_styles_do_not_change_values(tmp_path, newline) -> None:
    path = write_bytes(tmp_path, newline.join([b"source,unit", b"S1,001", b"S2,002", b""]))
    assert read_metadata(path).to_dict("list") == {"source": ["S1", "S2"], "unit": ["001", "002"]}


def test_one_column_and_empty_quoted_cell_are_valid_rectangular_data(tmp_path) -> None:
    path = write_bytes(tmp_path, b'unit\n""\n001')
    assert read_metadata(path).to_dict("list") == {"unit": ["", "001"]}


def test_tsv_suffix_and_explicit_format_override_use_only_the_declared_delimiter(tmp_path) -> None:
    content = b'source\tunit\tnote\nS1\t001\t"has\ta tab, and comma"\n'
    inferred = write_bytes(tmp_path, content, "metadata.TSV")
    overridden = write_bytes(tmp_path, content, "metadata.anything")
    expected = pd.DataFrame({"source": ["S1"], "unit": ["001"], "note": ["has\ta tab, and comma"]})
    pd.testing.assert_frame_equal(read_metadata(inferred), expected)
    pd.testing.assert_frame_equal(read_metadata(overridden, format="tsv"), expected)
    wrong_suffix = write_bytes(tmp_path, b"source,unit\nS1,001\n", "comma.tsv")
    assert read_metadata(wrong_suffix).to_dict("list") == {"source,unit": ["S1,001"]}
    assert read_metadata(wrong_suffix, format="csv").to_dict("list") == {"source": ["S1"], "unit": ["001"]}


@pytest.mark.parametrize("content", [b"source,unit,source\nS1,001,S2\n", b'"source",unit,"source"\nS1,001,S2\n'])
def test_duplicate_headers_are_rejected_before_dataframe_construction(tmp_path, content) -> None:
    path = write_bytes(tmp_path, content)
    with pytest.raises(TableReadError, match="duplicate header.*source"):
        read_metadata(path)


@pytest.mark.parametrize("content", [b"source,,group\nS1,001,A\n", b"source,  ,group\nS1,001,A\n", b"\nS1,001,A\n"])
def test_blank_headers_are_rejected(tmp_path, content) -> None:
    path = write_bytes(tmp_path, content)
    with pytest.raises(TableReadError, match="blank header"):
        read_metadata(path)


@pytest.mark.parametrize("content,record,field_count", [
    (b"source,unit\nS1\n", 1, 1),
    (b"source,unit\nS1,001,extra\n", 1, 3),
    (b"source,unit\nS1,001\n\n", 2, 0),
    (b"source,unit\n\nS1,001\n", 1, 0),
])
def test_ragged_and_blank_records_are_errors_with_record_and_line_context(tmp_path, content, record, field_count) -> None:
    path = write_bytes(tmp_path, content)
    with pytest.raises(TableReadError) as caught:
        read_metadata(path)
    message = str(caught.value)
    assert f"data record {record}" in message
    assert f"has {field_count} field(s); expected 2" in message
    assert "file line" in message
    assert str(path) in message


def test_ragged_record_after_multiline_field_reports_logical_record_separately(tmp_path) -> None:
    path = write_bytes(tmp_path, b'source,note\nS1,"line one\nline two"\nS2\n')
    with pytest.raises(TableReadError, match="data record 2, ending at file line 4"):
        read_metadata(path)


@pytest.mark.parametrize("content", [b"", b"\xef\xbb\xbf", b"source,unit", b"source,unit\n"])
def test_empty_and_header_only_inputs_are_errors(tmp_path, content) -> None:
    path = write_bytes(tmp_path, content)
    with pytest.raises(TableReadError, match="empty|only a header"):
        read_metadata(path)


@pytest.mark.parametrize("content", [b'source,unit\nS1,"unterminated\n', b'source,unit\nS1,"quoted"junk\n', b'"source,unit\nS1,001\n'])
def test_malformed_quoting_is_a_table_error_with_parser_location(tmp_path, content) -> None:
    path = write_bytes(tmp_path, content)
    with pytest.raises(TableReadError, match="cannot parse CSV.*file line"):
        read_metadata(path)


@pytest.mark.parametrize("content", [b"sour\xffce,unit\nS1,001\n", b"source,unit\nS1,\xff\n"])
def test_invalid_utf8_is_a_table_error_not_a_decoder_traceback(tmp_path, content) -> None:
    path = write_bytes(tmp_path, content)
    with pytest.raises(TableReadError, match="invalid UTF-8 near file line") as caught:
        read_metadata(path)
    assert str(path) in str(caught.value)


@pytest.mark.parametrize("content", [b"source,un\x00it\nS1,001\n", b"source,unit\nS1,00\x001\n", b'source,note\nS1,"line\n\x00next"\n'])
def test_embedded_nul_is_rejected_in_headers_data_and_quoted_multiline_values(tmp_path, content) -> None:
    path = write_bytes(tmp_path, content)
    with pytest.raises(TableReadError, match="embedded NUL character at file line"):
        read_metadata(path)


@pytest.mark.parametrize("name", ["metadata", "metadata.txt", "metadata.csv.gz", "metadata.xlsx"])
def test_unknown_suffix_is_rejected_without_sniffing_or_decompression(tmp_path, name) -> None:
    path = write_bytes(tmp_path, b"source,unit\nS1,001\n", name)
    with pytest.raises(TableReadError, match="cannot infer table format"):
        read_metadata(path)


@pytest.mark.parametrize("format", ["CSV", "excel", "", 1])
def test_invalid_explicit_format_is_rejected(tmp_path, format) -> None:
    path = write_bytes(tmp_path, b"source,unit\nS1,001\n")
    with pytest.raises(TableReadError, match="format must be 'csv' or 'tsv'"):
        read_metadata(path, format=format)


def test_filesystem_errors_propagate_and_reading_never_creates_a_missing_file(tmp_path) -> None:
    path = tmp_path / "missing.csv"
    with pytest.raises(FileNotFoundError):
        read_metadata(path)
    assert not path.exists()
    with pytest.raises(OSError):
        read_metadata(tmp_path, format="csv")


@pytest.mark.parametrize("bom", [b"", b"\xef\xbb\xbf"])
@pytest.mark.parametrize("newline", [b"\n", b"\r\n", b"\r"])
def test_metadata_fingerprint_covers_exact_original_bom_and_line_ending_bytes(tmp_path, bom, newline) -> None:
    content = bom + newline.join([b"source,unit,group", b"NA,001,nan", b"none,002,A", b""])
    path = write_bytes(tmp_path, content, "Original.CSV")
    frame, fingerprint = read_metadata_with_fingerprint(path)
    assert fingerprint == {
        "kind": "file", "name": "Original.CSV", "sha256": hashlib.sha256(content).hexdigest(),
        "size_bytes": len(content), "format": "csv", "encoding": "utf-8-sig", "delimiter": ",",
    }
    assert frame.to_dict("list") == {"source": ["NA", "none"], "unit": ["001", "002"], "group": ["nan", "A"]}
    pd.testing.assert_frame_equal(frame, read_metadata(path))


def test_metadata_fingerprint_records_explicit_tsv_format_without_exposing_absolute_path(tmp_path) -> None:
    content = 'source\tunit\tnote\r\nS1\t001\t"first\r\nsecond\tquoted"\r\n'.encode("utf-8-sig")
    path = write_bytes(tmp_path, content, "observations.unknown")
    frame, fingerprint = read_metadata_with_fingerprint(path, format="tsv")
    assert fingerprint == {
        "kind": "file", "name": path.name, "sha256": hashlib.sha256(content).hexdigest(),
        "size_bytes": len(content), "format": "tsv", "encoding": "utf-8-sig", "delimiter": "\t",
    }
    assert frame.at[0, "note"] == "first\r\nsecond\tquoted"
    assert str(tmp_path) not in repr(fingerprint)


def test_metadata_fingerprint_and_parsing_share_one_stream_even_across_small_byte_chunks(tmp_path, monkeypatch) -> None:
    path = tmp_path / "single-stream.csv"
    content = 'source,unit,note\r\nétude,001,"line one\r\nline two"\r\nS2,002,"a,b"\r\n'.encode("utf-8-sig")
    delivered: list[int] = []

    class ChunkedBytes(BytesIO):
        def readinto(self, buffer):
            count = super().readinto(memoryview(buffer)[:5])
            delivered.append(count)
            return count

        def read(self, *args, **kwargs):
            raise AssertionError("The metadata capture must stream bytes through readinto.")

    stream = ChunkedBytes(content)
    opens: list[tuple] = []
    original_open = Path.open

    def single_open(self, *args, **kwargs):
        if self != path:
            return original_open(self, *args, **kwargs)
        opens.append(args)
        assert len(opens) == 1, "The input must not be reopened to compute its fingerprint."
        assert args[0] == "rb"
        return stream

    monkeypatch.setattr(Path, "open", single_open)
    frame, fingerprint = read_metadata_with_fingerprint(path)
    assert len(opens) == 1
    assert len(delivered) > 2
    assert sum(delivered) == len(content)
    assert stream.closed
    assert fingerprint["sha256"] == hashlib.sha256(content).hexdigest()
    assert fingerprint["size_bytes"] == len(content)
    assert frame.to_dict("list") == {
        "source": ["étude", "S2"], "unit": ["001", "002"], "note": ["line one\r\nline two", "a,b"],
    }


def test_logically_identical_metadata_with_different_bytes_has_different_fingerprints(tmp_path) -> None:
    plain = write_bytes(tmp_path, b"source,unit\nS1,001\n", "plain.csv")
    marked = write_bytes(tmp_path, b"\xef\xbb\xbfsource,unit\r\nS1,001\r\n", "marked.csv")
    left, left_fingerprint = read_metadata_with_fingerprint(plain)
    right, right_fingerprint = read_metadata_with_fingerprint(marked)
    pd.testing.assert_frame_equal(left, right)
    assert left_fingerprint["sha256"] != right_fingerprint["sha256"]
    assert left_fingerprint["size_bytes"] != right_fingerprint["size_bytes"]


@pytest.mark.parametrize("content", [
    b"", b"source,unit\n", b"source,source\nS1,S2\n", b"source,unit\nS1\n",
    b'source,unit\nS1,"unfinished\n', b"source,unit\nS1,\xff\n", b"source,unit\nS1,\x00\n",
])
def test_fingerprinted_reader_preserves_original_table_errors(tmp_path, content) -> None:
    path = write_bytes(tmp_path, content)
    with pytest.raises(TableReadError) as original:
        read_metadata(path)
    with pytest.raises(TableReadError) as captured:
        read_metadata_with_fingerprint(path)
    assert str(captured.value) == str(original.value)


def test_fingerprinted_reader_preserves_filesystem_errors(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        read_metadata_with_fingerprint(tmp_path / "missing.csv")
