from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from provenance_aliasing.adapters import (
    check_recoverability,
    column_profile,
    from_csv,
    from_excel,
    from_filenames,
    from_incidence_table,
    from_long,
    from_sdrf,
    infer_columns,
    looks_like_prose,
    numeric_audit,
    parse_run_names,
    read_table,
    sniff_separator,
)


def test_read_table_handles_csv_tsv_and_misleading_extension(tmp_path: Path) -> None:
    comma = tmp_path / "comma.csv"
    comma.write_text("source,unit,group\nS1,U1,A\n", encoding="utf-8")
    tab = tmp_path / "tab.tsv"
    tab.write_text("source\tunit\tgroup\nS1\tU1\tA\n", encoding="utf-8")
    misleading = tmp_path / "actually-tab.csv"
    misleading.write_text(tab.read_text(encoding="utf-8"), encoding="utf-8")

    assert sniff_separator(comma) == ","
    assert sniff_separator(tab) == "\t"
    assert list(read_table(misleading).columns) == ["source", "unit", "group"]
    assert read_table(comma).iloc[0]["unit"] == "U1"


def test_read_table_preserves_identifier_text(tmp_path: Path) -> None:
    path = tmp_path / "ids.csv"
    path.write_text("source,unit,group\n007,001,A\n", encoding="utf-8")
    frame = read_table(path)
    assert frame.loc[0, "source"] == "007"
    assert frame.loc[0, "unit"] == "001"


def test_csv_excel_and_long_adapters_build_equivalent_corpora(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {"deposit": ["S1", "S1", "S2"], "sample": ["U1", "U2", "U3"], "condition": ["A", "B", "A"]}
    )
    csv_path = tmp_path / "table.csv"
    xlsx_path = tmp_path / "table.xlsx"
    frame.to_csv(csv_path, index=False)
    frame.to_excel(xlsx_path, index=False)
    kwargs = dict(source="deposit", unit="sample", group="condition")
    corpora = [from_csv(csv_path, **kwargs), from_excel(xlsx_path, **kwargs), from_long(frame, **kwargs)]
    for corpus in corpora[1:]:
        pd.testing.assert_frame_equal(corpus.frame, corpora[0].frame)


def test_filename_parser_and_adapter_contract() -> None:
    names = ["PXD001_D01_case.raw", "PXD002_D02_control.raw"]
    schema = {"source": 0, "unit": 1, "group": 2}
    parsed = parse_run_names(names, schema, expected_tokens=3)
    assert list(parsed["source"]) == ["PXD001", "PXD002"]
    corpus = from_filenames(
        names,
        schema,
        source_field="source",
        unit_field="unit",
        group_field="group",
        expected_tokens=3,
    )
    assert corpus.n_rows == 2
    assert corpus.groups == ["case", "control"]


def test_distinct_hla_filenames_reach_biological_duplicate_policy() -> None:
    names = [
        "PXD001_AUT-DN06-Lung_normal_HLA-I.raw",
        "PXD001_AUT-DN06-Lung_normal_HLA-II.raw",
    ]
    schema = {"dataset": 0, "sample": 1, "group": 2, "hla_class": 3}
    corpus = from_filenames(
        names,
        schema,
        source_field="dataset",
        unit_field=["dataset", "sample"],
        group_field="group",
        expected_tokens=4,
        duplicates="drop",
    )

    assert corpus.n_rows == 1
    assert corpus.n_records == 2
    assert corpus.records["hla_class"].tolist() == ["HLA-I", "HLA-II"]


def test_filename_parser_rejects_wrong_tokens_duplicates_and_bad_schema() -> None:
    schema = {"source": 0, "unit": 1, "group": 2}
    with pytest.raises(ValueError, match="token"):
        parse_run_names(["too_short.raw"], schema, expected_tokens=3)
    with pytest.raises(ValueError, match="duplicate"):
        parse_run_names(["S_U_A.raw", "S_U_A.raw"], schema)
    with pytest.raises((KeyError, ValueError)):
        parse_run_names(["S_U_A.raw"], {"source": 9})


def test_sdrf_accepts_dataframe_and_file(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {"comment[data file]": ["run1", "run2"], "source name": ["D1", "D2"], "characteristics[group]": ["A", "B"]}
    )
    kwargs = dict(
        source_col="source name",
        unit_col="comment[data file]",
        group_col="characteristics[group]",
    )
    from_frame = from_sdrf(frame, **kwargs)
    path = tmp_path / "study.sdrf.tsv"
    frame.to_csv(path, sep="\t", index=False)
    from_path = from_sdrf(path, **kwargs)
    pd.testing.assert_frame_equal(from_frame.frame, from_path.frame)


def test_numeric_audit_and_prose_detection() -> None:
    prose = pd.Series(["identified 1,200 peptides in 300 runs", "found 900 peptides in 400 runs"])
    audit = numeric_audit(prose)
    assert audit["n_direct_numeric"] == 0
    assert audit["n_multi_number"] == 2
    assert looks_like_prose(prose)
    assert not looks_like_prose(pd.Series([100, 200, 300]))


def test_incidence_adapter_duplicate_policy_and_prose_weight() -> None:
    frame = pd.DataFrame(
        {
            "source": ["S1", "S1"],
            "unit": ["U1", "U1"],
            "group": ["A", "A"],
            "mass": ["found 100 peptides in 300 runs"] * 2,
        }
    )
    with pytest.raises(ValueError, match="repeated"):
        from_incidence_table(frame, source="source", unit="unit", group="group")
    dropped = from_incidence_table(frame, source="source", unit="unit", group="group", duplicates="drop")
    assert dropped.n_rows == 1
    assert len(dropped.records) == 2
    with pytest.raises(ValueError, match="free text"):
        from_incidence_table(
            frame.drop_duplicates(["source", "unit"]),
            source="source",
            unit="unit",
            group="group",
            weight="mass",
        )


def test_long_adapter_forwards_composite_unit_duplicate_and_hierarchy_policy() -> None:
    frame = pd.DataFrame(
        {
            "dataset": ["PXD1", "PXD1"],
            "sample_id": ["AUT-DN06_Lung", "AUT-DN06_Lung"],
            "donor_id": ["AUT-DN06", "AUT-DN06"],
            "condition": ["normal", "normal"],
            "hla_class": ["HLA-I", "HLA-II"],
        }
    )
    corpus = from_long(
        frame,
        source="dataset",
        unit=["dataset", "sample_id"],
        group="condition",
        duplicates="drop",
        hierarchy={"donor": ["dataset", "donor_id"]},
    )

    assert corpus.n_rows == 1
    assert len(corpus.records) == 2
    assert corpus.frame.loc[0, "unit"] == ("PXD1", "AUT-DN06_Lung")
    assert "donor" in corpus.hierarchy_report().index


def test_recoverability_accepts_composite_unit_keys() -> None:
    frame = pd.DataFrame(
        {
            "dataset": ["PXD1", "PXD2"],
            "sample_id": ["sample-1", "sample-1"],
            "condition": ["normal", "tumor"],
        }
    )
    audit = check_recoverability(
        frame,
        source="dataset",
        unit=["dataset", "sample_id"],
        group="condition",
    )

    assert audit.loc["required_columns_present", "status"] == "ok"
    assert audit.loc["n_units", "value"] == 2


def test_recoverability_never_raises_for_missing_required_column() -> None:
    frame = pd.DataFrame({"source": ["S1"], "unit": ["U1"]})
    audit = check_recoverability(frame, source="source", unit="unit", group="missing")
    assert audit.loc["corpus_constructible", "status"] == "blocked"
    assert audit.loc["required_columns_present", "status"] == "blocked"


def test_column_profile_and_inference_show_evidence(worked_frame: pd.DataFrame) -> None:
    profile = column_profile(worked_frame)
    assert set(worked_frame.columns) == set(profile.index)
    assert {"n_unique", "frac_numeric", "looks_like_prose"} <= set(profile.columns)
    inferred = infer_columns(worked_frame)
    assert {"source", "unit", "group", "weight"} == set(inferred)
    assert "source" in inferred["source"]
