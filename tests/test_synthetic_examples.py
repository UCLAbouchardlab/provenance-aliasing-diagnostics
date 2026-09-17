"""Cross-omics acceptance fixtures mapped through the current reference API.

The JSON configuration remains a design draft, so this test maps its explicit
roles to today's constructor rather than claiming the future API is implemented.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from provenance_aliasing import Corpus, balanced_accuracy_ceiling, label_entropy
from provenance_aliasing.structure import structural_leave_one_source_out


EXAMPLES = Path(__file__).resolve().parent / "fixtures" / "synthetic"
EXPECTED = json.loads((EXAMPLES / "expected.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("example", EXPECTED["examples"], ids=lambda value: value["id"])
def test_synthetic_omics_metadata_matches_analytical_expectations(example) -> None:
    config = json.loads((EXAMPLES / example["config_file"]).read_text(encoding="utf-8"))
    frame = pd.read_csv(EXAMPLES / example["data_file"], dtype=str, keep_default_na=False)
    corpus = Corpus.from_frame(
        frame,
        source=config["columns"]["source"],
        unit=config["columns"]["unit"],
        group=config["columns"]["group"],
        weight=config["weighting"].get("column"),
        hierarchy=config["hierarchy"],
        dropna=False,
        duplicates="raise",
    )
    expected = example["design"]
    assert (corpus.n_records, corpus.n_rows, corpus.n_units, corpus.n_sources, corpus.n_groups) == (
        expected["n_records"], expected["n_incidences"], expected["n_units"],
        expected["n_sources"], expected["n_groups"],
    )
    assert corpus.total_weight == expected["total_mass"]
    entropy = label_entropy(corpus)
    values = {
        "entropy.H_G": entropy.H_G,
        "entropy.H_G_given_D": entropy.H_G_given_D,
        "entropy.mutual_information": entropy.mutual_information,
        "entropy.ratio": entropy.ratio,
        "entropy.U_G_given_D": entropy.uncertainty_coefficient,
    }
    for metric, actual in values.items():
        assert actual == pytest.approx(
            example["metrics"][metric]["value"], rel=0,
            abs=EXPECTED["absolute_tolerance"],
        )
    assert entropy.n_spanning_sources == expected["n_spanning_sources_positive_mass"]
    ceiling = example["metrics"]["ceiling.balanced_accuracy"]
    if ceiling["status"] == "ok":
        assert balanced_accuracy_ceiling(corpus).exact == pytest.approx(
            ceiling["value"], rel=0, abs=EXPECTED["absolute_tolerance"],
        )
    else:
        # Today's numerical API raises; the future orchestrator will report the
        # not_applicable status specified in the design fixture.
        with pytest.raises(ValueError, match="two-group"):
            balanced_accuracy_ceiling(corpus)
    structure = structural_leave_one_source_out(corpus)
    assert all(isinstance(keys, tuple) for keys in structure["emptied_unit_keys"])
    assert structure["emptied_total"].sum() == corpus.n_units
