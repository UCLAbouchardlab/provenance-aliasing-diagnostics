from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from provenance_aliasing.incidence import Corpus


ANCHORS = {
    "H_G": 0.9666186325481028,
    "H_G_given_D": 0.08041532740670795,
    "ratio": 0.08319240360050288,
    "n_rows": 84,
    "n_sources": 43,
    "n_units": 44,
    "n_spanning": 3,
}

HUB = "SRC-HUB"
NORMAL_UNITS = [f"N{i:02d}" for i in range(1, 31)]
CANCER_UNITS = [f"C{i:02d}" for i in range(1, 15)]


def worked_records() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    def add(source: str, unit: str, group: str) -> None:
        rows.append({"source": source, "unit": unit, "group": group})

    for unit in NORMAL_UNITS:
        add(HUB, unit, "Normal")

    for source, triples in {
        "SRC-SPAN-A": [("N01", "Normal"), ("C01", "Cancer")],
        "SRC-SPAN-B": [("N02", "Normal"), ("C02", "Cancer"), ("C03", "Cancer")],
        "SRC-SPAN-C": [("N03", "Normal"), ("C04", "Cancer")],
    }.items():
        for unit, group in triples:
            add(source, unit, group)

    for source, units in {
        "SRC-M5": ("C05", "C06", "C07", "C08", "C09"),
        "SRC-M3": ("C10", "C11", "C12"),
        "SRC-M2a": ("C03", "C13"),
        "SRC-M2b": ("C01", "C02"),
    }.items():
        for unit in units:
            add(source, unit, "Cancer")

    singleton_load = {
        "C03": 8, "C05": 5, "C06": 4, "C02": 3, "C07": 3,
        "C08": 2, "C09": 2, "C10": 2, "C11": 2,
        "C04": 1, "C12": 1, "C13": 1, "C14": 1,
    }
    k = 0
    for unit, count in singleton_load.items():
        for _ in range(count):
            add(f"SRC-{k:02d}", unit, "Cancer")
            k += 1

    assert k == 35
    assert len(rows) == ANCHORS["n_rows"]
    return rows


@pytest.fixture
def worked_frame() -> pd.DataFrame:
    frame = pd.DataFrame(worked_records())
    frame["weight"] = np.where(frame["source"] == HUB, 100.0, 1.0)
    frame["laboratory"] = np.where(frame["source"] == HUB, "LAB-HUB", "LAB-OTHER")
    return frame


@pytest.fixture
def worked_corpus(worked_frame: pd.DataFrame) -> Corpus:
    return Corpus.from_frame(worked_frame.copy(), name="worked")


@pytest.fixture
def weighted_corpus(worked_frame: pd.DataFrame) -> Corpus:
    return Corpus.from_frame(worked_frame.copy(), weight="weight", name="weighted")


@pytest.fixture
def crossed_corpus() -> Corpus:
    rows = [
        {"source": f"S{i}", "unit": f"U{i}-{group}", "group": group}
        for i in range(4)
        for group in ("A", "B")
    ]
    return Corpus.from_records(rows, name="crossed")


@pytest.fixture
def nested_corpus() -> Corpus:
    rows = [
        {"source": f"S-{group}-{i}", "unit": f"U-{group}-{i}", "group": group}
        for group in ("A", "B")
        for i in range(4)
    ]
    return Corpus.from_records(rows, name="nested")


@pytest.fixture
def profiles() -> pd.DataFrame:
    return pd.DataFrame(
        [[0.70, 0.20, 0.10], [0.60, 0.25, 0.15], [0.10, 0.20, 0.70], [0.15, 0.25, 0.60]],
        index=["u1", "u2", "u3", "u4"],
        columns=["a", "b", "c"],
    )


@pytest.fixture
def similarity() -> np.ndarray:
    return np.array(
        [
            [1.0, 0.8, 0.2, 0.1],
            [0.8, 1.0, 0.3, 0.2],
            [0.2, 0.3, 1.0, 0.6],
            [0.1, 0.2, 0.6, 1.0],
        ]
    )
