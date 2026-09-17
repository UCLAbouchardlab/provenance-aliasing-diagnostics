"""One declared metadata design, audited and evaluated by the frozen core.

Memory checks estimate additional dense working arrays, including intermediates.
They are not a process-wide memory cap or a runtime guarantee. Individual stages
can be resource-limited while less expensive requested quantities remain useful.
No group regrouping, parent promotion, or source selection is performed here.
"""
from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np
import pandas as pd

from ..ceiling import balanced_accuracy_ceiling
from ..entropy import bootstrap_ratio, label_entropy
from ..guards import cramers_v, design_rank_deficient
from ..incidence import Corpus
from ..structure import pair_sharing_by_group, structural_leave_one_source_out
from .config import AnalysisConfig
from .results import DiagnosticComputationError, DiagnosticResult
from .validation import _assess


_METRICS = {
    "design": ("n_records", "n_retained", "n_excluded", "n_sources", "n_units", "n_groups", "total_weight"),
    "entropy": ("H_G", "H_G_given_D", "mutual_information", "ratio", "U_G_given_D",
                "n_spanning_sources_positive_mass"),
    "ceiling": ("balanced_accuracy", "chance", "q_star", "sensitivity", "specificity"),
    "structure": ("n_spanning_sources", "pair_sharing.n_groups", "source_removal.n_sources"),
    "guards": ("cramers_v", "min_units_per_source", "n_singleton_sources", "design_rank",
               "design_columns", "rank_deficient"),
    "entropy_loso": ("n_sources",),
    "bootstrap": ("ratio_observed", "lower", "upper", "n_effective", "n_degenerate"),
}
_COUNTING = "unweighted metadata incidence; zero-weight records included"
_SKIPPED = object()


def _index(values, name: str) -> pd.Index:
    # Explicit object storage prevents pandas from expanding tuple IDs to axes.
    items = list(values)
    array = np.empty(len(items), dtype=object)
    array[:] = items
    return pd.Index(array, name=name)


class _Runner:
    def __init__(self, assessment) -> None:
        self.validation = assessment.report
        self.config = assessment.report.config
        self.projection = assessment.projection
        self.audit = assessment.audit
        self.mass_label = ("incidence mass: one per retained record" if self.config.weighting.mode == "incidence"
                           else f"row mass from {self.config.weighting.column!r}: {self.config.weighting.description}")
        self.limit = self.config.execution.max_working_memory_mb * 1024**2
        self.rows: dict[str, dict[str, Any]] = {}
        self.tables: dict[str, pd.DataFrame] = {}
        self.execution: list[dict[str, Any]] = []
        self._corpus: Corpus | None = None
        self._probability_corpus: Corpus | None = None
        self.sources: dict[Any, dict[str, Any]] = {}
        self.units: dict[Any, dict[str, Any]] = {}
        self.positive_groups: set[Any] = set()
        self.positive_sources: set[Any] = set()
        for source, unit, group, weight in self.projection[["source", "unit", "group", "weight"]].itertuples(index=False, name=None):
            s = self.sources.setdefault(source, {"units": set(), "groups": set(), "weights": [], "positive_groups": set()})
            s["units"].add(unit)
            s["groups"].add(group)
            s["weights"].append(weight)
            u = self.units.setdefault(unit, {"sources": set(), "group": group, "weights": []})
            u["sources"].add(source)
            u["weights"].append(weight)
            if weight > 0:
                s["positive_groups"].add(group)
                self.positive_groups.add(group)
                self.positive_sources.add(source)
        self.keys = {role: sorted(set(self.projection[role])) for role in ("source", "unit", "group")}
        self.encode = {role: {key: f"{role[0]}{i:012d}" for i, key in enumerate(keys)} for role, keys in self.keys.items()}
        self.decode = {role: {code: key for key, code in mapping.items()} for role, mapping in self.encode.items()}
        self.n, self.s, self.u, self.g = len(self.projection), len(self.sources), len(self.units), len(self.keys["group"])
        # The estimates include several simultaneous dense buffers and row work.
        self.mass_bytes = 128 * self.n + 128 * self.s * self.g
        self.incidence_bytes = 128 * self.n + 128 * self.s * self.u
        self.pair_bytes = self.incidence_bytes + 128 * self.u**2
        columns = self.s + max(self.g - 1, 0)
        self.rank_bytes = self.incidence_bytes + 128 * self.n * columns + 64 * min(self.n, columns)**2
        for family, names in _METRICS.items():
            for name in names:
                self.metric(f"{family}.{name}", None, status="not_requested", reason="This diagnostic was not requested.")

    def weighting(self, metric: str) -> str:
        if metric == "design.total_weight" or metric.split(".")[0] in {"entropy", "ceiling", "entropy_loso", "bootstrap"} or metric == "guards.cramers_v":
            return self.mass_label
        return _COUNTING

    def metric(self, name: str, value: int | float | None, *, status: str = "ok", reason: str = "") -> None:
        if status == "ok" and (value is None or not math.isfinite(value)):
            raise DiagnosticComputationError(name.split(".")[0], f"Unexpected nonfinite result for {name}.", validation=self.validation)
        self.rows[name] = {"metric": name, "family": name.split(".")[0], "value": value,
                           "status": status, "reason": reason, "weighting": self.weighting(name)}

    def unavailable(self, names, status: str, reason: str) -> None:
        for name in names:
            self.metric(name, None, status=status, reason=reason)

    def table(self, name: str, frame: pd.DataFrame, weighting: str, note: str = "") -> None:
        # Do not transfer core display prose, internal identifiers, or attrs.
        frame = frame.copy(deep=True)
        frame.attrs = {"weighting": weighting, "source_grain": self.config.grain.name,
                       "source_description": self.config.grain.description,
                       "unit_description": self.config.unit_description, "note": note}
        self.tables[name] = frame

    def run(self, stage: str, estimated_bytes: int, names, fn: Callable[[], Any]) -> Any:
        if estimated_bytes > self.limit:
            reason = (f"Estimated additional dense working memory ({estimated_bytes} bytes) exceeds "
                      f"the configured budget ({self.limit} bytes).")
            self.unavailable(names, "resource_limited", reason)
            self.execution.append({"stage": stage, "estimated_working_bytes": estimated_bytes,
                                   "budget_bytes": self.limit, "status": "resource_limited", "reason": reason})
            return _SKIPPED
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                result = fn()
        except DiagnosticComputationError:
            raise
        except Exception as exc:
            raise DiagnosticComputationError(stage, str(exc) or type(exc).__name__, validation=self.validation) from exc
        self.execution.append({"stage": stage, "estimated_working_bytes": estimated_bytes,
                               "budget_bytes": self.limit, "status": "ok", "reason": ""})
        return result

    def corpus(self, *, probabilities: bool = False) -> Corpus:
        cached = self._probability_corpus if probabilities else self._corpus
        if cached is not None:
            return cached
        frame = self.projection.copy(deep=True)
        for role in ("source", "unit", "group"):
            frame[role] = [self.encode[role][key] for key in frame[role]]
        if probabilities:
            # Cramer's V and bootstrap ratios are invariant to this scaling.
            # It avoids overflow of products or source-resampling mass totals.
            original = frame["weight"].to_numpy(dtype=float)
            scaled = original / float(self.validation.counts["total_weight"])
            if np.any((original > 0) & (scaled == 0)):
                raise ValueError("The relative weight range cannot be represented during probability scaling.")
            frame["weight"] = scaled
        built = Corpus.from_frame(
            frame, source="source", unit="unit", group="group", weight="weight", name=self.config.name,
            hierarchy={level: f"hierarchy::{level}" for level in self.config.hierarchy},
            dropna=False, duplicates="keep" if self.config.policies.duplicates == "keep_records" else "raise",
        )
        if built.n_rows != self.n or built.n_units != self.u or built.n_sources != self.s or built.n_groups != self.g:
            raise ValueError("Prepared metadata and core construction disagree on retained identifiers.")
        if not probabilities and not math.isclose(built.total_weight, self.validation.counts["total_weight"], rel_tol=1e-12, abs_tol=0):
            raise ValueError("Prepared metadata and core construction disagree on total mass.")
        if probabilities:
            self._probability_corpus = built
        else:
            self._corpus = built
        return built

    def canonical_index(self, frame: pd.DataFrame, role: str) -> pd.DataFrame:
        out = frame.copy(deep=True)
        out.index = _index([self.decode[role][key] for key in frame.index], role)
        return out

    def design(self) -> None:
        for name in _METRICS["design"]:
            self.metric(f"design.{name}", self.validation.counts[name])
        sources = pd.DataFrame([
            {"n_records": len(self.sources[key]["weights"]), "n_units": len(self.sources[key]["units"]),
             "n_groups": len(self.sources[key]["groups"]), "total_weight": math.fsum(self.sources[key]["weights"]),
             "n_positive_mass_groups": len(self.sources[key]["positive_groups"])} for key in self.keys["source"]
        ], index=_index(self.keys["source"], "source"))
        units = pd.DataFrame([
            {"group": self.units[key]["group"], "n_sources": len(self.units[key]["sources"]),
             "n_records": len(self.units[key]["weights"]), "total_weight": math.fsum(self.units[key]["weights"])}
            for key in self.keys["unit"]
        ], index=_index(self.keys["unit"], "unit"))
        note = "Counts include zero-weight records. Only total_weight and positive-mass fields use the declared mass."
        self.table("design.sources", sources, "mixed: unweighted counts and declared row mass", note)
        self.table("design.units", units, "mixed: unweighted counts and declared row mass", note)

    def entropy(self) -> None:
        names = [f"entropy.{name}" for name in _METRICS["entropy"]]
        result = self.run("entropy", self.mass_bytes, names, lambda: label_entropy(self.corpus()))
        if result is _SKIPPED:
            return
        for name, value in (("H_G", result.H_G), ("H_G_given_D", result.H_G_given_D),
                            ("mutual_information", result.mutual_information),
                            ("n_spanning_sources_positive_mass", result.n_spanning_sources)):
            self.metric(f"entropy.{name}", value, reason="Entropy quantities use bits; spanning sources require positive mass in multiple groups.")
        for name, value in (("ratio", result.ratio), ("U_G_given_D", result.uncertainty_coefficient)):
            if result.H_G <= 0:
                self.metric(f"entropy.{name}", None, status="undefined", reason="Marginal group entropy is zero; no positive-mass group contrast exists.")
            else:
                self.metric(f"entropy.{name}", value)
        per_source = self.canonical_index(result.per_source, "source")
        zero_mass = per_source["weight"] == 0
        per_source.loc[zero_mass, "H_G_given_d"] = np.nan
        per_source["H_G_given_d_status"] = np.where(zero_mass, "undefined", "ok")
        per_source["H_G_given_d_reason"] = np.where(zero_mass, "A zero-mass source has no conditional group distribution.", "")
        self.table("entropy.per_source", per_source, self.mass_label,
                   "Conditional entropies and contributions are in bits; spanning requires positive mass in at least two groups.")

    def ceiling(self) -> None:
        names = [f"ceiling.{name}" for name in _METRICS["ceiling"]]
        if self.g != 2 or len(self.positive_groups) != 2:
            self.unavailable(names, "not_applicable", "Requires exactly two retained groups, both with positive total weight; groups are never automatically collapsed or removed.")
            return
        result = self.run("ceiling", self.mass_bytes + 256 * self.s, names,
                          lambda: balanced_accuracy_ceiling(self.corpus()))
        if result is _SKIPPED:
            return
        assumption = "Conditional bound under exact independence of predictions from the declared source; not an empirical model accuracy or a test proving leakage."
        for name, value in (("balanced_accuracy", result.exact), ("chance", result.chance),
                            ("q_star", result.q_star), ("sensitivity", result.sensitivity), ("specificity", result.specificity)):
            self.metric(f"ceiling.{name}", value, reason=assumption)
        detail = pd.DataFrame([{
            "positive_group": self.decode["group"][result.positive_group],
            "negative_group": self.decode["group"][result.negative_group],
            "prior_positive": result.prior_positive, "q_star": result.q_star,
            "balanced_accuracy": result.exact, "sensitivity": result.sensitivity,
            "specificity": result.specificity, "chance": result.chance,
        }])
        self.table("ceiling.contrast", detail, self.mass_label, assumption)

    def structure(self) -> None:
        spanning_keys = [key for key in self.keys["source"] if len(self.sources[key]["groups"]) > 1]
        self.metric("structure.n_spanning_sources", len(spanning_keys))
        spanning = pd.DataFrame([
            {"n_records": len(self.sources[key]["weights"]), "n_units": len(self.sources[key]["units"]),
             "n_groups": len(self.sources[key]["groups"]), "groups": tuple(sorted(self.sources[key]["groups"]))}
            for key in spanning_keys
        ], index=_index(spanning_keys, "source"), columns=["n_records", "n_units", "n_groups", "groups"])
        self.table("structure.spanning_sources", spanning, _COUNTING)
        pair = self.run("structure.pair_sharing", self.pair_bytes, ["structure.pair_sharing.n_groups"],
                        lambda: pair_sharing_by_group(self.corpus()))
        if pair is not _SKIPPED:
            pair = self.canonical_index(pair, "group")
            no_pairs = pair["n_pairs"] == 0
            pair["pair_sharing_status"] = np.where(no_pairs, "undefined", "ok")
            pair["pair_sharing_reason"] = np.where(no_pairs, "Fewer than two distinct units in this group.", "")
            self.metric("structure.pair_sharing.n_groups", len(pair))
            self.table("structure.pair_sharing", pair, _COUNTING)
        removal = self.run("structure.source_removal", self.incidence_bytes + 128 * self.u * self.g + 128 * self.s * self.g,
                           ["structure.source_removal.n_sources"], lambda: structural_leave_one_source_out(self.corpus()))
        if removal is not _SKIPPED:
            overall = removal[["units_covered", "emptied_total"]].copy()
            overall["emptied_unit_keys"] = [tuple(self.decode["unit"][key] for key in values)
                                             for values in removal["emptied_unit_keys"]]
            overall = self.canonical_index(overall, "source")
            self.metric("structure.source_removal.n_sources", len(overall))
            self.table("structure.source_removal", overall, _COUNTING,
                       "A unit is lost when the removed source was its only contributor.")
            rows = []
            for source, record in removal.iterrows():
                for code, group in self.decode["group"].items():
                    rows.append({"source": self.decode["source"][source], "group": group,
                                 "units_covered": int(record[f"covers_{code}"]),
                                 "units_emptied": int(record[f"emptied_{code}"]),
                                 "fraction_emptied": float(record[f"frac_emptied_{code}"]),
                                 "fraction_status": "ok", "fraction_reason": ""})
            self.table("structure.source_removal_by_group", pd.DataFrame(rows), _COUNTING)

    def guards(self) -> None:
        sizes = [len(value["units"]) for value in self.sources.values()]
        self.metric("guards.min_units_per_source", min(sizes))
        self.metric("guards.n_singleton_sources", sum(size == 1 for size in sizes),
                    reason="Singleton sources describe limited replication; they do not by themselves establish provenance aliasing.")
        if len(self.positive_groups) < 2 or len(self.positive_sources) < 2:
            self.metric("guards.cramers_v", None, status="undefined", reason="Requires at least two positive-mass sources and two positive-mass groups.")
        else:
            value = self.run("guards.cramers_v", self.mass_bytes, ["guards.cramers_v"],
                             lambda: cramers_v(self.corpus(probabilities=True), bias_corrected=False))
            if value is not _SKIPPED:
                self.metric("guards.cramers_v", value,
                            reason="Uncorrected symmetric association on declared row mass; no significance test or automatic aliasing threshold.")
        names = ["guards.design_rank", "guards.design_columns", "guards.rank_deficient"]
        if self.g < 2:
            self.unavailable(names, "not_applicable", "The retained metadata has one group, so there is no group contrast in the design.")
            return
        n_columns = self.s + self.g - 1
        result = self.run("guards.design_rank", self.rank_bytes, names,
                          lambda: design_rank_deficient(self.corpus(), max_cells=self.n * n_columns))
        if result is _SKIPPED:
            return
        # Use the numerical rank only. Core's structural shortcut and explanatory
        # prose are not generally valid for multiclass designs.
        rank = result["rank"]
        if not math.isfinite(rank):
            raise DiagnosticComputationError("guards.design_rank", "Core did not return a numerical design rank.", validation=self.validation)
        note = "Numerical rank of full source indicators plus all but one group indicator, using NumPy's default rank tolerance; unweighted."
        self.metric("guards.design_rank", int(rank), reason=note)
        self.metric("guards.design_columns", n_columns, reason=note)
        self.metric("guards.rank_deficient", int(rank < n_columns), reason=note)

    def entropy_loso(self) -> None:
        if self.s < 2:
            self.metric("entropy_loso.n_sources", None, status="not_applicable", reason="Removing the only source leaves no observations.")
            return
        def calculate() -> pd.DataFrame:
            corpus = self.corpus()
            baseline = label_entropy(corpus)
            rows = []
            for source in corpus.sources:
                keep = [key for key in corpus.sources if key != source]
                remaining_mass = math.fsum(
                    math.fsum(self.sources[self.decode["source"][key]]["weights"]) for key in keep
                )
                # The legacy LOSO helper catches ValueError and loses a known
                # baseline contribution on zero-mass removals. Gate that one
                # expected case here and let unexpected numerical errors surface.
                sub = label_entropy(corpus.subset(sources=keep)) if remaining_mass > 0 else None
                ratio = sub.ratio if sub is not None and sub.H_G > 0 else None
                base_ratio = baseline.ratio if baseline.H_G > 0 else None
                delta = ratio - base_ratio if ratio is not None and base_ratio is not None else None
                pct = 100.0 * delta / base_ratio if delta is not None and abs(base_ratio) > 1e-12 else None
                values = {
                    "ratio_without": ratio, "delta_ratio": delta,
                    "abs_delta_ratio": abs(delta) if delta is not None else None,
                    "pct_change": pct, "H_G_without": sub.H_G if sub is not None else None,
                    "H_G_given_D_without": sub.H_G_given_D if sub is not None else None,
                    "original_contribution": float(baseline.per_source.at[source, "contribution"]),
                }
                reasons = {
                    "ratio_without": "Remaining observations have zero mass or zero marginal group entropy.",
                    "delta_ratio": "The baseline or remaining entropy ratio is undefined.",
                    "abs_delta_ratio": "The baseline or remaining entropy ratio is undefined.",
                    "pct_change": "The remaining ratio is undefined, or the baseline ratio is undefined or numerically zero.",
                    "H_G_without": "Removing this source leaves zero total mass.",
                    "H_G_given_D_without": "Removing this source leaves zero total mass.",
                }
                record = {"source": source}
                for column, value in values.items():
                    if value is not None and not math.isfinite(value):
                        raise ValueError(f"Unexpected nonfinite source-removal value for {column}.")
                    record[column] = value
                    record[f"{column}_status"] = "ok" if value is not None else "undefined"
                    record[f"{column}_reason"] = "" if value is not None else reasons[column]
                rows.append(record)
            return pd.DataFrame(rows).set_index("source")

        result = self.run("entropy_loso", self.mass_bytes + 256 * self.n, ["entropy_loso.n_sources"], calculate)
        if result is _SKIPPED:
            return
        result = self.canonical_index(result, "source")
        self.metric("entropy_loso.n_sources", len(result))
        self.table("entropy_loso.sources", result, self.mass_label,
                   "Source-removal influence changes both conditional and marginal label distributions.")

    def bootstrap(self) -> None:
        names = [f"bootstrap.{name}" for name in _METRICS["bootstrap"]]
        if len(self.positive_groups) < 2:
            self.unavailable(names, "undefined", "Marginal group entropy is zero; a normalized entropy-ratio distribution is undefined.")
            return
        b = self.config.diagnostics.bootstrap_replicates
        result = self.run("bootstrap", self.mass_bytes + 64 * b, names,
                          lambda: bootstrap_ratio(self.corpus(probabilities=True), B=b, seed=self.config.diagnostics.seed))
        if result is _SKIPPED:
            return
        note = ("Sources resampled with replacement; 2.5th/97.5th percentiles of finite entropy ratios. "
                "Sensitivity summaries, not automatically calibrated confidence intervals; degenerate replicates are reported.")
        self.metric("bootstrap.ratio_observed", result.observed, reason=note)
        self.metric("bootstrap.n_effective", result.n_effective)
        self.metric("bootstrap.n_degenerate", result.n_degenerate)
        for name, value in (("lower", result.lo), ("upper", result.hi)):
            if result.n_effective:
                self.metric(f"bootstrap.{name}", value, reason=note)
            else:
                self.metric(f"bootstrap.{name}", None, status="undefined", reason="No bootstrap replicate retained a positive-mass group contrast.")
        finite = np.isfinite(result.values)
        values = pd.DataFrame({"replicate": np.arange(b), "ratio": result.values,
                               "status": np.where(finite, "ok", "undefined"),
                               "reason": np.where(finite, "", "Resampled sources have zero mass or zero marginal entropy.")})
        self.table("bootstrap.replicates", values, self.mass_label, note)
        self.tables["bootstrap.replicates"].attrs.update(seed=self.config.diagnostics.seed, resampling_unit="source")

    def finish(self) -> DiagnosticResult:
        self.table("input.observations", self.projection, self.mass_label,
                   "Normalized retained observations; audit.projection_row_position links to this table. Parent columns preserve the declared mapping without changing the analysis unit.")
        for family in ("design", "entropy", "ceiling", "structure", "guards", "entropy_loso"):
            if family in self.config.diagnostics.include:
                try:
                    getattr(self, family)()
                except DiagnosticComputationError:
                    raise
                except Exception as exc:
                    raise DiagnosticComputationError(family, str(exc) or type(exc).__name__, validation=self.validation) from exc
        if self.config.diagnostics.bootstrap_replicates > 0:
            try:
                self.bootstrap()
            except DiagnosticComputationError:
                raise
            except Exception as exc:
                raise DiagnosticComputationError("bootstrap", str(exc) or type(exc).__name__, validation=self.validation) from exc
        execution = pd.DataFrame(self.execution, columns=["stage", "estimated_working_bytes", "budget_bytes", "status", "reason"])
        self.table("execution.stages", execution, "not a scientific quantity",
                   "Conservative estimates of additional numerical working arrays; metadata, result snapshots, runtime, and total process memory are not hard-capped.")
        metrics = pd.DataFrame(self.rows.values(), columns=["metric", "family", "value", "status", "reason", "weighting"])
        return DiagnosticResult(validation=self.validation, metrics=metrics, tables=self.tables, audit=self.audit)


def diagnose(frame: pd.DataFrame, *, config: AnalysisConfig) -> DiagnosticResult:
    """Validate metadata, prepare its declared projection, and run selected diagnostics.

    Invalid metadata raises MetadataValidationError with the validation report.
    Unexpected numerical failures raise DiagnosticComputationError with the stage
    and original exception. Expected undefined, inapplicable, unrequested, and
    memory-limited quantities receive explicit statuses instead of numeric zeros.

    Bootstrap is enabled by a positive bootstrap_replicates value independently
    of include; otherwise it is not requested. One source grain and one analysis
    unit are evaluated per call. The supplied DataFrame is never modified.
    """
    assessment = _assess(frame, config=config)
    assessment.report.raise_for_errors()
    return _Runner(assessment).finish()
