"""
An integrated report of the declared provenance design.

What can the metadata say about source-group aliasing before a feature model is
fit? ``diagnose`` applies the incidence diagnostics to a :class:`Corpus` and
collects their results in :class:`Diagnosis`. It opens no data files and requires
no knowledge of whether the analysis units contain peptides, metabolites,
transcripts, or cells.

Workflow:
---------
1. Construct a Corpus with explicit source, unit, group, and weighting definitions.
2. Call ``diagnose(corpus, grain=...)`` with a readable description of source grain.
3. Inspect the entropy result, crossing, structural support, ceiling, guards,
and any notes about stages that could not be computed.
4. Render the result with ``to_frame``, ``to_markdown``, ``summary``, or ``verdict``.

Weighting and scope:
--------------------
Entropy and the binary ceiling use the corpus's row mass. Structural pair counts
and removal summaries use binary incidence and receive a separate weighting label
in the report. The grain argument describes the existing source column; it does
not regroup the observations or change the calculation.

Interpretation:
---------------
The ceiling concerns exact provenance invariance under the declared design.
The verdict's descriptive bands are reporting conventions, not hypothesis tests
or validated biological decision thresholds. Neither a favorable ratio nor a
high ceiling establishes that a measured biological contrast is valid.

Missing optional stages are retained as notes and undefined quantities. With
``strict=True``, errors raised while evaluating those stages propagate; missing
functions still receive availability notes. The central entropy calculation is
required and its errors always propagate.

Contents:
---------
Diagnosis - component results, provenance context, and reporting methods.
diagnose - run the metadata diagnostics for one corpus.
"""

from __future__ import annotations

import importlib
import inspect
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd

from .entropy import BootstrapResult, EntropyResult, bootstrap_ratio, label_entropy
from .incidence import Corpus

UNDECLARED_GRAIN = "undeclared (the source column exactly as supplied)"
"""
Label used when the caller has not described the source grain.

The calculation still uses the source column exactly as supplied. Declare its
meaning before quoting a ratio, because deposit, laboratory, and run keys can
partition the same observations differently.
"""

STRUCTURAL_WEIGHTING = "incidence (row weights not used)"
"""
Reporting label for binary incidence counts and structural source-removal quantities.

These reported quantities do not use row mass. Record counts can still depend
on duplicate retention even though numerical row weights do not enter them.
"""

# Reporting conventions, not hypothesis tests. They exist so `verdict()` can pick a
# word; the boundaries are arbitrary and every caller should quote the number too.
CEILING_CHANCE_BAND = 0.55
"""
Upper ceiling value, 0.55, described as close to chance in a binary report.

This is a descriptive band, not a statistical test of equality to chance.
Exact chance is checked separately against ``Diagnosis.chance_level``.
"""

CEILING_USABLE = 0.70
"""
Ceiling threshold, 0.70, above which the verdict describes greater available headroom.

The threshold controls prose only; it does not guarantee useful prediction
or establish biological validity.
"""

RATIO_ALIASED = 0.20
"""
Fallback R threshold, 0.20, for the strongest aliasing wording when no ceiling is finite.

This conventional band is not a proven estimability boundary.
"""

RATIO_CROSSED = 0.60
"""
Fallback R threshold, 0.60, separating the remaining ratio-based verdict bands.

Used only when a finite ceiling is unavailable. The number and its declared
grain and weighting remain necessary for interpretation.
"""

CROSSED_STRATUM_RATIO = 0.50
"""
Crossed-subcorpus R threshold, 0.50, used for the more favorable crossed-stratum wording.

A high ratio inside a small crossed subset does not establish support in the
full corpus or sufficient independent replication within the subset.
"""

_SEARCH_PATH: tuple[str, ...] = (
    "ceiling",
    "structure",
    "guards",
    "grain",
    "calibration",
    "metrics",
    "design",
)


def _resolve(name: str, preferred: str | None = None) -> Callable[..., Any] | None:
    """
    Find a diagnostic function in the installed core modules.

    Parameters:
    ----------
    name : str - attribute name to find.
    preferred : str or None, default None - module searched before the common path.

    Returns:
    -------
    callable or None - the first matching attribute, or None when unavailable.

    Notes:
    ------
    ImportErrors are treated as unavailable modules. Other import-time exceptions
    are not suppressed here. The function resolves names without executing diagnostics.
    """
    tried = (preferred,) + _SEARCH_PATH if preferred else _SEARCH_PATH
    seen: set[str] = set()
    for mod in tried:
        if mod in seen:
            continue
        seen.add(mod)
        try:
            module = importlib.import_module(f".{mod}", __package__)
        except ImportError:
            continue
        fn = getattr(module, name, None)
        if fn is not None:
            return fn
    return None


def _call(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """
    Invoke a function with only the keyword arguments accepted by its signature.

    Functions with arbitrary keyword arguments receive all supplied keywords.
    Uninspectable signatures also receive all keywords, allowing the callable to
    validate them. Positional arguments are always forwarded unchanged.
    """
    if not kwargs:
        return fn(*args)
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):  # builtins and C callables have no signature
        return fn(*args, **kwargs)
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return fn(*args, **kwargs)
    return fn(*args, **{k: v for k, v in kwargs.items() if k in params})

def _is_scalar(v: Any) -> bool:
    """
    Recognize the scalar types retained in a report mapping, including None and NumPy scalars.
    """
    return v is None or isinstance(
        v, (str, bool, int, float, np.integer, np.floating, np.bool_)
    )


def _as_float(v: Any) -> float:
    """
    Convert a report value to float, using ``nan`` for None, strings, or failed conversion.
    """
    if v is None or isinstance(v, str):
        return float("nan")
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _as_mapping(obj: Any) -> dict[str, Any]:
    """
    Extract scalar fields from a mapping, as_dict result, dataclass, or object.

    Returns an empty dictionary when no supported representation is available.
    Tables, arrays, and other non-scalar fields are omitted; this is reporting
    adaptation rather than lossless serialization.
    """
    if obj is None:
        return {}
    if isinstance(obj, Mapping):
        return {str(k): v for k, v in obj.items() if _is_scalar(v)}
    as_dict = getattr(obj, "as_dict", None)
    if callable(as_dict):
        try:
            return {str(k): v for k, v in dict(as_dict()).items() if _is_scalar(v)}
        except Exception:  # pragma: no cover - defensive, a broken as_dict is not fatal
            pass
    if is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: getattr(obj, f.name) for f in fields(obj) if _is_scalar(getattr(obj, f.name))}
    if hasattr(obj, "__dict__"):
        return {k: v for k, v in vars(obj).items() if not k.startswith("_") and _is_scalar(v)}
    return {}


def _as_frame(obj: Any) -> pd.DataFrame | None:
    """
    Obtain a DataFrame from a table, Series, or object's to_frame method.

    Returns None when the object has no usable tabular view or its conversion
    raises. Existing DataFrames are returned directly, without a copy.
    """
    if obj is None:
        return None
    if isinstance(obj, pd.DataFrame):
        return obj
    if isinstance(obj, pd.Series):
        return obj.to_frame()
    to_frame = getattr(obj, "to_frame", None)
    if callable(to_frame):
        try:
            out = to_frame()
        except Exception:  # pragma: no cover - defensive
            return None
        if isinstance(out, pd.DataFrame):
            return out
    return None


def _pick(mapping: Mapping[str, Any], keys: Sequence[str]) -> Any:
    """
    Return the value for the first candidate key present in a mapping, or None.
    """
    for k in keys:
        if k in mapping:
            return mapping[k]
    return None


def _pick_column(frame: pd.DataFrame, keys: Sequence[str]) -> str | None:
    """
    Find the first candidate column by case-insensitive string label, or return None.
    """
    lower = {str(c).lower(): c for c in frame.columns}
    for k in keys:
        if k in lower:
            return lower[k]
    return None


_CEILING_VALUE_KEYS = (
    "ceiling",
    "balanced_accuracy_ceiling",
    "balanced_accuracy",
    "exact",
    "value",
    "bound",
)


def _row(quantity: str, value: Any, weighting: str, grain: str, note: str = "") -> dict:
    """
    Construct one quantity/value reporting row with weighting, grain, and note.
    """
    return {
        "quantity": quantity,
        "value": value,
        "weighting": weighting,
        "grain": grain,
        "note": note,
    }


def _numeric_columns(frame: pd.DataFrame, preferred: Sequence[str]) -> list[str]:
    """
    Select numeric or Boolean columns, preferring exact names or prefixes when matched.

    Falls back to all numeric/Boolean columns when none match the preferred names.
    """
    numeric = [
        c for c in frame.columns
        if pd.api.types.is_numeric_dtype(frame[c]) or pd.api.types.is_bool_dtype(frame[c])
    ]
    keep = [c for c in numeric if any(str(c) == p or str(c).startswith(p) for p in preferred)]
    return keep or numeric


_PAIR_SHARING_COLUMNS = ("n_units", "n_pairs", "n_pairs_sharing", "pair_sharing_rate")
_STRUCTURAL_COLUMNS = ("units_covered", "emptied", "frac_emptied")


def _frame_rows(
    prefix: str,
    frame: pd.DataFrame,
    *,
    weighting: str,
    grain: str,
    note: str = "",
    preferred: Sequence[str] = (),
) -> list[dict]:
    """
    Flatten selected numeric table cells into named report quantities.

    Each row label and column name becomes ``prefix[index].column``. The supplied
    weighting, grain, and note accompany every emitted value.
    """
    columns = _numeric_columns(frame, preferred)
    return [
        _row(f"{prefix}[{idx}].{col}", _as_float(frame.at[idx, col]), weighting, grain, note)
        for idx in frame.index
        for col in columns
    ]


def _extremum_rows(
    prefix: str,
    frame: pd.DataFrame,
    *,
    weighting: str,
    grain: str,
    preferred: Sequence[str] = (),
) -> list[dict]:
    """
    Summarize selected numeric columns by their maxima and the attaining row labels.

    Missing values are dropped after numeric conversion. A column without remaining
    values emits an undefined maximum; otherwise the note identifies the first
    maximum and the number of available rows.
    """
    rows: list[dict] = []
    for col in _numeric_columns(frame, preferred):
        s = pd.to_numeric(frame[col], errors="coerce").dropna()
        if s.empty:
            rows.append(_row(f"{prefix}.max_{col}", float("nan"), weighting, grain, "no finite values"))
            continue
        rows.append(
            _row(
                f"{prefix}.max_{col}",
                float(s.max()),
                weighting,
                grain,
                f"maximum over {len(s)} sources, attained at {s.idxmax()!r}",
            )
        )
    return rows


_GUARD_VALUE_KEYS = ("value", "statistic", "score", "fired", "fires", "triggered", "flag")
_GUARD_NOTE_KEYS = ("what_it_means", "note", "message", "detail", "description", "interpretation")


def _guard_rows(frame: pd.DataFrame, *, weighting: str, grain: str) -> list[dict]:
    """
    Translate a guard table into the report's common quantity schema.

    Recognizes common names for the guard statistic, firing state, threshold, and
    interpretation. Row-specific grain and weighting labels take precedence.
    Undefined firing states remain explicit in the note rather than becoming False.
    """
    value_col = _pick_column(frame, _GUARD_VALUE_KEYS)
    note_col = _pick_column(frame, _GUARD_NOTE_KEYS)
    name_col = _pick_column(frame, ("name", "guard", "check", "test"))
    fires_col = _pick_column(frame, ("fires", "fired", "triggered"))
    thresh_col = _pick_column(frame, ("threshold", "cutoff", "at"))
    w_col = _pick_column(frame, ("weighting",))
    g_col = _pick_column(frame, ("grain",))

    rows: list[dict] = []
    for idx, record in frame.iterrows():
        label = str(record[name_col]) if name_col else str(idx)
        bits: list[str] = []
        if fires_col is not None:
            fired = record[fires_col]
            state = "undefined" if pd.isna(fired) else ("FIRES" if bool(fired) else "does not fire")
            if thresh_col is not None and not pd.isna(record[thresh_col]):
                state += f" at {_fmt(record[thresh_col])}"
            bits.append(state)
        if note_col is not None and isinstance(record[note_col], str) and record[note_col]:
            bits.append(record[note_col])
        note = "; ".join(bits)
        row_w = str(record[w_col]) if w_col is not None and isinstance(record[w_col], str) else weighting
        row_g = str(record[g_col]) if g_col is not None and isinstance(record[g_col], str) else grain

        if value_col is not None:
            rows.append(_row(f"guard.{label}", _as_float(record[value_col]), row_w, row_g, note))
        else:  # unknown schema: emit every numeric field the guard reported
            rows.extend(
                _row(f"guard.{label}.{c}", _as_float(v), row_w, row_g, note)
                for c, v in record.items()
                if _is_scalar(v) and not isinstance(v, str)
            )
    return rows


def _fmt(v: Any, digits: int = 4) -> str:
    """
    Format report scalars using significant digits and ``n/a`` for unavailable numbers.

    Boolean values use lowercase words, integers retain integer formatting, and
    other objects use their string representation.
    """
    if v is None:
        return "n/a"
    if isinstance(v, (bool, np.bool_)):
        return "true" if bool(v) else "false"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        x = float(v)
        if not np.isfinite(x):
            return "n/a"
        if x == int(x) and abs(x) < 1e6:
            return str(int(x))
        return f"{x:.{digits}g}"
    return str(v)


def _count(n: int, singular: str, plural: str | None = None) -> str:
    """
    Format a count with its singular or plural noun for report prose.
    """
    return f"{n} {singular}" if n == 1 else f"{n} {plural or singular + 's'}"


def _md_cell(v: Any) -> str:
    """
    Format one Markdown table cell, escaping pipes and removing embedded line breaks.
    """
    return _fmt(v).replace("|", r"\|").replace("\n", " ").replace("\r", " ")


def _md_table(frame: pd.DataFrame) -> str:
    """
    Render a DataFrame as a Markdown table without exporting its index.

    Uses local formatting helpers, so no external table-rendering dependency or
    file-writing operation is required.
    """
    head = "| " + " | ".join(str(c) for c in frame.columns) + " |"
    rule = "| " + " | ".join("---" for _ in frame.columns) + " |"
    body = [
        "| " + " | ".join(_md_cell(v) for v in record) + " |"
        for record in frame.itertuples(index=False, name=None)
    ]
    return "\n".join([head, rule, *body])


@dataclass(frozen=True)
class Diagnosis:
    """
    Collected metadata diagnostics for one corpus, source grain, and weighting.

    Returned by :func:`diagnose`. The object keeps component results available
    for inspection while providing a common table and narrative rendering.

    Attributes:
    -----------
    corpus, weighting, grain : str - corpus name, analytical mass definition,
    and the caller's description of source grain.
    design : pd.Series - incidence and provenance-handling counts from Corpus.describe.
    entropy : EntropyResult - required full-corpus entropy calculation.
    n_groups : int, default 2 - group count used to report chance as ``1/n_groups``.
    spanning : list - canonical keys of sources represented in multiple groups.
    crossed_entropy : EntropyResult or None - entropy restricted to spanning sources.
    crossed_design : pd.Series or None - design counts for that restricted corpus.
    pair_sharing : pd.DataFrame or None - within-group shared-source pair counts.
    structural_loso : pd.DataFrame or None - units losing support under source removal.
    spanning_table : pd.DataFrame or None - details of structurally spanning sources.
    ceiling : object or None - original ceiling result, retained for inspection.
    ceiling_value : float - identified exact balanced-accuracy ceiling, otherwise ``nan``.
    ceiling_weighting : str - weighting reported by the ceiling result, when present.
    ceiling_detail : dict - scalar fields extracted from the ceiling result.
    guards : pd.DataFrame or None - values, thresholds, and guard interpretations.
    bootstrap : BootstrapResult or None - optional source-resampling sensitivity.
    notes : list of str - missing grain and unavailable or failed diagnostic stages.

    Notes:
    ------
    None may mean a stage was inapplicable, unavailable, or failed; inspect notes
    before interpreting an absent result. ``ratio`` and ``crossed_ratio`` expose
    the full and restricted entropy ratios. The frozen dataclass does not make
    its tables, lists, or dictionaries immutable.
    """

    corpus: str
    weighting: str
    grain: str
    design: pd.Series = field(repr=False)
    entropy: EntropyResult = field(repr=False)
    n_groups: int = 2
    spanning: list[str] = field(default_factory=list, repr=False)
    crossed_entropy: EntropyResult | None = field(default=None, repr=False)
    crossed_design: pd.Series | None = field(default=None, repr=False)
    pair_sharing: pd.DataFrame | None = field(default=None, repr=False)
    structural_loso: pd.DataFrame | None = field(default=None, repr=False)
    spanning_table: pd.DataFrame | None = field(default=None, repr=False)
    ceiling: Any | None = field(default=None, repr=False)
    ceiling_value: float = float("nan")
    ceiling_weighting: str = ""
    ceiling_detail: dict[str, Any] = field(default_factory=dict, repr=False)
    guards: pd.DataFrame | None = field(default=None, repr=False)
    bootstrap: BootstrapResult | None = field(default=None, repr=False)
    notes: list[str] = field(default_factory=list, repr=False)

    @property
    def ratio(self) -> float:
        """
        Full-corpus ``H(G|D)/H(G)`` under the stored weighting and source grain.
        """
        return self.entropy.ratio

    @property
    def crossed_ratio(self) -> float:
        """
        Residual-entropy ratio within structurally spanning sources, or ``nan`` when unavailable.
        """
        return self.crossed_entropy.ratio if self.crossed_entropy is not None else float("nan")

    @property
    def chance_level(self) -> float:
        """
        Balanced-accuracy chance level ``1/n_groups``, or ``nan`` when no groups are recorded.

        This reporting property does not extend the binary ceiling calculation to
        multigroup designs.
        """
        return 1.0 / self.n_groups if self.n_groups else float("nan")

    def to_frame(self) -> pd.DataFrame:
        """
        Render the diagnostic as one row per reportable quantity.

        Returns:
        -------
        pd.DataFrame - columns ``quantity``, ``value``, ``weighting``, ``grain``,
        and ``note``. Includes selected design counts, entropies and source counts,
        crossed-subcorpus summaries, pair sharing, maxima of structural removal fields,
        ceiling details, guards, and optional bootstrap summaries.

        Notes:
        ------
        Structural quantities receive a binary-incidence weighting label, while
        entropy and ceiling values retain their mass definition. Unavailable stages
        emit undefined values with explanatory notes. Structural removal is summarized
        by maxima here; inspect ``structural_loso`` for the full source-level table.
        The method returns an in-memory table and writes no files.
        """
        w, g, s = self.weighting, self.grain, STRUCTURAL_WEIGHTING
        rows: list[dict] = []
        d = self.design
        rows += [
            _row("design.n_rows", _as_float(d.get("n_rows")), s, g, "incidence rows after cleaning"),
            _row("design.n_sources", _as_float(d.get("n_sources")), s, g, "distinct sources at this grain"),
            _row("design.n_units", _as_float(d.get("n_units")), s, g, ""),
            _row("design.n_groups", _as_float(d.get("n_groups")), s, g, ""),
            _row("design.units_per_source_max", _as_float(d.get("units_per_source_max")), s, g,
                 "the hub, if there is one"),
            _row("design.sources_per_unit_max", _as_float(d.get("sources_per_unit_max")), s, g, ""),
            _row("design.units_with_one_source", _as_float(d.get("units_with_one_source")), s, g,
                 "for these units the unit effect and the source effect are the same effect"),
            _row("design.sources_spanning_groups", float(len(self.spanning)), s, g,
                 "sources contributing to more than one group; only these can contribute to H(G|D)"),
        ]
        e = self.entropy
        rows += [
            _row("entropy.H_G", e.H_G, w, g, "bits"),
            _row("entropy.H_G_given_D", e.H_G_given_D, w, g, "bits"),
            _row("entropy.mutual_information", e.mutual_information, w, g, "I(G;D) = H(G) - H(G|D), bits"),
            _row("entropy.ratio", e.ratio, w, g,
                 "H(G|D)/H(G); near 0 the source determines the label, near 1 it says nothing"),
            _row("entropy.U_G_given_D", e.uncertainty_coefficient, w, g,
                 "fraction of the label's entropy explained by provenance; NOT an accuracy "
                 "and not a proportion of cases"),
            _row("entropy.n_spanning_sources",float(e.n_spanning_sources),w,g,"positive-weight sources contributing to multiple biological groups",),
            _row("entropy.n_nested_sources",float(e.n_nested_sources),w,g, "positive-weight sources restricted to one biological group; "
        "each contributes zero conditional label entropy",),
            _row("entropy.n_zero_weight_sources",float(e.n_zero_weight_sources),w,g,"sources carrying no weight",),
            ]
        if not e.per_source.empty:
            top = e.per_source["contribution"].astype(float)
            rows.append(
                _row("entropy.max_source_contribution", float(top.max()), w, g,
                     f"largest single-source term in H(G|D), from {top.idxmax()!r}")
            )
        if self.crossed_entropy is not None:
            c, cd = self.crossed_entropy, self.crossed_design
            rows += [
                _row("crossed.ratio", c.ratio, w, g,
                     "the same diagnostic restricted to sources that contribute to more than "
                     "one group; read beside entropy.ratio, not instead of it"),
                _row("crossed.U_G_given_D", c.uncertainty_coefficient, w, g, ""),
                _row("crossed.n_sources", float(len(self.spanning)), s, g, ""),
                _row("crossed.n_rows", _as_float(cd.get("n_rows")) if cd is not None else float("nan"),
                     s, g, "size of the subcorpus in which biological group varies within provenance strata"),
                _row("crossed.n_units", _as_float(cd.get("n_units")) if cd is not None else float("nan"),
                     s, g, ""),
            ]
        else:
            rows.append(
                _row("crossed.ratio", float("nan"), w, g,
                     "undefined: no source contributes to more than one group, so there is no "
                     "stratum in which provenance varies at fixed group")
            )
        if self.pair_sharing is not None:
            rows += _frame_rows(
                "pair_sharing", self.pair_sharing, weighting=s, grain=g,
                note="unit pairs sharing at least one source, within group",
                preferred=_PAIR_SHARING_COLUMNS,
            )
        else:
            rows.append(_row("pair_sharing", float("nan"), s, g, self._why("pair sharing")))
        if self.structural_loso is not None:
            rows += _extremum_rows(
                "structural_loso", self.structural_loso, weighting=s, grain=g,
                preferred=_STRUCTURAL_COLUMNS,
            )
        else:
            rows.append(
                _row("structural_loso", float("nan"), s, g, self._why("structural leave-one-source-out"))
            )
        cw = self.ceiling_weighting or w
        if self.ceiling is not None or np.isfinite(self.ceiling_value):
            skip = set(_CEILING_VALUE_KEYS) | {"weighting", "grain", "corpus", "name"}
            described = ", ".join(
                f"{k} {v!r}" for k, v in self.ceiling_detail.items()
                if isinstance(v, str) and v and k not in skip
            )
            rows.append(
                _row("ceiling.balanced_accuracy", self.ceiling_value, cw, g, 
                     ("exact upper bound on balanced accuracy for a predictor "
                    "whose representation is exactly invariant to the declared "
                    f"provenance; chance is {self.chance_level:.3g}" + (f"; {described}" if described else "")),
                    )
            )
            rows += [
                _row(f"ceiling.{k}", _as_float(v), cw, g, "")
                for k, v in self.ceiling_detail.items()
                if k not in skip and not isinstance(v, str)
            ]
        else:
            rows.append(
                _row("ceiling.balanced_accuracy", float("nan"), cw, g, self._why("balanced-accuracy ceiling"))
            )
        if self.guards is not None:
            rows += _guard_rows(self.guards, weighting=w, grain=g)
        else:
            rows.append(_row("guard", float("nan"), w, g, self._why("guard report")))
        if self.bootstrap is not None:
            b = self.bootstrap
            rows += [
                _row("bootstrap.lo", b.lo, w, g, b.note),
                _row("bootstrap.hi", b.hi, w, g, b.note),
                _row("bootstrap.n_effective", float(b.n_effective), w, g,
                     f"{b.n_degenerate} resamples dropped as degenerate (one group survived)"),
                _row("bootstrap.frac_exactly_zero", b.frac_exactly_zero, w, g,
                     "fraction of non-degenerate resamples with ratio exactly zero"),
            ]

        return pd.DataFrame(rows, columns=["quantity", "value", "weighting", "grain", "note"])

    def _why(self, stage: str) -> str:
        """
        Find the first note explaining a missing stage, or report that it was not computed.
        """
        for n in self.notes:
            if n.lower().startswith(stage.lower()):
                return n
        return f"{stage}: not computed"

    def to_markdown(self, *, extra: pd.DataFrame | None = None) -> str:
        """
        Render a readable diagnosis with context, verdict, quantity table, and stage notes.

        Parameters:
        ----------
        extra : pd.DataFrame or None, default None, keyword-only - additional reporting
        rows. Columns are aligned to the standard quantity table before appending.

        Returns:
        -------
        str - Markdown text ready for review or saving by the caller.

        Notes:
        ------
        Extra rows change the displayed table only; they do not alter component
        calculations or the verdict. Numerical formatting is for presentation, so
        retain component results or ``to_frame`` for full-precision values. The prose
        must be interpreted under the stated grain and weighting and alongside notes.
        """
        e = self.entropy
        frame = self.to_frame()
        if extra is not None and not extra.empty:
            extra = extra.reindex(columns=frame.columns)
            frame = pd.concat([frame, extra], ignore_index=True)    

        prose = (
            f"Provenance was declared at the grain {self.grain} and quantities were computed "
            f"under {self.weighting}. The corpus has {_fmt(self.design.get('n_rows'))} incidence "
            f"rows over {_fmt(self.design.get('n_sources'))} sources, "
            f"{_fmt(self.design.get('n_units'))} units and {_fmt(self.design.get('n_groups'))} groups. "
            f"Residual label entropy is H(G|D) = {_fmt(e.H_G_given_D)} bits against "
            f"H(G) = {_fmt(e.H_G)} bits, a ratio of {_fmt(e.ratio)}, so knowing the source removes "
            f"{_fmt(100.0 * e.uncertainty_coefficient)}% of the label's entropy. "
            f"{e.n_spanning_sources} of {e.n_sources} sources span biological groups; "
            f"{e.n_nested_sources} positive-weight sources are nested within one group."
        )
        if np.isfinite(self.ceiling_value):
            prose += (
                f" The exact provenance-invariant balanced-accuracy ceiling is "
                f"{_fmt(self.ceiling_value)} (chance {_fmt(self.chance_level)}). "
                "A predictor whose representation is exactly independent of the "
                "declared provenance cannot exceed this value under the stated weighting."
            )
        if self.crossed_entropy is not None:
            prose += (
                f" Restricted to the {_count(len(self.spanning), 'source')} that contribute to "
                f"more than one group, the ratio is {_fmt(self.crossed_entropy.ratio)}."
            )

        out = [
            f"## Provenance diagnosis: {self.corpus}",
            "",
            f"Verdict. {self.verdict()}",
            "",
            prose,
            "",
            _md_table(frame),
        ]
        if self.notes:
            out += ["", "**Notes.**", ""] + [f"- {n}" for n in self.notes]
        return "\n".join(out)

    def verdict(self) -> str:
        """
        Describe the provenance constraints using the stored diagnostics and reporting bands.

        Returns:
        -------
        str - a paragraph covering the full design, any available crossed stratum,
        and the declared grain and weighting.

        Computation:
        -----------
        A finite ceiling takes precedence. Exact chance, values up to 0.55, values
        up to 0.70, and values above 0.70 receive progressively less constrained
        wording. Without a finite ceiling, R is described using boundaries 0.20 and
        0.60. A crossed-subcorpus ratio of at least 0.50 receives the more favorable
        crossed-stratum wording.

        Notes:
        ------
        These bands select prose; they are neither significance tests nor universal
        estimability thresholds. In particular, the fallback word "estimable" does
        not establish causal identification, adequate replication, or biological
        validity. Inspect numerical results and stage notes rather than treating this
        paragraph as an independent analysis. An unavailable crossed result requires
        its note to distinguish absent crossing from a computation failure.
        """
        ceil, ratio, chance = self.ceiling_value, self.entropy.ratio, self.chance_level
        parts: list[str] = []

        if np.isfinite(ceil):
            if ceil <= chance + 1e-12:
                parts.append(
                    f"Not estimable under exact provenance invariance: "
                    f"the balanced-accuracy ceiling is {ceil:.4g}, equal to "
                    f"the chance level {chance:.3g}."
                )
            elif ceil <= CEILING_CHANCE_BAND:
                parts.append(
                    f"Severely provenance-constrained: the provenance-invariant "
                    f"balanced-accuracy ceiling is only {ceil:.4g}, close to the "
                    f"chance level {chance:.3g}."
                )
            elif ceil <= CEILING_USABLE:
                parts.append(
                    f"Provenance-constrained: the provenance-invariant ceiling "
                    f"is {ceil:.4g} balanced accuracy (chance {chance:.3g})."
                )
            else:
                parts.append(
                    f"The incidence design leaves greater provenance-invariant "
                    f"headroom, with a balanced-accuracy ceiling of {ceil:.4g} "
                    f"(chance {chance:.3g}). This does not by itself establish "
                    "that an observed biological contrast is valid."
                )
        elif np.isfinite(ratio):
            if ratio <= RATIO_ALIASED:
                parts.append(
                    f"Not estimable as assembled: the label retains only {ratio:.4g} of its "
                    f"entropy once the source is known, so the group effect and the source effect "
                    f"are very nearly the same effect."
                )
            elif ratio <= RATIO_CROSSED:
                parts.append(
                    f"Partially estimable: {ratio:.4g} of the label's entropy survives "
                    f"conditioning on the source, so group and provenance are entangled but not "
                    f"identical."
                )
            else:
                parts.append(
                    f"Estimable: {ratio:.4g} of the label's entropy survives conditioning on the "
                    f"source, so provenance carries little information about the group."
                )
        else:
            parts.append(
                "Undetermined: neither the ceiling nor the entropy ratio is defined for this "
                "corpus, so no claim about estimability is made here."
            )

        if self.crossed_entropy is not None:
            cr = self.crossed_entropy.ratio
            n_rows = _fmt(self.crossed_design.get("n_rows")) if self.crossed_design is not None else "n/a"
            crossing = _count(len(self.spanning), "source")
            scores = "scores" if len(self.spanning) == 1 else "score"
            if np.isfinite(cr) and cr >= CROSSED_STRATUM_RATIO:
                parts.append(
                    f"A crossed stratum does exist: the {crossing} contributing to more than one "
                    f"group ({n_rows} rows) {scores} {cr:.4g}, so within that stratum the contrast "
                    f"may be estimable, at that much reduced size."
                )
            else:
                parts.append(
                    f"The {crossing} contributing to more than one group {scores} {cr:.4g} among "
                    f"themselves, so even the crossed stratum is only weakly informative about "
                    f"the label."
                )
        else:
            parts.append(
                "No source contributes to more than one group, so there is no "
                "provenance stratum in which the biological group varies. At this "
                "declared grain, the incidence structure therefore provides no "
                "within-source information for separating group from provenance."
            )

        parts.append(
            f"This is a statement about a design, not about any analysis or any author, and it "
            f"holds conditional on the declared grain ({self.grain}) and weighting "
            f"({self.weighting}); the same corpus can score differently at another grain."
        )
        return " ".join(parts)

    def as_dict(self) -> dict:
        """
        Return a nested reporting dictionary with context, summaries, notes, and table rows.

        Includes design counts, entropy summaries, crossed entropy, spanning-source keys,
        ceiling details, optional bootstrap summary, and the rendered quantity table.
        It does not preserve every full component table or the bootstrap draw array.
        Values may include ``nan`` and NumPy or pandas scalars; this is not guaranteed
        to be strict-JSON-ready without further serialization handling.
        """
        return {
            "corpus": self.corpus,
            "grain": self.grain,
            "weighting": self.weighting,
            "verdict": self.verdict(),
            "design": {k: v for k, v in self.design.items()},
            "entropy": self.entropy.as_dict(),
            "crossed": self.crossed_entropy.as_dict() if self.crossed_entropy is not None else None,
            "spanning_sources": list(self.spanning),
            "ceiling": {
                "balanced_accuracy": self.ceiling_value,
                "weighting": self.ceiling_weighting or self.weighting,
                **{k: v for k, v in self.ceiling_detail.items() if k not in _CEILING_VALUE_KEYS},
            },
            "bootstrap": self.bootstrap.as_dict() if self.bootstrap is not None else None,
            "notes": list(self.notes),
            "table": self.to_frame().to_dict(orient="records"),
        }

    def summary(self) -> str:
        """
        One-line report of R, U, ceiling, crossing, source grain, and weighting.
        """
        return (
            f"{self.corpus}: ratio = {_fmt(self.entropy.ratio)}, "
            f"U(G|D) = {_fmt(self.entropy.uncertainty_coefficient)}, "
            f"ceiling = {_fmt(self.ceiling_value)} (chance {_fmt(self.chance_level)}), "
            f"{len(self.spanning)} of {_count(self.entropy.n_sources, 'source')} cross groups "
            f"[grain: {self.grain}; weighting: {self.weighting}]"
        )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        """
        Compact representation wrapping the diagnostic summary.
        """
        return f"Diagnosis({self.summary()})"

def diagnose(
    corpus: Corpus,
    *,
    grain: str | None = None,
    bootstrap: int = 0,
    seed: int = 0,
    strict: bool = False,
) -> Diagnosis:
    """
    Run the metadata diagnostic stack for one corpus and collect the available results.

    Parameters:
    ----------
    corpus : Corpus - validated incidence design with its weights and record lineage.
    grain : str or None, default None, keyword-only - readable definition of the
    source column, such as study or laboratory. This labels the result; it does
    not reproject the corpus. None adds an undeclared-grain note.
    bootstrap : int, default 0, keyword-only - number of source-resampling
    replicates. Zero skips that stage; positive values are passed as an integer.
    seed : int, default 0, keyword-only - seed used when bootstrap is requested.
    strict : bool, default False, keyword-only - re-raise exceptions from optional
    diagnostic evaluations. False records the exception in notes and continues.

    Returns:
    -------
    Diagnosis - required full-corpus entropy and design counts, plus available
    crossed entropy, pair sharing, structural source removal, spanning-source
    details, exact binary ceiling, guards, and optional bootstrap sensitivity.

    Raises:
    ------
    TypeError
    - if corpus is not a Corpus.
    ValueError
    - if bootstrap is negative or the required entropy calculation is invalid.
    Exceptions from required calculations always propagate. With strict=True,
    exceptions raised by optional stages propagate as well.

    Notes:
    ------
    The exact ceiling is implemented for two groups. A multigroup corpus may
    therefore produce useful entropy and structural results while the ceiling
    stage is noted as unavailable in a non-strict run. Missing functions receive
    availability notes even with strict=True.

    Bootstrap limits summarize resampling of the observed sources and are not
    automatically calibrated confidence intervals. Read ``notes`` before using
    the reporting helpers, especially when a component is None or non-finite.
    No feature matrix is read and no classifier or correction method is fit.
    """
    if not isinstance(corpus, Corpus):
        raise TypeError(f"diagnose() takes a Corpus, got {type(corpus).__name__}")
    if bootstrap < 0:
        raise ValueError("bootstrap must be a non-negative number of replicates")

    notes: list[str] = []
    grain_label = grain or UNDECLARED_GRAIN
    if grain is None:
        notes.append(
            "grain: undeclared. Every number below is conditional on what a source means; "
            "declare it with diagnose(corpus, grain=...) before quoting any of them."
        )

    def stage(label: str, fn: Callable[..., Any] | None, *args: Any, **kw: Any) -> Any:
        """
        Run one optional diagnostic and record unavailability or failure by stage name.

        Missing functions return None with a note. Evaluation exceptions are re-raised
        in strict mode; otherwise their type and message are recorded and None is returned.
        """
        if fn is None:
            notes.append(f"{label}: not available in this installation")
            return None
        try:
            return fn(*args, **kw)
        except Exception as exc:
            if strict:
                raise
            notes.append(f"{label}: not computed ({type(exc).__name__}: {exc})")
            return None

    entropy = label_entropy(corpus)
    design = corpus.describe()
    spanning = corpus.spanning_sources()

    crossed_entropy: EntropyResult | None = None
    crossed_design: pd.Series | None = None
    if spanning:
        crossed = stage("crossed subcorpus", corpus.crossed_subcorpus)
        if crossed is not None:
            crossed_entropy = stage("crossed subcorpus entropy", label_entropy, crossed)
            crossed_design = crossed.describe()
    else:
        notes.append(
            "crossed subcorpus: undefined -- no source contributes to more than one group, "
            "so the corpus is completely aliased at this grain."
        )

    pair_sharing = _as_frame(
        stage("pair sharing", _resolve("pair_sharing_by_group", "structure"), corpus)
    )
    structural_loso = _as_frame(
        stage(
            "structural leave-one-source-out",
            _resolve("structural_leave_one_source_out", "structure"),
            corpus,
        )
    )
    spanning_table = _as_frame(
        stage("spanning source table", _resolve("spanning_source_table", "structure"), corpus)
    ) if spanning else None

    ceiling_obj = stage(
        "balanced-accuracy ceiling", _resolve("balanced_accuracy_ceiling", "ceiling"), corpus
    )
    ceiling_detail = _as_mapping(ceiling_obj)
    if isinstance(ceiling_obj, (int, float, np.integer, np.floating)) and not isinstance(
        ceiling_obj, bool
    ):
        ceiling_value = float(ceiling_obj)
        ceiling_detail = {}
    else:
        ceiling_value = _as_float(_pick(ceiling_detail, _CEILING_VALUE_KEYS))
    ceiling_weighting = str(_pick(ceiling_detail, ("weighting",)) or "")
    if ceiling_obj is not None and not np.isfinite(ceiling_value):
        notes.append(
            "balanced-accuracy ceiling: the ceiling module returned a result whose value field "
            "could not be identified; the object is on Diagnosis.ceiling."
        )

    guard_fn = _resolve("guard_report", "guards")
    guards = _as_frame(
        stage(
            "guard report",
            _call if guard_fn is not None else None,
            guard_fn,
            corpus,
            grain=grain,
            ceiling=ceiling_value if np.isfinite(ceiling_value) else None,
        )
    )

    boot: BootstrapResult | None = None
    if bootstrap:
        boot = stage("bootstrap", bootstrap_ratio, corpus, B=int(bootstrap), seed=int(seed))

    return Diagnosis(
        corpus=corpus.name,
        weighting=corpus.weighting,
        grain=grain_label,
        design=design,
        entropy=entropy,
        n_groups=corpus.n_groups,
        spanning=spanning,
        crossed_entropy=crossed_entropy,
        crossed_design=crossed_design,
        pair_sharing=pair_sharing,
        structural_loso=structural_loso,
        spanning_table=spanning_table,
        ceiling=ceiling_obj,
        ceiling_value=ceiling_value,
        ceiling_weighting=ceiling_weighting,
        ceiling_detail=ceiling_detail,
        guards=guards,
        bootstrap=boot,
        notes=notes,
    )
