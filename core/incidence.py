"""
The incidence corpus and its record-level provenance.

What is a source, what is an analysis unit, and what contrast does the group label
describe? These roles must be declared before provenance aliasing can be measured.
The corpus supplies a common representation across resources and omics assays
without inferring those scientific roles from file names or molecular features.

Data structure:
---------------
``frame`` contains the normalized analytical projection: one retained observation
per row, with source, unit, group, and weight. ``records`` retains a pandas deep
copy of the supplied records, including metadata and rows excluded from analysis.
An internal record projection connects these views and tracks exclusion reasons.

Keys and hierarchy:
-------------------
A role may name one column or several columns forming a tuple-valued key. Text
components are stripped and missing identifiers remain missing. Composite keys
make local identifiers explicit, for example a sample ID qualified by its study.
Each analysis unit must carry one group. Declared parent metadata can connect
cells or aliquots to a donor, but the code never infers parentage from identifiers
or fills missing sample metadata from dataset totals.

Repeated observations:
----------------------
Repeated normalized ``(source, unit, group)`` keys are diagnosed separately from
the raw records. The default warns and retains them. Callers can explicitly keep,
drop, or reject repetitions; dropping keeps the first equal-weight observation
and refuses unequal weights rather than selecting a scientific reducer.

Weighting:
---------
Every retained row has mass 1 unless a weight column is supplied. Weights must be
real, finite, and non-negative, with a finite positive total. Weighted source x
group totals drive entropy and ceiling calculations. Binary source x unit
membership drives structural calculations and includes zero-weight observations.

Contents:
---------
Corpus - construction, normalized views, hierarchy diagnostics, and reprojection.
DuplicateObservationWarning - repeated analytical keys retained under "warn".
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Iterable, Mapping, Sequence
import warnings

import numpy as np
import pandas as pd

__all__ = ["Corpus", "DuplicateObservationWarning"]

KeySpec = str | Sequence[str]
Key = object
DuplicatePolicy = str

_REQUIRED = ("source", "unit", "group")
_DUPLICATE_POLICIES = frozenset({"warn", "keep", "drop", "raise"})
_MISSING_TEXT = frozenset({"", "nan", "none", "<na>"})
_HIERARCHY_PREFIX = "__hierarchy__:"
_DUPLICATE_REPORT_COLUMNS = (
    "source",
    "unit",
    "group",
    "n_records",
    "excess_rows",
    "kind",
    "varying_columns",
    "record_positions",
)
_HIERARCHY_REPORT_COLUMNS = (
    "columns",
    "n_units",
    "n_parents",
    "n_units_with_parent",
    "n_units_missing_parent",
    "n_records_missing_parent",
    "n_partially_mapped_units",
    "n_multi_unit_parents",
    "max_units_per_parent",
)


class DuplicateObservationWarning(UserWarning):
    """
    A normalized source-unit-group observation key occurs more than once.

    Emitted when ``duplicates="warn"`` retains repeated analytical observations.
    Inspect ``Corpus.duplicate_report`` to distinguish exact repeated records from
    records that share an analytical key but differ in other metadata.
    """


def _empty_duplicate_report() -> pd.DataFrame:
    """
    Return an empty duplicate table with the standard collision-report columns.
    """
    return pd.DataFrame(columns=list(_DUPLICATE_REPORT_COLUMNS))


def _normalise_spec(spec: KeySpec, *, role: str) -> tuple[str, ...]:
    """
    Validate a role's column specification and return a non-empty tuple of names.

    A single string names one column. Sequences must contain unique, non-empty
    strings; malformed specifications raise TypeError or ValueError with the role
    included in the message.
    """
    if isinstance(spec, str):
        columns = (spec,)
    else:
        try:
            columns = tuple(spec)
        except TypeError as exc:  # pragma: no cover - defensive API boundary
            raise TypeError(f"{role} must be a column name or a sequence of names") from exc
    if not columns or any(not isinstance(column, str) or not column for column in columns):
        raise ValueError(f"{role} must name at least one non-empty column")
    if len(set(columns)) != len(columns):
        raise ValueError(f"{role} contains repeated column names: {columns!r}")
    return columns


def _missing_columns(df: pd.DataFrame, specs: Iterable[tuple[str, ...]]) -> list[str]:
    """
    Return sorted required column names absent from the input DataFrame.
    """
    required = {column for spec in specs for column in spec}
    return sorted(required.difference(df.columns))


def _normalise_component(values: pd.Series) -> pd.Series:
    """
    Strip identifier components while preserving missing values and structured keys.

    Parameters:
    ----------
    values : pd.Series - one physical identifier column, possibly containing
    already canonical tuple-valued keys.

    Returns:
    -------
    pd.Series - normalized identifiers with a fresh positional index. Scalars
    become stripped strings. Missing values and the case-insensitive tokens
    ``""``, ``"nan"``, ``"none"``, and ``"<na>"`` become ``pd.NA``.
    Tuple-valued keys retain their tuple structure with stripped text components;
    an empty tuple or any missing component makes the key missing.

    Notes:
    ------
    Tuple preservation allows analytical projections to be rebuilt without
    turning composite identifiers into ambiguous display strings.
    """
    raw = values.reset_index(drop=True)
    cleaned = raw.astype("string").str.strip()
    missing = cleaned.isna() | cleaned.str.lower().isin(_MISSING_TEXT)
    cleaned = cleaned.mask(missing, pd.NA)
    tuple_positions = [
        position for position, value in enumerate(raw.tolist()) if isinstance(value, tuple)
    ]
    if not tuple_positions:
        return cleaned

    result = cleaned.astype(object)
    for position in tuple_positions:
        value = raw.iloc[position]
        components: list[str] = []
        invalid = len(value) == 0
        for component in value:
            if _is_missing_value(component):
                invalid = True
                break
            text = str(component).strip()
            if text.lower() in _MISSING_TEXT:
                invalid = True
                break
            components.append(text)
        result.iloc[position] = pd.NA if invalid else tuple(components)
    return result


def _column_series(df: pd.DataFrame, column: str) -> pd.Series:
    """
    Select exactly one physical column by label.

    Raises ValueError when a duplicated column label makes the selection ambiguous;
    missing labels raise the pandas KeyError.
    """
    location = df.columns.get_loc(column)
    if not isinstance(location, (int, np.integer)):
        raise ValueError(f"column label {column!r} is duplicated in the input frame")
    return df.iloc[:, int(location)]


def _normalise_key(df: pd.DataFrame, spec: tuple[str, ...]) -> pd.Series:
    """
    Build one scalar or composite analytical key from declared physical columns.

    Single-column specifications use normalized components directly. Multiple
    columns become ordered tuples of text; a missing component makes the entire
    key missing. The returned Series uses positional row order.
    """
    components = [_normalise_component(_column_series(df, column)) for column in spec]
    if len(components) == 1:
        return components[0]

    values: list[object] = []
    for row in zip(*(component.tolist() for component in components)):
        if any(pd.isna(value) for value in row):
            values.append(pd.NA)
        else:
            values.append(tuple(str(value) for value in row))
    return pd.Series(values, index=pd.RangeIndex(len(df)), dtype=object)


def _is_missing_value(value: object) -> bool:
    """
    Recognize scalar pandas missing values without treating array-like results as Boolean.
    """
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return bool(missing) if isinstance(missing, (bool, np.bool_)) else False


def _stable_value(value: object) -> object:
    """
    Represent metadata values comparably for duplicate diagnostics.

    Missing scalars share one token. Hashable values are paired with their type;
    unhashable objects use type and repr. This supports collision descriptions,
    not a general serialization or identity scheme for arbitrary objects.
    """
    if _is_missing_value(value):
        return ("missing",)
    try:
        hash(value)
    except TypeError:
        return (type(value).__name__, repr(value))
    return (type(value).__name__, value)


def _as_index(values: Sequence[object]) -> pd.Index:
    """
    Construct an object Index that retains tuple keys as single labels.

    Disables pandas' automatic conversion of tuple-valued keys into a MultiIndex.
    """
    return pd.Index(list(values), dtype=object, tupleize_cols=False)


def _sorted_unique(values: pd.Series) -> list[Key]:
    """
    Return unique non-missing keys in deterministic order.

    Uses natural sorting when possible, otherwise sorts by type name and repr.
    """
    unique = list(pd.unique(values.dropna()))
    try:
        return sorted(unique)
    except TypeError:
        return sorted(unique, key=lambda value: (type(value).__name__, repr(value)))


def _validate_key_shape(values: pd.Series, *, role: str) -> None:
    """
    Require a role to use either scalar keys or tuple keys consistently.

    Missing values are ignored. Mixing scalar and tuple identifiers raises ValueError.
    """
    shapes = {"tuple" if isinstance(value, tuple) else "scalar" for value in values.dropna()}
    if len(shapes) > 1:
        raise ValueError(
            f"{role} mixes scalar and tuple-valued keys; use one declared key shape"
        )


def _selection(values: Sequence[object] | object) -> list[object]:
    """
    Treat a string selection as one key and other selections as iterables of keys.

    Wrap a composite tuple key in a sequence so its components are not interpreted
    as separate selections.
    """
    if isinstance(values, (str, bytes)):
        return [values]
    return list(values)  # type: ignore[arg-type]


def _validate_weights(values: pd.Series, *, context: str = "weights") -> None:
    """
    Reject invalid row weights and nonrepresentable aggregate mass.

    Parameters:
    ----------
    values : pd.Series - weights from rows already eligible for analysis.
    context : str, default "weights", keyword-only - label used in error messages.

    Raises:
    ------
    ValueError
    - if weights cannot be converted to real finite numbers, contain missing or
    negative values, have a non-finite floating-point sum, or have no positive mass.

    Notes:
    ------
    Eligibility is selected by the caller before validation. Individual zero
    weights are allowed. Checking the total also catches overflow from adding
    individually finite weights.
    """
    numeric = pd.to_numeric(values, errors="coerce")
    if np.iscomplexobj(numeric.to_numpy()):
        raise ValueError(f"{context} must be real, finite numeric values")
    try:
        weights = numeric.to_numpy(dtype=float, na_value=np.nan)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{context} must be finite numeric values") from exc
    if not np.isfinite(weights).all():
        raise ValueError(f"{context} must be finite numeric values without missing values")
    if (weights < 0).any():
        raise ValueError(f"{context} must be non-negative")
    # Even individually finite weights can overflow a floating-point reduction.
    # Convert that condition to an input error before downstream normalization.
    with np.errstate(over="ignore", invalid="ignore"):
        total = weights.sum(dtype=np.float64)
    if not np.isfinite(total):
        raise ValueError(f"{context} sum must be finite; reduce the weight scale")
    if total <= 0:
        raise ValueError(f"{context} sum to zero")


@dataclass(frozen=True)
class Corpus:
    """
    A normalized incidence design with the records needed to explain its construction.

    Use :meth:`from_frame` or :meth:`from_records` to declare column roles. The
    legacy direct constructor accepts a normalized frame and rebuilds its lineage
    when lineage fields have not already been supplied.

    Attributes:
    -----------
    frame : pd.DataFrame - retained analytical rows with ``source``, ``unit``,
    ``group``, and float ``weight`` columns. Every unit has one group label.
    name : str, default "corpus" - label carried into diagnostic results.
    weighting : str, default "incidence rows" - description of analytical mass.
    records : pd.DataFrame - preserved input records, including metadata and rows
    excluded from the analytical projection. Transformations retain their relevant
    record lineage.
    duplicate_policy : str, default "warn" - treatment of repeated analytical keys.
    duplicate_report : pd.DataFrame - one row per repeated source-unit-group key,
    including record counts, excess rows, collision kind, varying metadata columns,
    and zero-based positions in ``records``.
    hierarchy : mapping - declared parent level to its physical key columns.
    source_spec, unit_spec, group_spec : tuple of str - column definitions of the
    three analytical roles.
    weight_column : str or None - original explicit weight column, when supplied.
    _record_projection : pd.DataFrame - internal normalized record lineage and
    analysis inclusion/exclusion status; not an additional observation table.

    Notes:
    ------
    The frozen dataclass prevents field reassignment, but its DataFrames and
    mappings remain mutable. Use the transformation methods to preserve invariants.
    Pandas deep copies do not recursively copy arbitrary Python objects stored in
    object-valued cells.

    Neither the number of records nor the number of incidence rows establishes
    independent replication. Declare the biological unit and parent hierarchy
    before interpreting source support or an entropy ratio.
    """

    frame: pd.DataFrame
    name: str = "corpus"
    weighting: str = "incidence rows"
    records: pd.DataFrame | None = field(default=None, repr=False, compare=False)
    duplicate_policy: DuplicatePolicy = "warn"
    duplicate_report: pd.DataFrame = field(
        default_factory=_empty_duplicate_report, repr=False, compare=False
    )
    hierarchy: Mapping[str, tuple[str, ...]] = field(
        default_factory=dict, repr=False, compare=False
    )
    source_spec: tuple[str, ...] = field(
        default=("source",), repr=False, compare=False
    )
    unit_spec: tuple[str, ...] = field(default=("unit",), repr=False, compare=False)
    group_spec: tuple[str, ...] = field(default=("group",), repr=False, compare=False)
    weight_column: str | None = field(default=None, repr=False, compare=False)
    _record_projection: pd.DataFrame | None = field(
        default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        # Keep the legacy ``Corpus(frame=...)`` constructor useful and validated. All
        # factory/transform paths supply both lineage fields and skip reconstruction.
        """
        Validate supplied analytical weights or build lineage for the direct constructor.

        Factory-created instances already carry records and their projection. A legacy
        ``Corpus(frame=...)`` call is reconstructed through ``from_frame`` and receives
        the same normalization and collision checks, retaining its weighting label.
        """
        if not isinstance(self.frame, pd.DataFrame):
            raise TypeError("frame must be a pandas DataFrame")
        if self.records is not None and self._record_projection is not None:
            _validate_weights(self.frame["weight"], context="analytical weights")
            return
        weight = "weight" if "weight" in self.frame.columns else None
        built = type(self).from_frame(
            self.frame,
            source="source",
            unit="unit",
            group="group",
            weight=weight,
            name=self.name,
            duplicates=self.duplicate_policy,
            hierarchy=self.hierarchy or None,
        )
        for item in fields(self):
            value = getattr(built, item.name)
            if item.name == "weighting":
                value = self.weighting
            object.__setattr__(self, item.name, value)

    @staticmethod
    def _validate_duplicate_policy(duplicates: DuplicatePolicy) -> None:
        """
        Require one of ``"warn"``, ``"keep"``, ``"drop"``, or ``"raise"``.
        """
        if duplicates not in _DUPLICATE_POLICIES:
            allowed = ", ".join(repr(policy) for policy in sorted(_DUPLICATE_POLICIES))
            raise ValueError(f"duplicates must be one of {allowed}; got {duplicates!r}")

    @classmethod
    def from_frame(
        cls,
        df: pd.DataFrame,
        *,
        source: KeySpec = "source",
        unit: KeySpec = "unit",
        group: KeySpec = "group",
        weight: str | None = None,
        name: str = "corpus",
        dropna: bool = True,
        duplicates: DuplicatePolicy = "warn",
        hierarchy: Mapping[str, KeySpec] | None = None,
    ) -> "Corpus":
        """
        Build a corpus by declaring the analytical roles of an input table.

        Column declarations make the representation independent of the assay. A source
        may be a study or run; a unit may be a donor, sample, or cell, provided that
        definition matches the intended analysis. Composite roles preserve the scope
        of identifiers that are only unique within a resource.

        Parameters:
        ----------
        df : pd.DataFrame - input records. All supplied rows and metadata columns are
        preserved in ``records``, including rows excluded from analysis.
        source, unit, group : str or sequence of str, keyword-only - columns defining
        the three roles; defaults are ``"source"``, ``"unit"``, and ``"group"``.
        A multi-column declaration becomes a tuple-valued key in column order.
        weight : str or None, default None, keyword-only - mass column. None assigns
        mass 1 per analytical row. Accepted weights must be finite, real, non-negative,
        and have finite positive total mass.
        name : str, default "corpus", keyword-only - reporting label.
        dropna : bool, default True, keyword-only - exclude records with missing
        normalized source, unit, or group keys. False raises on those records.
        duplicates : str, default "warn", keyword-only - repeated-key policy.
        ``"warn"`` retains every row and emits one warning; ``"keep"`` retains
        silently; ``"drop"`` keeps the first observation for each equal-weight key;
        ``"raise"`` rejects any repeated analytical key.
        hierarchy : mapping or None, default None, keyword-only - parent level name
        to one or more physical key columns, for example a donor identifier. Missing
        parents remain missing and do not exclude an otherwise complete analytical row.

        Returns:
        -------
        Corpus - normalized analytical rows, preserved records, declared key
        specifications, duplicate diagnostics, and record-level inclusion lineage.

        Raises:
        ------
        TypeError
        - if df is not a DataFrame or a column specification is not usable.
        KeyError
        - if a declared column is absent.
        ValueError
        - if keys, weights, or policies are invalid, or no analytical rows remain.
        - if one unit carries multiple groups or multiple non-missing parents at a level.
        - if repeated keys are forbidden, or dropping would discard unequal weights.

        Notes:
        ------
        Scalar identifiers are converted to stripped text; recognized missing tokens
        remain missing. Collisions are checked after normalization, so distinct raw
        spellings can resolve to the same analytical key. Duplicate dropping is not
        aggregation: unequal masses require an explicit reducer before construction.
        No hierarchy or biological relationship is inferred from the identifier text.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")
        cls._validate_duplicate_policy(duplicates)

        source_spec = _normalise_spec(source, role="source")
        unit_spec = _normalise_spec(unit, role="unit")
        group_spec = _normalise_spec(group, role="group")
        hierarchy_specs: dict[str, tuple[str, ...]] = {}
        for level, spec in (hierarchy or {}).items():
            if not isinstance(level, str) or not level.strip():
                raise ValueError("hierarchy level names must be non-empty strings")
            clean_level = level.strip()
            if clean_level in hierarchy_specs:
                raise ValueError(f"hierarchy level {clean_level!r} is repeated")
            hierarchy_specs[clean_level] = _normalise_spec(
                spec, role=f"hierarchy[{clean_level!r}]"
            )

        specs = [source_spec, unit_spec, group_spec, *hierarchy_specs.values()]
        missing = _missing_columns(df, specs)
        if missing:
            raise KeyError(f"columns not found in frame: {missing}; have {list(df.columns)}")
        if weight is not None and weight not in df.columns:
            raise KeyError(f"weight column {weight!r} not found; have {list(df.columns)}")

        projection = pd.DataFrame(
            {
                "source": _normalise_key(df, source_spec),
                "unit": _normalise_key(df, unit_spec),
                "group": _normalise_key(df, group_spec),
            }
        )
        for role in _REQUIRED:
            _validate_key_shape(projection[role], role=role)
        projection["complete"] = projection[list(_REQUIRED)].notna().all(axis=1)
        n_missing = int((~projection["complete"]).sum())
        if n_missing and not dropna:
            raise ValueError(f"{n_missing} rows have a missing source/unit/group")

        if weight is None:
            projection["weight"] = 1.0
            weighting = "incidence rows"
        else:
            numeric_weight = pd.to_numeric(
                _column_series(df, weight).reset_index(drop=True),
                errors="coerce",
            )
            accepted_weight = numeric_weight.loc[projection["complete"]]
            if not accepted_weight.empty:
                _validate_weights(accepted_weight, context=f"weight column {weight!r}")
            projection["weight"] = numeric_weight.astype(float)
            weighting = f"weighted by {weight}"

        for level, spec in hierarchy_specs.items():
            projection[f"{_HIERARCHY_PREFIX}{level}"] = _normalise_key(df, spec)

        return cls._from_projection(
            records=df,
            projection=projection,
            name=name,
            weighting=weighting,
            duplicates=duplicates,
            hierarchy=hierarchy_specs,
            source_spec=source_spec,
            unit_spec=unit_spec,
            group_spec=group_spec,
            weight_column=weight,
            emit_warning=True,
        )

    @classmethod
    def from_records(cls, records: Iterable[Mapping[str, object]], **kw: object) -> "Corpus":
        """
        Build a corpus from an iterable of record mappings.

        Parameters:
        ----------
        records : iterable of mappings - consumed into a list and converted to a
        pandas DataFrame.
        **kw : keyword arguments - column roles, weighting, missing-key policy,
        duplicate policy, and hierarchy accepted by :meth:`from_frame`.

        Returns:
        -------
        Corpus - the same validated representation as ``from_frame``.

        Notes:
        ------
        The iterable is materialized in memory. Validation and errors are delegated
        to DataFrame construction and ``from_frame``.
        """
        return cls.from_frame(pd.DataFrame(list(records)), **kw)

    @classmethod
    def _from_projection(
        cls,
        *,
        records: pd.DataFrame,
        projection: pd.DataFrame,
        name: str,
        weighting: str,
        duplicates: DuplicatePolicy,
        hierarchy: Mapping[str, tuple[str, ...]],
        source_spec: tuple[str, ...],
        unit_spec: tuple[str, ...],
        group_spec: tuple[str, ...],
        weight_column: str | None,
        emit_warning: bool,
    ) -> "Corpus":
        """
        Rebuild analytical rows and lineage from normalized record-level keys.

        Parameters:
        ----------
        records, projection : pd.DataFrame, keyword-only - raw records and an aligned
        projection containing source, unit, group, weight, and completeness status.
        name, weighting : str, keyword-only - provenance labels for the new corpus.
        duplicates : str, keyword-only - repeated analytical-key policy.
        hierarchy : mapping, keyword-only - declared parent-level specifications.
        source_spec, unit_spec, group_spec : tuple of str, keyword-only - physical
        column declarations to retain with the transformed representation.
        weight_column : str or None, keyword-only - original explicit weight column.
        emit_warning : bool, keyword-only - whether the warn policy emits its warning.

        Returns:
        -------
        Corpus - a copied record view, validated analytical frame, collision report,
        and refreshed inclusion flags and exclusion reasons.

        Notes:
        ------
        This is the common construction path for factories and transformations.
        Eligible rows are validated for mass, unit-group purity, and declared parents
        before duplicate handling. Weight validation is repeated after dropping rows.
        Record positions are positional indices into the new record view.
        """
        cls._validate_duplicate_policy(duplicates)
        if len(records) != len(projection):
            raise ValueError("record/projection lineage length mismatch")
        missing_projection = set((*_REQUIRED, "weight", "complete")).difference(
            projection.columns
        )
        if missing_projection:
            raise ValueError(f"projection is missing columns: {sorted(missing_projection)}")

        raw = records.copy(deep=True)
        projected = projection.reset_index(drop=True).copy(deep=True)
        eligible = projected["complete"].fillna(False).astype(bool)
        if not eligible.any():
            raise ValueError("corpus is empty after cleaning")

        analysis = projected.loc[eligible, [*_REQUIRED, "weight"]].copy()
        _validate_weights(analysis["weight"], context="analytical weights")
        analysis["_record_position"] = analysis.index.astype(int)

        # A unit is the inferential entity and therefore cannot carry two labels,
        # even when it appears under more than one source. Declare a composite unit
        # when local sample identifiers repeat across datasets.
        group_counts = analysis.groupby("unit", sort=False, dropna=False)["group"].nunique()
        impure_units = group_counts[group_counts > 1]
        if not impure_units.empty:
            examples = list(impure_units.index[:5])
            raise ValueError(
                "analysis units carry more than one group; declare the correct composite "
                f"unit or repair the labels. Examples: {examples!r}"
            )

        cls._validate_hierarchy(projected, hierarchy)
        duplicate_report = cls._build_duplicate_report(raw, analysis)
        has_duplicates = not duplicate_report.empty
        if has_duplicates:
            n_keys = len(duplicate_report)
            n_excess = int(duplicate_report["excess_rows"].sum())
            examples = [
                (row.source, row.unit, row.group)
                for row in duplicate_report.head(3).itertuples(index=False)
            ]
            message = (
                f"found {n_excess} repeated/duplicate observation row(s) across "
                f"{n_keys} normalized (source, unit, group) key(s); examples: {examples!r}. "
                f"duplicates={duplicates!r}: "
                + (
                    "retaining all analytical rows; choose 'drop' or 'raise' explicitly"
                    if duplicates == "warn"
                    else "policy applied"
                )
            )
            if duplicates == "raise":
                raise ValueError(message)
            if duplicates == "warn" and emit_warning:
                warnings.warn(message, DuplicateObservationWarning, stacklevel=3)

        keep_mask = pd.Series(True, index=analysis.index)
        if has_duplicates and duplicates == "drop":
            duplicate_rows = analysis.duplicated(list(_REQUIRED), keep=False)
            for _, collision in analysis.loc[duplicate_rows].groupby(
                list(_REQUIRED), sort=False, dropna=False
            ):
                if collision["weight"].nunique(dropna=False) > 1:
                    key = tuple(collision.iloc[0][list(_REQUIRED)])
                    raise ValueError(
                        "cannot drop repeated observation key "
                        f"{key!r} with unequal weights; provide an explicit reducer "
                        "before Corpus construction"
                    )
            keep_mask = ~analysis.duplicated(list(_REQUIRED), keep="first")

        retained = analysis.loc[keep_mask].copy()
        if retained.empty:  # pragma: no cover - eligible rows always retain a first copy
            raise ValueError("corpus is empty after duplicate handling")
        _validate_weights(retained["weight"], context="retained analytical weights")

        projected["analysis_included"] = False
        projected["exclusion_reason"] = pd.Series(pd.NA, index=projected.index, dtype="string")
        projected.loc[~eligible, "exclusion_reason"] = "missing_required_key"
        retained_positions = retained["_record_position"].astype(int).tolist()
        projected.loc[retained_positions, "analysis_included"] = True
        if has_duplicates and duplicates == "drop":
            dropped_positions = analysis.loc[~keep_mask, "_record_position"].astype(int).tolist()
            projected.loc[dropped_positions, "exclusion_reason"] = "duplicate_key"

        frame = retained[[*_REQUIRED, "weight"]].reset_index(drop=True)
        frame["weight"] = frame["weight"].astype(float)
        return cls(
            frame=frame,
            name=name,
            weighting=weighting,
            records=raw,
            duplicate_policy=duplicates,
            duplicate_report=duplicate_report.reset_index(drop=True),
            hierarchy=dict(hierarchy),
            source_spec=tuple(source_spec),
            unit_spec=tuple(unit_spec),
            group_spec=tuple(group_spec),
            weight_column=weight_column,
            _record_projection=projected,
        )

    @staticmethod
    def _validate_hierarchy(
        projection: pd.DataFrame,
        hierarchy: Mapping[str, tuple[str, ...]],
    ) -> None:
        """
        Check that each eligible unit has at most one non-missing parent per declared level.

        Missing parents are allowed, including partially mapped units. Conflicting
        non-missing parents or absent projection columns raise ValueError. The check
        does not infer relationships between different parent levels.
        """
        eligible = projection["complete"].fillna(False).astype(bool)
        for level in hierarchy:
            column = f"{_HIERARCHY_PREFIX}{level}"
            if column not in projection:
                raise ValueError(f"projection is missing hierarchy level {level!r}")
            pairs = projection.loc[eligible, ["unit", column]]
            nonmissing = pairs.loc[pairs[column].notna()]
            if nonmissing.empty:
                continue
            counts = nonmissing.groupby("unit", sort=False)[column].nunique()
            conflicts = counts[counts > 1]
            if not conflicts.empty:
                examples = list(conflicts.index[:5])
                raise ValueError(
                    f"analysis units map to multiple non-missing {level!r} parent keys; "
                    f"examples: {examples!r}"
                )

    @staticmethod
    def _build_duplicate_report(
        records: pd.DataFrame, analysis: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Describe repeated analytical keys using their original records.

        Parameters:
        ----------
        records : pd.DataFrame - original records, selected by positional lineage.
        analysis : pd.DataFrame - eligible normalized rows with ``_record_position``.

        Returns:
        -------
        pd.DataFrame - one row per repeated source-unit-group key. ``kind`` is
        ``"exact_record"`` when all original fields agree, otherwise
        ``"projected_key_collision"``. ``varying_columns`` and ``record_positions``
        retain tuples, and ``excess_rows`` counts repetitions beyond the first record.
        """
        duplicate_rows = analysis.duplicated(list(_REQUIRED), keep=False)
        if not duplicate_rows.any():
            return _empty_duplicate_report()

        rows: list[dict[str, object]] = []
        collisions = analysis.loc[duplicate_rows].groupby(
            list(_REQUIRED), sort=False, dropna=False
        )
        for key, collision in collisions:
            key_tuple = key if isinstance(key, tuple) else (key,)
            positions = tuple(collision["_record_position"].astype(int).tolist())
            varying: list[object] = []
            raw_collision = records.iloc[list(positions)]
            for column_position, column in enumerate(records.columns):
                tokens = {
                    _stable_value(value)
                    for value in raw_collision.iloc[:, column_position].tolist()
                }
                if len(tokens) > 1:
                    varying.append(column)
            rows.append(
                {
                    "source": key_tuple[0],
                    "unit": key_tuple[1],
                    "group": key_tuple[2],
                    "n_records": len(positions),
                    "excess_rows": len(positions) - 1,
                    "kind": "exact_record" if not varying else "projected_key_collision",
                    "varying_columns": tuple(varying),
                    "record_positions": positions,
                }
            )
        return pd.DataFrame(rows, columns=list(_DUPLICATE_REPORT_COLUMNS))

    @property
    def sources(self) -> list[Key]:
        """
        Distinct normalized source keys in deterministic order, including zero-weight sources.
        """
        return _sorted_unique(self.frame["source"])

    @property
    def units(self) -> list[Key]:
        """
        Distinct normalized analysis-unit keys in deterministic order.
        """
        return _sorted_unique(self.frame["unit"])

    @property
    def groups(self) -> list[Key]:
        """
        Distinct normalized group keys in deterministic order, including zero-mass groups.
        """
        return _sorted_unique(self.frame["group"])

    @property
    def n_rows(self) -> int:
        """
        Number of retained analytical rows after missing-key and duplicate handling.
        """
        return len(self.frame)

    @property
    def n_records(self) -> int:
        """
        Number of preserved input records, including records excluded from analysis.
        """
        return len(self.records) if self.records is not None else len(self.frame)

    @property
    def n_sources(self) -> int:
        """
        Number of distinct sources represented in the analytical frame.
        """
        return int(self.frame["source"].nunique())

    @property
    def n_units(self) -> int:
        """
        Number of distinct declared analysis units, without weighting by repeated records.
        """
        return int(self.frame["unit"].nunique())

    @property
    def n_groups(self) -> int:
        """
        Number of group labels represented in analytical rows, regardless of their mass.
        """
        return int(self.frame["group"].nunique())

    @property
    def total_weight(self) -> float:
        """
        Sum of retained analytical row weights, in the caller's declared mass units.
        """
        return float(self.frame["weight"].sum())

    @property
    def n_duplicate_keys(self) -> int:
        """
        Number of repeated analytical keys diagnosed before duplicate dropping.
        """
        return len(self.duplicate_report)

    @property
    def n_duplicate_rows(self) -> int:
        """
        Total excess records beyond the first in each diagnosed repeated analytical key.

        This remains a diagnostic count when the drop policy excludes those rows.
        """
        if self.duplicate_report.empty:
            return 0
        return int(self.duplicate_report["excess_rows"].sum())

    @property
    def incidence(self) -> pd.DataFrame:
        """
        Binary source x unit membership with canonical keys on both axes.

        Returns:
        -------
        pd.DataFrame, shape (n_sources, n_units) - integer 1 for any recorded
        source-unit connection and 0 otherwise, ordered by ``sources`` and ``units``.

        Notes:
        ------
        Repeated rows count once. Zero-weight observations still establish membership.
        Tuple-valued keys remain single axis labels rather than a MultiIndex.
        """
        matrix = pd.crosstab(
            index=[self.frame["source"]],
            columns=[self.frame["unit"]],
        )
        matrix = matrix.reindex(
            index=_as_index(self.sources), columns=_as_index(self.units)
        ).fillna(0)
        matrix.index.name = "source"
        matrix.columns.name = "unit"
        return (matrix > 0).astype(int)

    @property
    def source_group_weight(self) -> pd.DataFrame:
        """
        Weighted source x group table used by entropy and ceiling calculations.

        Returns:
        -------
        pd.DataFrame, shape (n_sources, n_groups) - summed retained row mass for
        each source-group pair, with absent pairs filled by zero and axes in canonical
        source/group order.

        Notes:
        ------
        Each retained repeated record contributes its weight. Sources and groups with
        zero total mass remain represented, so structural membership and positive-mass
        crossing need not have the same counts.
        """
        table = pd.crosstab(
            index=[self.frame["source"]],
            columns=[self.frame["group"]],
            values=self.frame["weight"],
            aggfunc="sum",
        )
        table = table.reindex(
            index=_as_index(self.sources), columns=_as_index(self.groups)
        ).fillna(0.0)
        table.index.name = "source"
        table.columns.name = "group"
        return table

    @property
    def unit_group(self) -> pd.Series:
        """
        The group label of each analysis unit, in canonical unit order.

        Returns:
        -------
        pd.Series - group labels named ``group`` and indexed by unit.

        Raises:
        ------
        ValueError
        - if a unit carries multiple groups, for example after external DataFrame mutation.
        """
        counts = self.frame.groupby("unit", sort=False)["group"].nunique()
        if (counts > 1).any():  # defensive for externally-mutated dataframes
            bad = list(counts[counts > 1].index[:5])
            raise ValueError(f"units carry more than one group: {bad!r}")
        unique = self.frame.drop_duplicates("unit")
        result = pd.Series(
            unique["group"].to_numpy(),
            index=_as_index(unique["unit"].tolist()),
            name="group",
        )
        result.index.name = "unit"
        return result.reindex(_as_index(self.units))

    def spanning_sources(self) -> list[Key]:
        """
        Return sources whose retained observations include more than one group.

        Uses membership irrespective of weight. A structurally spanning source need
        not have positive entropy if all but one of its groups carry zero mass.
        """
        counts = self.frame.groupby("source", sort=False)["group"].nunique()
        return _sorted_unique(pd.Series(counts[counts > 1].index, dtype=object))

    def units_per_source(self) -> pd.Series:
        """
        Distinct-unit counts from binary incidence, indexed by source.

        Repeated rows and row mass do not change these counts.
        """
        return self.incidence.sum(axis=1).rename("units_per_source")

    def sources_per_unit(self) -> pd.Series:
        """
        Distinct-source counts from binary incidence, indexed by analysis unit.
        """
        return self.incidence.sum(axis=0).rename("sources_per_unit")

    def hierarchy_report(self) -> pd.DataFrame:
        """
        Summarize declared parent mappings without counting assay rows as extra children.

        Returns:
        -------
        pd.DataFrame - indexed by hierarchy level, with its ``columns``, counts of
        units and parents, units with or without parents, records missing parents,
        partially mapped units, parents containing multiple units, and maximum units
        per parent. No declared hierarchy gives an empty table with the same schema.

        Notes:
        ------
        The report uses records with complete analytical keys, including repeated keys
        later dropped from analysis. Each distinct child contributes once to parent
        size. A unit with both known and missing parent records is partially mapped,
        while a unit with no known parent is missing its parent. Metadata is not imputed.
        """
        if not self.hierarchy:
            empty = pd.DataFrame(columns=list(_HIERARCHY_REPORT_COLUMNS))
            empty.index = pd.Index([], name="level")
            return empty
        assert self._record_projection is not None
        projection = self._record_projection
        eligible = projection["complete"].fillna(False).astype(bool)
        rows: list[dict[str, object]] = []
        for level, spec in self.hierarchy.items():
            column = f"{_HIERARCHY_PREFIX}{level}"
            pairs = projection.loc[eligible, ["unit", column]].copy()
            units = list(pd.unique(pairs["unit"]))
            parents_by_unit: dict[object, set[object]] = {}
            partial_units = 0
            for unit, unit_rows in pairs.groupby("unit", sort=False, dropna=False):
                parents = set(unit_rows.loc[unit_rows[column].notna(), column].tolist())
                parents_by_unit[unit] = parents
                if parents and unit_rows[column].isna().any():
                    partial_units += 1
            unit_parent_pairs = {
                (unit, next(iter(parents)))
                for unit, parents in parents_by_unit.items()
                if parents
            }
            parent_to_units: dict[object, set[object]] = {}
            for unit, parent in unit_parent_pairs:
                parent_to_units.setdefault(parent, set()).add(unit)
            parent_sizes = [len(child_units) for child_units in parent_to_units.values()]
            missing_units = sum(not parents for parents in parents_by_unit.values())
            rows.append(
                {
                    "level": level,
                    "columns": tuple(spec),
                    "n_units": len(units),
                    "n_parents": len(parent_to_units),
                    "n_units_with_parent": len(units) - missing_units,
                    "n_units_missing_parent": missing_units,
                    "n_records_missing_parent": int(pairs[column].isna().sum()),
                    "n_partially_mapped_units": partial_units,
                    "n_multi_unit_parents": sum(size > 1 for size in parent_sizes),
                    "max_units_per_parent": max(parent_sizes, default=0),
                }
            )
        return pd.DataFrame(rows).set_index("level")[list(_HIERARCHY_REPORT_COLUMNS)]

    def at_unit(
        self,
        level: str,
        *,
        duplicates: DuplicatePolicy = "drop",
        name: str | None = None,
    ) -> "Corpus":
        """
        Reproject the analysis at an explicitly declared parent level.

        Parameters:
        ----------
        level : str - a key in ``hierarchy``, such as a declared donor level.
        duplicates : str, default "drop", keyword-only - policy for collisions after
        child units become parent units. Dropping retains one equal-weight observation
        per source-parent-group key.
        name : str or None, default None, keyword-only - output label; defaults to
        the original name followed by ``@level``.

        Returns:
        -------
        Corpus - the same preserved records reprojected at the parent unit grain.

        Raises:
        ------
        KeyError
        - if the parent level was not declared.
        ValueError
        - if no eligible parent rows remain, a parent carries multiple groups,
        hierarchy constraints conflict, or the requested duplicate policy fails.

        Notes:
        ------
        Missing parents remain missing and are excluded from analysis rather than
        combined into one unknown unit. Unequal weights cannot be silently dropped;
        choose and apply any required scientific reducer before construction.
        This operation changes the analysis unit, not the source grain.
        """
        if level not in self.hierarchy:
            raise KeyError(
                f"hierarchy level {level!r} is not declared; have {list(self.hierarchy)}"
            )
        assert self.records is not None and self._record_projection is not None
        projection = self._record_projection.copy(deep=True)
        parent_column = f"{_HIERARCHY_PREFIX}{level}"
        projection["unit"] = projection[parent_column]
        projection["complete"] = projection[list(_REQUIRED)].notna().all(axis=1)
        return type(self)._from_projection(
            records=self.records,
            projection=projection,
            name=name or f"{self.name}@{level}",
            weighting=self.weighting,
            duplicates=duplicates,
            hierarchy=self.hierarchy,
            source_spec=self.source_spec,
            unit_spec=self.hierarchy[level],
            group_spec=self.group_spec,
            weight_column=self.weight_column,
            emit_warning=True,
        )

    def subset(
        self,
        *,
        sources: Sequence[object] | None = None,
        units: Sequence[object] | None = None,
        groups: Sequence[object] | None = None,
        name: str | None = None,
    ) -> "Corpus":
        """
        Restrict analytical observations and their record lineage to selected keys.

        Parameters:
        ----------
        sources, units, groups : sequences or None, default None, keyword-only -
        canonical keys to retain. Supplied selections are combined by intersection;
        None leaves that role unrestricted. Wrap a composite tuple key in a list.
        name : str or None, default None, keyword-only - output name; defaults to
        the original name followed by ``[subset]``.

        Returns:
        -------
        Corpus - selected records and a rebuilt analytical projection under the
        existing duplicate policy, hierarchy, and weighting.

        Raises:
        ------
        ValueError
        - if no complete analytical rows remain or their total retained mass is zero.

        Notes:
        ------
        Selections apply to normalized projected keys, not raw display spellings or
        DataFrame index labels. The operation preserves relevant excluded records
        when their projected keys satisfy the same selection.
        """
        assert self.records is not None and self._record_projection is not None
        projection = self._record_projection
        keep = pd.Series(True, index=projection.index)
        if sources is not None:
            keep &= projection["source"].isin(_selection(sources))
        if units is not None:
            keep &= projection["unit"].isin(_selection(units))
        if groups is not None:
            keep &= projection["group"].isin(_selection(groups))
        if not (keep & projection["complete"].fillna(False)).any():
            raise ValueError("subset is empty")
        positions = np.flatnonzero(keep.to_numpy())
        raw = self.records.iloc[positions]
        projected = projection.iloc[positions].reset_index(drop=True).copy(deep=True)
        return type(self)._from_projection(
            records=raw,
            projection=projected,
            name=name or f"{self.name}[subset]",
            weighting=self.weighting,
            duplicates=self.duplicate_policy,
            hierarchy=self.hierarchy,
            source_spec=self.source_spec,
            unit_spec=self.unit_spec,
            group_spec=self.group_spec,
            weight_column=self.weight_column,
            emit_warning=False,
        )

    def crossed_subcorpus(self, name: str | None = None) -> "Corpus":
        """
        Restrict the corpus to sources represented in more than one group.

        Parameters:
        ----------
        name : str or None, default None - output label, otherwise ``name[crossed]``.

        Returns:
        -------
        Corpus - the subset defined by :meth:`spanning_sources`.

        Raises:
        ------
        ValueError
        - if no sources span groups or the resulting subset has no valid positive mass.

        Notes:
        ------
        Selection uses structural membership, including zero-weight observations.
        The crossed subset's entropy must be evaluated under its retained weights.
        """
        spanning = self.spanning_sources()
        if not spanning:
            raise ValueError(f"{self.name!r} has no source contributing to more than one group")
        return self.subset(sources=spanning, name=name or f"{self.name}[crossed]")

    def regroup(self, mapping: Mapping[object, object] | pd.Series, name: str | None = None) -> "Corpus":
        """
        Map existing group keys to a new contrast while retaining contributing raw labels.

        Parameters:
        ----------
        mapping : mapping or pd.Series - old normalized group key to new group label.
        Only groups present as keys are retained; several old groups may map to one label.
        name : str or None, default None - output name, otherwise ``name[regrouped]``.

        Returns:
        -------
        Corpus - selected records and a projection carrying the mapped group values,
        with the existing source/unit definitions, weights, hierarchy, and duplicate policy.

        Raises:
        ------
        ValueError
        - if every eligible row is removed or rebuilding violates corpus constraints.

        Notes:
        ------
        Mapping values are assigned directly, without the text normalization performed
        by ``from_frame``. Supply consistent usable group keys. Missing mapped labels
        remain in the retained record lineage but do not enter the analytical frame.
        """
        labels = dict(mapping) if not isinstance(mapping, pd.Series) else mapping.to_dict()
        assert self.records is not None and self._record_projection is not None
        projection = self._record_projection
        keep = projection["group"].isin(labels)
        if not (keep & projection["complete"].fillna(False)).any():
            raise ValueError("regroup removed every row")
        positions = np.flatnonzero(keep.to_numpy())
        raw = self.records.iloc[positions]
        projected = projection.iloc[positions].reset_index(drop=True).copy(deep=True)
        projected["group"] = projected["group"].map(labels)
        projected["complete"] = projected[list(_REQUIRED)].notna().all(axis=1)
        return type(self)._from_projection(
            records=raw,
            projection=projected,
            name=name or f"{self.name}[regrouped]",
            weighting=self.weighting,
            duplicates=self.duplicate_policy,
            hierarchy=self.hierarchy,
            source_spec=self.source_spec,
            unit_spec=self.unit_spec,
            group_spec=self.group_spec,
            weight_column=self.weight_column,
            emit_warning=False,
        )

    def with_source(
        self, df: pd.DataFrame, column: KeySpec, name: str | None = None
    ) -> "Corpus":
        """
        Replace the source grain using metadata aligned to retained analytical rows.

        Parameters:
        ----------
        df : pd.DataFrame - exactly one metadata row per row of ``frame``, in the
        same positional order. Its index is not used to align observations.
        column : str or sequence of str - columns defining the new source key.
        name : str or None, default None - output label; otherwise derived from the
        original name and the new source columns.

        Returns:
        -------
        Corpus - original records with reprojected sources and refreshed duplicate
        diagnostics under the existing policy. Unit labels, groups, and row weights
        are preserved through the record projection.

        Raises:
        ------
        KeyError
        - if a requested source column is absent.
        ValueError
        - if lengths differ, source keys are missing, lineage cannot be aligned, or
        new collisions violate the duplicate policy.

        Notes:
        ------
        Excluded duplicate records inherit the replacement for their retained key
        only when that mapping is unique. An ambiguous mapping raises. The caller
        must guarantee positional alignment; equal row counts cannot verify it.
        Supplied source metadata updates the projection rather than replacing the
        preserved raw records.
        """
        if len(df) != len(self.frame):
            raise ValueError(f"frame length {len(df)} != corpus length {len(self.frame)}")
        source_spec = _normalise_spec(column, role="source")
        missing = _missing_columns(df, [source_spec])
        if missing:
            raise KeyError(f"columns not found in frame: {missing}; have {list(df.columns)}")
        new_sources = _normalise_key(df, source_spec)
        if new_sources.isna().any():
            raise ValueError("new source grain has missing keys")

        assert self.records is not None and self._record_projection is not None
        projection = self._record_projection.copy(deep=True)
        projection["source"] = projection["source"].astype(object)
        included_positions = projection.index[projection["analysis_included"].astype(bool)].tolist()
        if len(included_positions) != len(new_sources):
            raise ValueError("analytical frame and record lineage are out of alignment")

        old_keys: dict[int, tuple[object, object, object]] = {
            position: tuple(projection.loc[position, list(_REQUIRED)])
            for position in projection.index
        }
        replacement_by_key: dict[tuple[object, object, object], set[object]] = {}
        for ordinal, position in enumerate(included_positions):
            replacement = new_sources.iloc[ordinal]
            projection.at[position, "source"] = replacement
            replacement_by_key.setdefault(old_keys[position], set()).add(replacement)

        # Rows excluded only because of duplicate dropping belong to the retained
        # analytical key and inherit that explicitly supplied source. Invalid raw
        # rows remain invalid and are never guessed into a grain.
        for position in projection.index[~projection["analysis_included"].astype(bool)]:
            replacements = replacement_by_key.get(old_keys[position], set())
            if len(replacements) == 1:
                projection.at[position, "source"] = next(iter(replacements))
            elif len(replacements) > 1:
                raise ValueError(
                    "cannot align excluded duplicate records to multiple new source keys"
                )
        projection["complete"] = projection[list(_REQUIRED)].notna().all(axis=1)
        label = "+".join(source_spec)
        return type(self)._from_projection(
            records=self.records,
            projection=projection,
            name=name or f"{self.name}@{label}",
            weighting=self.weighting,
            duplicates=self.duplicate_policy,
            hierarchy=self.hierarchy,
            source_spec=source_spec,
            unit_spec=self.unit_spec,
            group_spec=self.group_spec,
            weight_column=self.weight_column,
            emit_warning=True,
        )

    def describe(self) -> pd.Series:
        """
        Return the compact design context needed beside a diagnostic result.

        Returns:
        -------
        pd.Series - corpus and weighting labels; record, analytical-row, source,
        unit, and group counts; duplicate policy and counts; declared hierarchy levels;
        median and maximum units per source and sources per unit; units with one source;
        and the number of structurally spanning sources.

        Notes:
        ------
        Support counts use binary incidence, not row mass. This is a description of
        declared membership and data handling, not an estimate of effective sample size.
        """
        units_per_source = self.units_per_source()
        sources_per_unit = self.sources_per_unit()
        hierarchy = self.hierarchy_report()
        return pd.Series(
            {
                "name": self.name,
                "weighting": self.weighting,
                "n_records": self.n_records,
                "n_rows": self.n_rows,
                "n_sources": self.n_sources,
                "n_units": self.n_units,
                "n_groups": self.n_groups,
                "duplicate_policy": self.duplicate_policy,
                "n_duplicate_keys": self.n_duplicate_keys,
                "n_duplicate_rows": self.n_duplicate_rows,
                "hierarchy_levels": tuple(hierarchy.index),
                "units_per_source_median": float(units_per_source.median()),
                "units_per_source_max": int(units_per_source.max()),
                "sources_per_unit_median": float(sources_per_unit.median()),
                "sources_per_unit_max": int(sources_per_unit.max()),
                "units_with_one_source": int((sources_per_unit == 1).sum()),
                "sources_spanning_groups": len(self.spanning_sources()),
            }
        )

    def __repr__(self) -> str:
        """
        Compact representation of analysis size, retained records, weighting, and duplicates.
        """
        duplicate_note = (
            f", {self.n_duplicate_rows} duplicate row(s), policy={self.duplicate_policy!r}"
            if self.n_duplicate_rows
            else ""
        )
        return (
            f"Corpus({self.name!r}: {self.n_rows} analysis rows/{self.n_records} records, "
            f"{self.n_sources} sources, {self.n_units} units, {self.n_groups} groups, "
            f"{self.weighting}{duplicate_note})"
        )
