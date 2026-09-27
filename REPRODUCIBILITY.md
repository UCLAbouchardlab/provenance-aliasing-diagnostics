# Reproducibility

## Environment

The package supports Python 3.11 and 3.12. Create a clean environment and install the package and test dependencies from the repository:

```bash
python -m venv .venv
```

Activate it with `source .venv/bin/activate` on POSIX or
`.venv\Scripts\Activate.ps1` in PowerShell, then run:

```bash
python -m pip install ".[dev,excel]"
python -m pytest
```

NumPy, pandas, and SciPy are runtime dependencies. The `excel` extra installs the engine used for `.xlsx` input. Exact installed versions are recorded by normal environment tools such as `python -m pip freeze`.

## Authoritative examples

The small, synthetic, cross-omics inputs in `tests/fixtures/synthetic/` are the repository's authoritative examples. Their expected diagnostic values are in `expected.json`. These fixtures are synthetic test data, not manuscript data.

## Determinism and randomness

Input validation, configuration resolution, fingerprints, and diagnostics without sampling are deterministic for the same inputs and dependency behavior. Randomized routines use local NumPy generators rather than the process-wide random state. The default diagnostic seed is `0`; a supplied seed makes bootstrap and sampled operations reproducible for the same software environment and arguments.

Stochastic operations include requested bootstrap resampling and sampled simulation or calibration paths. Exhaustive paths are deterministic. Reproducibility across materially different dependency versions or numerical platforms is not guaranteed beyond the tested expectations.

## Run records and verification

The CLI can write a run-record JSON beside a validation or diagnostic report. It records the resolved configuration, command and exit code, package/Python/NumPy/pandas versions, exact input byte sizes and SHA-256 fingerprints, and a fingerprint of the JSON report.

`provenance-aliasing verify` checks whether the supplied input bytes, configuration, and optional report match that record. A matching checksum establishes content identity with the recorded bytes. It does **not** establish data authenticity, scientific validity, correctness of the analysis, or reproduction of the calculation; verification does not rerun diagnostics. Environment differences are reported separately.

Metadata and configuration-file hashes cover exact bytes. The resolved
configuration and report hashes cover canonical JSON content, so report whitespace
and object-key ordering do not change the report fingerprint. Omitting `--report`
leaves report content explicitly unchecked. The run record does not include SciPy
or optional adapter dependency versions; retain `python -m pip freeze` alongside
records when those APIs are used or a complete environment inventory is needed.

## Expected outputs and tolerances

Validation produces a structured validation report. Diagnosis produces validation information, metric rows, result tables, an audit table, and execution-status records. Expected unavailable states are represented explicitly, including `undefined`, `not_applicable`, `not_requested`, and `resource_limited`.

Synthetic acceptance expectations use an absolute tolerance of `1e-12` and zero relative tolerance where currently asserted. Individual algorithms may document other validation tolerances in their API docstrings; those are not replaced by the synthetic acceptance tolerance.

## Manuscript boundary

This repository documents reproducibility of the package API, CLI, synthetic fixtures, and run-record identity checks. It does not currently contain or claim a complete manuscript-analysis or figure-reproduction workflow. Such a workflow would require separately documented authoritative data, environment, commands, and expected artifacts.
