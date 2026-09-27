# Provenance Aliasing Diagnostics

`provenance-aliasing-diagnostics` checks observation metadata for relationships between provenance sources and group labels. It validates an explicitly mapped table and reports design, entropy, ceiling, structure, and guard diagnostics. It assesses metadata rather than assay measurements and does not determine scientific validity on its own.

Python 3.11 and 3.12 are supported.

## Installation

From a clone of this repository:

```bash
pip install .
```

For development and tests:

```bash
pip install ".[dev,excel]"
python -m pytest
```

The `excel` extra is needed only for Excel input through the adapter API:
`pip install ".[excel]"`. NumPy, pandas, and SciPy are installed by the base package.
To install a built wheel, use `pip install path/to/provenance_aliasing_diagnostics-0.1.0.dev0-py3-none-any.whl`.

## Command line

The same operational commands are available through `provenance-aliasing`, `python -m provenance_aliasing`, and `python -m provenance_aliasing.api`.
Run `provenance-aliasing --help` or append `--help` to any command for options.
The following working example uses files shipped in the repository and source archive;
run it from the repository root. Wheels contain the runtime package only.

```bash
provenance-aliasing validate tests/fixtures/synthetic/bulk-rna-seq.csv \
  --config tests/fixtures/synthetic/bulk-rna-seq.mapping.json \
  --output validation.json

provenance-aliasing diagnose tests/fixtures/synthetic/bulk-rna-seq.csv \
  --config tests/fixtures/synthetic/bulk-rna-seq.mapping.json \
  --output diagnostics.json --run-record run-record.json

provenance-aliasing verify run-record.json \
  --metadata tests/fixtures/synthetic/bulk-rna-seq.csv \
  --config tests/fixtures/synthetic/bulk-rna-seq.mapping.json \
  --report diagnostics.json --output verification.json
```

`validate` checks the mapped metadata without running diagnostics. `diagnose` validates and runs the selected diagnostics. `verify` compares supplied inputs and an optional report with a saved run record; it does not rerun the analysis.

Inputs are UTF-8 CSV or TSV observation metadata and a JSON configuration. The
mapping explicitly names source, analysis-unit, and group columns, the provenance
grain, and the unit description; it can also specify hierarchy, weights, missing
value/duplicate policies, diagnostic selection, seed, and resource budget. See
the [complete synthetic mapping](tests/fixtures/synthetic/bulk-rna-seq.mapping.json).
Identifiers are read as text, preserving leading zeros. Assay measurements are
not required. Excel is supported through `provenance_aliasing.adapters.from_excel`,
not directly by these CLI commands.

Outputs are JSON: a validation report with issues; a diagnostic report with
metrics, tables, audit information and execution statuses; and a verification
report with matching/mismatching checks. `--run-record` adds a provenance JSON.
Without `--output`, JSON goes to stdout; summaries and errors go to stderr.
Existing files require `--overwrite`; input files are protected from replacement.
Exit codes are `0` for success, `2` for invalid input/usage or verification mismatch,
`3` for computation/output failure, and `4` for a resource-limited partial diagnosis.

## Python API

```python
import pandas as pd
from provenance_aliasing.api import AnalysisConfig, diagnose, validate

metadata = pd.read_csv(
    "tests/fixtures/synthetic/bulk-rna-seq.csv",
    dtype=str,
    keep_default_na=False,
)
config = AnalysisConfig.from_json(
    "tests/fixtures/synthetic/bulk-rna-seq.mapping.json"
)
validation = validate(metadata, config=config)
assert validation.valid
result = diagnose(metadata, config=config)
print(result.to_json())
```

`provenance_aliasing.api` explicitly exports configuration, validation, diagnostic,
and run-record/result types. Configuration components such as `ColumnMapping`,
`ProvenanceGrain`, and `DiagnosticSelection` live in
`provenance_aliasing.api.config`. For file-based API workflows use `run_analysis`
and `verify_run` from `provenance_aliasing.api`. Numerical primitives are available
from modules such as `provenance_aliasing.metrics`. Use the metadata API's
`diagnose` above; the root package's `diagnose` is the lower-level corpus report API.
The current development version is `0.1.0.dev0`.

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for seeds, fingerprints, expected outputs, and numerical tolerances.

## Scope boundary

The package API and CLI provide metadata diagnostics. They are not automatically a manuscript or figure-reproduction bundle; no such workflow is claimed unless it is separately documented with its authoritative inputs and environment.

## Citation

`CITATION.cff` is pending an authoritative author list in approved order.
See [CITATION_METADATA_NEEDED.md](CITATION_METADATA_NEEDED.md). No DOI or preferred
paper citation is currently established for this software.

## License

MIT is selected. Louis must supply the actual `LICENSE`, including the approved
copyright line. Until that file is supplied and wired into the package metadata,
public release remains blocked; this repository does not yet supply a license grant.
