# Maintainer release procedure

Current development version: `0.1.0.dev0`. Do not change it merely to prepare
this repository for review. Public release is blocked on the metadata below.

## Before the first public release

- Obtain Louis's literal MIT `LICENSE` with approved copyright wording. Add
  `license = "MIT"` and `license-files = ["LICENSE"]` to `[project]` in
  `pyproject.toml`; add `include LICENSE` to `MANIFEST.in`.
- Obtain citation authors and order as listed in
  [CITATION_METADATA_NEEDED.md](CITATION_METADATA_NEEDED.md). Create and validate
  `CITATION.cff` (`python -m pip install cffconvert`, `cffconvert --validate`),
  add `include CITATION.cff` to `MANIFEST.in`, and replace pending README notices
  with links to the approved metadata. DOI/preferred citation are optional.
- Configure a protected GitHub Environment named `pypi`: required maintainer
  reviewers, prevention of self-review where available, and release-tag deployment
  restrictions. Do not enable publishing without these protections.
- In PyPI, configure an existing-project Trusted Publisher or a pending publisher:
  project `provenance-aliasing-diagnostics`, owner `UCLAbouchardlab`, repository
  `provenance-aliasing-diagnostics`, workflow `release.yml`, environment `pypi`.
  This uses OIDC; do not add a long-lived PyPI API token.
- Only after approval and configuration, set the GitHub **repository** Actions
  variable `PYPI_PUBLISH_ENABLED` to `true`. With this variable absent or any other
  value, the publishing job is skipped. No publication is attempted by ordinary
  pushes or pull requests. TestPyPI requires its own publisher/environment and
  explicit authorization; it is not automatically used here.

## Each approved release

1. Choose the version, update `pyproject.toml` and `CITATION.cff` consistently,
   and update versioned installation examples. The runtime version is read from
   installed distribution metadata; there is no second source version constant.
2. Review license/citation metadata. Run `python -m pytest` with the development
   and Excel extras; require all six OS/Python 3.11/3.12 jobs and packaging checks.
3. Remove only generated `build/`, `dist/`, and `*.egg-info/` outputs in this
   checkout, then run:

   ```bash
   python -m build
   python -m twine check dist/*
   python scripts/check_dist.py
   ```

4. Create a fresh virtual environment **outside the checkout**. Install the built
   wheel without extras. Change to a directory outside the checkout and run its
   Python with the absolute path to `scripts/smoke_wheel.py`. It checks site-packages
   imports, version, API, SciPy, every CLI entry point and command, output protection,
   and verification mismatch behavior. Then install the same wheel with `[excel]`
   and run that script with `--excel`. Do not reuse an environment with extras for
   the base-wheel check.
5. Inspect the sdist/wheel member lists and metadata. Require docs and all synthetic
   test fixtures in the sdist, approved LICENSE/citation, and only runtime modules
   plus distribution metadata in the wheel. Retain the CI artifact for review.
6. Merge reviewed changes only with maintainer approval. Tag that approved commit
   as `v<version>` and create/publish the corresponding GitHub Release.
7. `release.yml` runs the same complete test and packaging workflow, building the
   sdist/wheel **once**. It validates citation/license/version, then downloads those
   exact artifacts in the protected publishing job. Only that job has
   `id-token: write`; other jobs have read-only repository access.
8. Approve the protected environment deployment when ready to publish. After
   publication, install the exact version from PyPI in a new environment, verify
   its version and site-packages import, and rerun the installed-wheel smoke checks
   against the matching source checkout. Inspect PyPI metadata and artifacts.

Workflow setup follows the [PyPA release guide](https://packaging.python.org/en/latest/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/)
and [PyPI Trusted Publishing documentation](https://docs.pypi.org/trusted-publishers/using-a-publisher/).
