# Maintainer metadata required before public release

No explicit software-author list or canonical manuscript-author list is stored
in the repository. Git commit order is not authorship authority.

Supply the software citation authors in approved order, with each author's given
and family names (or an explicit organization name). Supply package author and/or
maintainer attribution and any public contact details if those fields are desired.
ORCID identifiers are optional and must be provided, not inferred.

The confirmed citation fields are:

```yaml
cff-version: 1.2.0
title: provenance-aliasing-diagnostics
version: 0.1.0.dev0
repository-code: https://github.com/UCLAbouchardlab/provenance-aliasing-diagnostics
license: MIT
```

After author approval, create `CITATION.cff` with these fields, a citation message,
and the approved authors. Validate with `cffconvert --validate`. Omit DOI and
`preferred-citation` unless maintainers supply authoritative values; their absence
is not a blocker.

MIT is selected, but Louis must supply the literal `LICENSE` and copyright line.
Do not infer ownership or the copyright year. When supplied, set
`license = "MIT"` and `license-files = ["LICENSE"]` in `[project]`, include both
final metadata files in the sdist, and rebuild/inspect both distributions.
