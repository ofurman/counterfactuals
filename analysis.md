Branch Comparison: lukasz/diversity-comparison vs develop

Shared history

Both branches share the same base and ~45 commits with identical messages (cherry-picked/rebased). The vast majority of work is the same on both sides.

Unique to lukasz/diversity-comparison (your branch)

- 4 recent fixes not yet on develop:
  - a821e19 - metrics calculation fix
  - 067028c - handling failed CF search in numerical features
  - e9a011f - one-hot categorical feature handling for ARES and GlobeCE
  - 0b6c8c3 - handling single dim matrices
- Docs/meta differences: different docs deploy approach (3 commits), Apache-2.0 license commit, authors fix, dicoflex module name fix
- 2 unique files: scripts/generate_latex_tables.py, scripts/generate_latex_tables_single_dataset.py

Unique to develop

- 01f70ee - restore main README and docs (likely overwrites/resets docs content)
- b1ce15f - Documentation autodeploy (#52) - different CI/CD approach for docs
- 5a3acbb - PyPI publishing setup as ce-library (#54) - adds __init__.py, publish.yml, pre-commit-config changes

Why so many conflicts

The branches diverged and then had the same changes applied independently (cherry-picks), but with slightly different merge histories. The 3 develop-only commits (README restore, autodeploy, PyPI publishing) touch
  many of the same files, causing conflicts especially in:
- Docs (12 conflicts) - develop restored/rewrote docs
- Pipelines (12 conflicts) - likely formatting/import differences from the PyPI refactor
- Config/meta (pyproject.toml, mkdocs.yml, LICENSE, README) - both sides made packaging/docs changes

Recommendation

The simplest approach: rebase your 4 unique fix commits onto develop rather than merging, since the bulk of history is shared. Something like:

git checkout develop
git checkout -b lukasz/diversity-comparison-v2
git cherry-pick a821e19 067028c e9a011f 0b6c8c3

This avoids dealing with 30+ conflict files that are just duplicated history clashing with itself.
