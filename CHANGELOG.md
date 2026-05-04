# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.6.0] - 2026-05-04

First public PyPI release. Pre-1.0 API audit pass: tightens public surface,
fixes naming inconsistencies, unblocks `python -m astraeval`, and renames the
distribution from `astraea` to `astraeval` because the former name was already
reserved on PyPI.

### Added

- Top-level `astraeval` package now re-exports the core composition contract:
  `EvalRun`, `RunResult`, `SampleResult`, `RunManifest`, `Sample`, `Response`,
  `Provider`, `Metric`, `MetricResult`, and the full exception hierarchy. A
  short example can now be written without learning subpackage paths.
- `python -m astraeval` now works alongside the `astraeval` console script,
  delegating to the same entry point.

### Changed

- **Breaking.** Distribution and import name renamed from `astraea` to
  `astraeval`. Replace `pip install astraea` with `pip install astraeval`,
  `from astraea import ...` with `from astraeval import ...`, and the CLI
  invocation `astraea run config.yaml` with `astraeval run config.yaml`. The
  `astraea` name was reserved on PyPI by an unrelated account; `astraeval`
  is the published distribution.
- **Breaking.** `astraeval.providers.base.request_key` renamed to
  `hash_request` for consistency with `astraeval.core.manifest.hash_dataset`
  and `hash_params`. Update imports:
  `from astraeval.providers import hash_request`.
- **Breaking.** `astraeval.metrics.llm_judge.LLMJudge.ask` now returns the
  full `Response` (was: `str`). Call sites that only need the text should
  use `judge.ask(prompt).text`. The change lets downstream tooling capture
  token counts and latency for judge calls, matching the audit-trail goals
  of the rest of the harness.

### Removed

- Public alias `astraeval.reports.json_report.run_subdir_name`. The function
  was not in `__all__` and duplicated the private implementation. The
  directory layout is documented on `write_run` and surfaced via its return
  value.

## [0.5.0] - 2026-05-04 (unreleased on PyPI)

Project renamed from `llm-evals` to `astraea`. Never published to PyPI
because the `astraea` name was reserved; see 0.6.0 for the follow-up rename
to `astraeval`. Functionality unchanged from 0.4.0.

## [0.4.0] and earlier

Early development under the previous `llm-evals` name. See git history.
