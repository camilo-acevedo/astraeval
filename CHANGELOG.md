# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.6.0] - 2026-05-04

Pre-1.0 API audit pass. Tightens public surface, fixes naming inconsistencies,
and unblocks `python -m astraea`.

### Added

- Top-level `astraea` package now re-exports the core composition contract:
  `EvalRun`, `RunResult`, `SampleResult`, `RunManifest`, `Sample`, `Response`,
  `Provider`, `Metric`, `MetricResult`, and the full exception hierarchy. A
  short example can now be written without learning subpackage paths.
- `python -m astraea` now works alongside the `astraea` console script,
  delegating to the same entry point.

### Changed

- **Breaking.** `astraea.providers.base.request_key` renamed to `hash_request`
  for consistency with `astraea.core.manifest.hash_dataset` and `hash_params`.
  Update imports: `from astraea.providers import hash_request`.
- **Breaking.** `astraea.metrics.llm_judge.LLMJudge.ask` now returns the full
  `Response` (was: `str`). Call sites that only need the text should use
  `judge.ask(prompt).text`. The change lets downstream tooling capture token
  counts and latency for judge calls, matching the audit-trail goals of the
  rest of the harness.

### Removed

- Public alias `astraea.reports.json_report.run_subdir_name`. The function was
  not in `__all__` and duplicated the private implementation. The directory
  layout is documented on `write_run` and surfaced via its return value.

## [0.5.0] - 2026-05-04

Project renamed from `llm-evals` to `astraea`. See
[`feat!: rename package and CLI to astraea`](https://github.com/camilo-acevedo/astraea/commits/main)
for the full migration. Functionality is unchanged from 0.4.0.

## [0.4.0] and earlier

Early development under the previous `llm-evals` name. See git history.
