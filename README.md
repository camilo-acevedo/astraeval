# llm-evals

> Reproducible LLM evaluation harness with run manifests, prompt hashing, and SQLite caching.

[![CI](https://github.com/bryamacevedo/llm-evals/actions/workflows/ci.yml/badge.svg)](https://github.com/bryamacevedo/llm-evals/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> **Status:** pre-alpha (v0.0.1) — Phase 0 scaffolding. APIs will change.

---

## Why another LLM eval harness?

Most evaluation libraries (DeepEval, Ragas, promptfoo) are excellent at scoring
models but treat the run itself as ephemeral. `llm-evals` is built for teams
that need to **audit and reproduce** scores months later:

- **Run manifests** — every evaluation emits a `run.manifest.json` with hashes
  of prompts, dataset content, and pinned model versions. Reproducibility is a
  first-class concern, not a footnote.
- **SQLite call cache** — keyed by `hash(provider, model, prompt, params)`. Re-run
  evaluations without re-paying for tokens.
- **CI-friendly** — non-zero exit codes when metrics regress past a threshold.
  Designed for PR gates, not just notebooks.
- **Spanish-first datasets** — sample datasets ship in both English and Spanish
  so LatAm teams aren't stuck translating fixtures.

## Architecture

```
            ┌──────────────┐
            │  config.yaml │
            └──────┬───────┘
                   │
         ┌─────────▼─────────┐
         │     EvalRun       │  ← orchestrator
         └─┬───────┬───────┬─┘
           │       │       │
   ┌───────▼─┐  ┌──▼───┐  ┌▼────────┐
   │Provider │  │Dataset│  │ Metric  │
   │(LLM)    │  │       │  │         │
   └────┬────┘  └───────┘  └────┬────┘
        │                       │
   ┌────▼────┐             ┌────▼─────┐
   │ Cache   │             │  Judge   │
   │(SQLite) │             │  (LLM)   │
   └─────────┘             └──────────┘
                  │
        ┌─────────▼──────────┐
        │  run.manifest.json │  + JSON / HTML reports
        └────────────────────┘
```

## Quickstart

> Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/bryamacevedo/llm-evals.git
cd llm-evals
uv sync --all-extras
uv run pytest
```

CLI usage (coming in Phase 3):

```bash
uv run llm-evals run examples/rag.yaml
uv run llm-evals diff runs/2026-05-01/ runs/2026-05-04/
```

## Roadmap

- [x] **Phase 0** — Repo skeleton, tooling, CI
- [ ] **Phase 1** — Provider/Dataset/Metric abstractions, SQLite cache
- [ ] **Phase 2** — MVP metrics: faithfulness, answer relevance, context precision, hallucination flag
- [ ] **Phase 3** — CLI + JSON/HTML reports + run-diff
- [ ] **Phase 4** — Example datasets (EN/ES) + docs
- [ ] **Phase 5** — Publish to PyPI

## Development

```bash
uv sync --all-extras
uv run pre-commit install
uv run ruff check .
uv run mypy
uv run pytest --cov
```

## License

[MIT](LICENSE) © 2026 Bryam Camilo Acevedo
