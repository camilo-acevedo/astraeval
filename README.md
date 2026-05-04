# llm-evals

> Reproducible LLM evaluation harness with run manifests, prompt hashing, and SQLite caching.

[![CI](https://github.com/camilo-acevedo/llm-evals/actions/workflows/ci.yml/badge.svg)](https://github.com/camilo-acevedo/llm-evals/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> **Status:** alpha (v0.4.0) — feature-complete through Phase 4. API may still shift before 1.0.

---

## Why another LLM eval harness?

Most evaluation libraries (DeepEval, Ragas, promptfoo) are excellent at scoring
models but treat the run itself as ephemeral. `llm-evals` is built for teams
that need to **audit and reproduce** scores months later:

- **Run manifests** — every evaluation emits a `manifest.json` with hashes of
  prompts, dataset content, and pinned model versions. Reproducibility is a
  first-class concern, not a footnote.
- **SQLite call cache** — keyed by `hash(provider, model, prompt, params)`.
  Re-run evaluations without re-paying for tokens.
- **CI-friendly** — non-zero exit codes when metrics drop below a threshold or
  regress against a baseline. Designed for PR gates, not just notebooks.
- **Spanish-first datasets** — example fixtures ship in both English and
  Spanish so LatAm teams aren't stuck translating boilerplate.

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
        │   manifest.json    │  + summary.json + samples.jsonl + report.html
        └────────────────────┘
```

## Quickstart

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/). The Quickstart runs
fully **offline** — it uses the bundled `FakeProvider` with canned answers, so
no API key is needed.

```bash
git clone https://github.com/camilo-acevedo/llm-evals.git
cd llm-evals
uv sync --all-extras

uv run llm-evals run examples/configs/qa_rag_en.yaml
```

Expected output (this config is intentionally calibrated to fail one of two
thresholds, demonstrating the CI gate):

```text
Run 33618d875e28360d (fake/demo-model, 5 samples)
  exact_match         0.400  [FAIL]
  hallucination_flag  0.900  [OK]
Reports written to examples/output/qa_rag_en/<timestamp>_<run_id>
```

The committed sample lives at `examples/output/qa_rag_en/sample/`. Open
`report.html` in a browser to see the per-sample drill-down.

## CLI

```bash
llm-evals run <config.yaml> [--output-dir DIR] [--no-cache]
llm-evals diff <baseline-run-dir> <candidate-run-dir> [--max-regression FLOAT]
llm-evals --version
```

Exit codes:

| Code | Meaning |
|---:|---|
| `0` | success |
| `1` | one or more metric thresholds violated |
| `2` | configuration or runtime error |

## Configuration

Run configurations are plain YAML. A minimal config:

```yaml
provider:
  type: anthropic       # or openai, ollama, fake
  model: claude-opus-4-7

dataset:
  type: jsonl
  path: data/qa.jsonl

metrics:
  - type: faithfulness
  - type: answer_relevance
  - type: hallucination_flag

judge:                  # required by LLM-as-judge metrics above
  type: anthropic
  model: claude-opus-4-7

thresholds:
  faithfulness: 0.85
  answer_relevance: 0.80

output:
  dir: runs
  formats: [json, html]
```

See [`examples/configs/`](examples/configs/) for three end-to-end configs and
their dataset fixtures.

## Bundled examples

Each example writes a JSON manifest, summary, samples log, and self-contained
HTML report under `examples/output/<name>/sample/`.

| Config | Dataset | Metrics | Demonstrates |
|---|---|---|---|
| [`qa_rag_en.yaml`](examples/configs/qa_rag_en.yaml) | [QA EN](examples/datasets/qa_rag_en.jsonl) | exact_match, hallucination_flag | Mixed pass/fail with the threshold gate (exit 1) |
| [`qa_rag_es.yaml`](examples/configs/qa_rag_es.yaml) | [QA ES](examples/datasets/qa_rag_es.jsonl) | exact_match, hallucination_flag | Spanish dataset, both thresholds satisfied (exit 0) |
| [`summarization.yaml`](examples/configs/summarization.yaml) | [Summarization](examples/datasets/summarization.jsonl) | hallucination_flag | Near-miss threshold violation (0.889 < 0.900) |

Browse the committed samples directly:

- [`examples/output/qa_rag_en/sample/report.html`](examples/output/qa_rag_en/sample/report.html)
- [`examples/output/qa_rag_es/sample/report.html`](examples/output/qa_rag_es/sample/report.html)
- [`examples/output/summarization/sample/report.html`](examples/output/summarization/sample/report.html)

## Metrics

| Metric | Type | Requires |
|---|---|---|
| `exact_match` | heuristic | `expected` field on each sample |
| `hallucination_flag` | heuristic | non-empty `context` |
| `faithfulness` | LLM-as-judge | `judge` block + non-empty `context` |
| `answer_relevance` | LLM-as-judge | `judge` block |
| `context_precision` | LLM-as-judge | `judge` block + non-empty `context` |

Heuristic metrics are free to run; LLM-as-judge metrics issue one upstream call
per sample. Cache misses happen once per unique `(provider, model, prompt,
params)` tuple.

## CI integration

A typical PR gate looks like this:

```yaml
- name: Run evaluation gate
  run: uv run llm-evals run eval/config.yaml
  # exits 1 when thresholds are violated; the workflow fails accordingly
```

For comparing PR-vs-baseline:

```yaml
- name: Compare against main baseline
  run: uv run llm-evals diff baseline/sample candidate/sample --max-regression 0.05
```

## Providers

| Provider | Extra to install | Constructor key |
|---|---|---|
| Anthropic | `pip install 'llm-evals[anthropic]'` | `provider.type: anthropic` |
| OpenAI / OpenAI-compatible | `pip install 'llm-evals[openai]'` | `provider.type: openai` |
| Ollama (local models) | `pip install 'llm-evals[ollama]'` | `provider.type: ollama` |
| Fake (tests / offline demos) | (built in) | `provider.type: fake` |

`uv sync --all-extras` installs every provider SDK in one shot.

## Development

```bash
uv sync --all-extras
uv run pre-commit install
uv run ruff check .
uv run mypy
uv run pytest --cov
```

## Roadmap

- [x] **Phase 0** — Repo skeleton, tooling, CI
- [x] **Phase 1** — Provider/Dataset/Metric abstractions, SQLite cache
- [x] **Phase 2** — MVP metrics: faithfulness, answer relevance, context precision, hallucination flag
- [x] **Phase 3** — CLI + JSON/HTML reports + run-diff
- [x] **Phase 4** — Example datasets (EN/ES) + docs
- [ ] **Phase 5** — Publish to PyPI

## License

[MIT](LICENSE) © 2026 Bryam Camilo Acevedo
