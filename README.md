# llm-evals

> **Audit-grade LLM evaluation for Python.** Reproducible runs, prompt hashing, SQLite caching, and a CI gate that fails PRs when metrics regress.

[![CI](https://github.com/camilo-acevedo/llm-evals/actions/workflows/ci.yml/badge.svg)](https://github.com/camilo-acevedo/llm-evals/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Type-checked: mypy strict](https://img.shields.io/badge/type--checked-mypy_strict-2b6cb0.svg)](pyproject.toml)
[![Code style: ruff](https://img.shields.io/badge/style-ruff-000000.svg)](https://docs.astral.sh/ruff/)
[![Coverage: 94%](https://img.shields.io/badge/coverage-94%25-brightgreen.svg)](#development)

**Status:** alpha (v0.4.0) — feature-complete through Phase 4. The public API may still change before 1.0.

Preview a real run report without cloning anything:
[`qa_rag_en/sample/report.html`](examples/output/qa_rag_en/sample/report.html) ·
[`qa_rag_es/sample/report.html`](examples/output/qa_rag_es/sample/report.html) ·
[`summarization/sample/report.html`](examples/output/summarization/sample/report.html)

---

## Table of contents

- [Why llm-evals?](#why-llm-evals)
- [Architecture](#architecture)
- [Quickstart](#quickstart-offline-no-api-key)
- [Installation](#installation)
- [Configuration](#configuration)
- [CLI reference](#cli-reference)
- [Programmatic API](#programmatic-api)
- [Metrics](#metrics)
- [Reproducibility and manifests](#reproducibility-and-manifests)
- [Caching](#caching)
- [Providers](#providers)
- [Bundled examples](#bundled-examples)
- [CI integration](#ci-integration)
- [Development](#development)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)

---

## Why llm-evals?

Most evaluation libraries are excellent at **scoring** models. They are less
opinionated about what happens **after** the score: which prompt produced it,
against which dataset, with which model version, on which day. When a metric
moves three months later, the question "what changed?" is hard to answer.

`llm-evals` is built around three opinions:

1. **Every run is auditable.** A `manifest.json` records hashes of the prompt,
   dataset, and forwarded provider parameters, alongside ISO-8601 timestamps,
   provider name, model identifier, and per-metric aggregates. Two runs with
   identical inputs share a deterministic `run_id`.
2. **Every call is cacheable.** A SQLite cache keyed by
   `sha256(provider, model, prompt, params)` means a re-run after a config
   tweak only pays for the calls that changed.
3. **Every regression is gateable.** Thresholds in YAML map cleanly to exit
   codes. A `--max-regression` budget on `llm-evals diff` fails a PR when any
   metric drops more than allowed against a stored baseline.

### Compared to alternatives

|                              | llm-evals | DeepEval | Ragas | promptfoo |
| ---------------------------- | :-------: | :------: | :---: | :-------: |
| Run manifests (hashes, IDs)  |    yes    |    no    |  no   |   partial |
| SQLite call cache by default |    yes    |    no    |  no   |    yes    |
| Spanish-first sample data    |    yes    |    no    |  no   |    no     |
| CLI exit-code thresholds     |    yes    |   yes    |  no   |    yes    |
| LLM-as-judge metrics         |    yes    |   yes    |  yes  |    yes    |
| Strict typing (mypy strict)  |    yes    |   no     |  no   |    n/a    |
| Pure-Python, no Node runtime |    yes    |   yes    |  yes  |    no     |

`llm-evals` is intentionally smaller in scope: it ships the four metrics that
cover ~80% of RAG evaluations and skips the rest. If you need 30 metrics out
of the box, DeepEval or Ragas are better choices.

---

## Architecture

```
            ┌──────────────┐
            │  config.yaml │
            └──────┬───────┘
                   │ load_yaml + RunConfig.from_dict
                   ▼
         ┌─────────────────────┐
         │       EvalRun       │  orchestrator
         └──┬──────┬───────┬───┘
            │      │       │
   ┌────────▼─┐  ┌─▼────┐ ┌▼────────┐
   │ Provider │  │Dataset│ │ Metric  │  (× N)
   │  (LLM)   │  │       │ │         │
   └────┬─────┘  └───────┘ └────┬────┘
        │                       │
   ┌────▼─────┐            ┌────▼─────┐
   │  Cache   │            │  Judge   │  (LLM, optional)
   │ (SQLite) │            │          │
   └──────────┘            └──────────┘
                  │
                  ▼
        ┌────────────────────┐
        │   RunManifest +    │  manifest.json + summary.json
        │   per-sample log   │  + samples.jsonl + report.html
        └────────────────────┘
```

### Lifecycle of a single sample

1. The `Dataset` yields a `Sample` (`input`, optional `expected`, optional
   `context`, optional `metadata`).
2. The `Provider` receives the sample's `input` and returns a `Response`
   (text, model, provider, token counts, finish reason, latency).
3. The `Cache` is consulted first, keyed by the SHA-256 of
   `(provider, model, prompt, params)`. A hit short-circuits the upstream
   call entirely.
4. Each configured `Metric` scores the `(Sample, Response)` pair and produces
   a `MetricResult` (score in `[0, 1]`, optional reason, free-form metadata).
5. Aggregates are folded across all samples to produce the run summary.
6. A `RunManifest` is emitted with content-derived hashes, freezing exactly
   what ran for later audit.

---

## Quickstart (offline, no API key)

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/camilo-acevedo/llm-evals.git
cd llm-evals
uv sync --all-extras

uv run llm-evals run examples/configs/qa_rag_en.yaml
```

The bundled example uses a `FakeProvider` with canned answers, so no Anthropic
or OpenAI key is required to see the harness end-to-end. Expected output:

```text
Run 33618d875e28360d (fake/demo-model, 5 samples)
  exact_match         0.400  [FAIL]
  hallucination_flag  0.900  [OK]
Reports written to examples/output/qa_rag_en/<timestamp>_<run_id>
```

The `[FAIL]` is intentional: the config sets `exact_match` threshold to
`0.500` and the canned answers score `0.400`, so the CLI exits with code `1`.
This is the CI gate behaviour. Loosen the threshold to see the happy path.

The committed reference run lives at
[`examples/output/qa_rag_en/sample/`](examples/output/qa_rag_en/sample/) — open
`report.html` in a browser for the full drill-down view.

---

## Installation

### From source (current)

```bash
git clone https://github.com/camilo-acevedo/llm-evals.git
cd llm-evals
uv sync --all-extras   # or: pip install -e ".[all]"
```

### Provider extras

Each provider is opt-in to keep the install footprint small:

```bash
pip install 'llm-evals[anthropic]'        # Anthropic only
pip install 'llm-evals[openai]'           # OpenAI / OpenAI-compatible only
pip install 'llm-evals[ollama]'           # Ollama only
pip install 'llm-evals[all]'              # all three SDKs
```

Without any extra installed, only the `FakeProvider` (for tests and offline
demos) is usable.

### From PyPI

Coming in Phase 5. Until then, install from a Git tag:

```bash
pip install git+https://github.com/camilo-acevedo/llm-evals.git@v0.4.0
```

---

## Configuration

Run configurations are plain YAML, validated by frozen dataclasses with
explicit error messages.

### Reference

```yaml
# Required: model under evaluation
provider:
  type: anthropic | openai | ollama | fake
  model: claude-opus-4-7
  api_key: <optional>            # falls back to provider's env var
  base_url: <optional>           # OpenAI-compatible endpoints
  host: <optional>               # Ollama daemon URL
  default_max_tokens: 1024       # Anthropic only
  responses: ["..."]             # Fake only
  params:                        # forwarded on every complete() call
    temperature: 0.0

# Required: where samples come from
dataset:
  type: jsonl
  path: data/qa.jsonl

# Required: at least one metric
metrics:
  - type: exact_match
    normalize: true              # casefold + strip
  - type: faithfulness
  - type: answer_relevance
  - type: context_precision
  - type: hallucination_flag
    normalize_case: true

# Optional: SQLite cache (enabled by default at .llm-evals-cache.sqlite)
cache:
  enabled: true
  path: .llm-evals-cache.sqlite

# Required when any LLM-as-judge metric is used
judge:
  type: anthropic
  model: claude-opus-4-7
  params:
    temperature: 0.0

# Optional: aggregate gates. Missing thresholds for unknown metrics fail
# rather than pass silently, so typos surface immediately.
thresholds:
  faithfulness: 0.85
  answer_relevance: 0.80

# Optional: report destination and formats
output:
  dir: runs
  formats: [json, html]
```

### Sample dataset format (JSONL)

One JSON object per line:

```json
{"input": "What does LLM stand for?", "expected": "Large Language Model", "context": ["A Large Language Model is..."], "metadata": {"id": "qa-001"}}
```

`input` is required. `expected`, `context`, and `metadata` are optional but
required by specific metrics (see [Metrics](#metrics)).

---

## CLI reference

```text
llm-evals --version
llm-evals run <config.yaml> [--output-dir DIR] [--no-cache]
llm-evals diff <baseline-dir> <candidate-dir> [--max-regression FLOAT]
```

### `llm-evals run`

Loads the YAML configuration, builds an `EvalRun`, executes it, persists
configured reports, and gates on thresholds.

| Flag             | Effect                                                                    |
| ---------------- | ------------------------------------------------------------------------- |
| `--output-dir`   | Override `output.dir` from the config (e.g. on a CI scratch directory)    |
| `--no-cache`     | Disable the SQLite request cache regardless of `cache.enabled` in YAML    |

### `llm-evals diff`

Compares two run output directories. Prints a per-metric side-by-side table
with deltas. With `--max-regression`, exits non-zero when any metric drops
beyond the budget.

A metric present in the baseline but absent in the candidate counts as a full
regression of its baseline value, so a "removed by mistake" metric cannot
sneak past the gate.

### Exit codes

| Code | Meaning                                                                   |
| ---: | ------------------------------------------------------------------------- |
| `0`  | success                                                                   |
| `1`  | one or more metric thresholds violated, or `diff` saw a banned regression |
| `2`  | configuration or runtime error (bad YAML, missing dataset, provider error)|

---

## Programmatic API

When the CLI does not fit (custom dataset loaders, integration tests, embedded
in a larger pipeline), use the library directly:

```python
from llm_evals.core.cache import Cache
from llm_evals.core.eval_run import EvalRun
from llm_evals.datasets.jsonl import load_jsonl
from llm_evals.metrics.exact_match import ExactMatch
from llm_evals.metrics.faithfulness import Faithfulness
from llm_evals.metrics.llm_judge import LLMJudge
from llm_evals.providers.anthropic_provider import AnthropicProvider
from llm_evals.providers.cached import CachedProvider

# Wrap your provider in a cache so re-runs do not re-pay tokens
provider = CachedProvider(
    AnthropicProvider(api_key="..."),
    Cache(".llm-evals-cache.sqlite"),
)

# LLM-as-judge metrics receive a configured judge
judge = LLMJudge(provider, model="claude-opus-4-7")

run = EvalRun(
    provider=provider,
    dataset=load_jsonl("data/qa.jsonl"),
    metrics=[ExactMatch(), Faithfulness(judge)],
    model="claude-opus-4-7",
    params={"temperature": 0.0},
)

result = run.execute()

print(result.summary)              # {"exact_match": 0.62, "faithfulness": 0.91}
print(result.manifest.run_id)      # 16-char hex
print(result.manifest.to_json())   # full audit record
```

Every dataclass (`Sample`, `Response`, `MetricResult`, `SampleResult`,
`RunResult`, `RunManifest`) is `frozen=True` so results are safe to share
across threads or pass to long-lived processes.

---

## Metrics

| Metric              | Type         | Requires                       | Score in `[0, 1]` |
| ------------------- | ------------ | ------------------------------ | :---------------: |
| `exact_match`       | heuristic    | `expected`                     |        yes        |
| `hallucination_flag`| heuristic    | non-empty `context`            |        yes        |
| `faithfulness`      | LLM-as-judge | `judge` block + `context`      |        yes        |
| `answer_relevance`  | LLM-as-judge | `judge` block                  |        yes        |
| `context_precision` | LLM-as-judge | `judge` block + `context`      |        yes        |

Heuristic metrics cost nothing at runtime. Judge metrics issue **one** upstream
call per sample (single-pass decompose-and-verify), cached per request key.

### `exact_match`

```text
score = 1.0 if normalize(expected) == normalize(response) else 0.0
```

Normalisation is `str.strip().casefold()` by default. Disable with
`normalize: false` for byte-for-byte comparison.

### `hallucination_flag`

```text
extracted = numbers(response) ∪ proper_nouns(response)
hallucinated = { token ∈ extracted : token ∉ joined_context }
score = 1.0 - |hallucinated| / |extracted|         (or 1.0 if extracted is empty)
```

Numbers are matched by `\d[\d.,]*` (covers `42`, `3.14`, `1,200`). Proper nouns
are sequences of consecutive capitalised words. Comparison is case-insensitive
by default; flip `normalize_case: false` for strict matching.

This is a **cheap triage signal**, not a substitute for `faithfulness`. False
positives are common (sentence-initial capitals, surface-form variation). The
recommended pairing is `faithfulness` + `hallucination_flag` in the same run.

### `faithfulness`

The judge decomposes the answer into atomic claims and labels each as
`supported` or not against the context, in a single structured-output call:

```text
score = supported_claims / total_claims         (or 1.0 if no claims extracted)
```

An empty claim list scores `1.0` because an answer that asserts nothing cannot
have hallucinated. Malformed claim entries from a misbehaving judge count as
unsupported rather than aborting the whole evaluation.

### `answer_relevance`

The judge returns a continuous score directly:

```text
score = clamp(judge_score, 0.0, 1.0)
```

JSON booleans (which Python treats as ints) are rejected explicitly so a
`True/False` verdict cannot silently coerce to `1.0/0.0`.

### `context_precision`

For each chunk in `sample.context`, the judge labels it `used` or not:

```text
score = used_chunks / total_chunks
```

A high score means the retriever surfaced little noise; a low score means it
over-retrieved. Empty verdict lists raise `MetricError` rather than producing a
`0/0` score, since "no opinion" differs from "every chunk was useless".

---

## Reproducibility and manifests

Every run emits a `manifest.json`. Two runs with identical configuration
produce manifests that share a `run_id`:

```json
{
  "run_id": "33618d875e28360d",
  "provider": "anthropic",
  "model": "claude-opus-4-7",
  "metric_names": ["exact_match", "hallucination_flag"],
  "sample_count": 5,
  "dataset_hash": "8a6c4f...",
  "params_hash": "d44d2e...",
  "started_at": "2026-05-04T21:45:28+00:00",
  "finished_at": "2026-05-04T21:45:28+00:00",
  "summary": {"exact_match": 0.4, "hallucination_flag": 0.9}
}
```

### What the hashes prove

| Hash             | What it covers                                            |
| ---------------- | --------------------------------------------------------- |
| `dataset_hash`   | SHA-256 over the canonical JSON of every sample           |
| `params_hash`    | SHA-256 over the forwarded provider parameters dict       |
| `run_id`         | First 16 hex chars of `sha256(provider, model, dataset_hash, params_hash, started_at)` |

If `dataset_hash` differs between two runs, the dataset changed. If
`params_hash` differs, someone tweaked the temperature or token limits. If
both match but `summary` moved, the model itself (or its upstream provider)
shifted under your feet.

### On-disk layout

```
runs/<timestamp>_<run_id>/
├── manifest.json     # the audit record above
├── summary.json      # compact aggregates, sized for CI grep
├── samples.jsonl     # one JSON object per evaluated sample
└── report.html       # self-contained drill-down (when html in formats)
```

`summary.json` is the document a CI step pipes into `jq` to gate a PR.
`samples.jsonl` is for offline post-mortems.
`report.html` is for humans reviewing the failure.

---

## Caching

The `Cache` is a SQLite database with a single `responses` table, journaled
in WAL mode for safe concurrent reads.

### Cache key

```python
sha256(json.dumps({
    "provider": provider_name,   # e.g. "anthropic"
    "model":    model_id,         # e.g. "claude-opus-4-7"
    "prompt":   prompt_text,
    "params":   forwarded_params, # temperature, max_tokens, system, ...
}, sort_keys=True))
```

Order independence comes from `sort_keys=True`; any field that affects the
upstream response is part of the key. Bumping the model version, tweaking
`temperature`, or rewording the prompt all invalidate cache entries
automatically.

### When to bust the cache

Pass `--no-cache` on the CLI, or remove the SQLite file. There is no eviction
policy — entries live until you delete them. For long-running benchmarks,
that is usually what you want.

---

## Providers

| Provider | Install                              | Constructor key       | Notes |
| -------- | ------------------------------------ | --------------------- | ----- |
| Anthropic| `pip install 'llm-evals[anthropic]'` | `provider.type: anthropic` | `default_max_tokens` settable; Messages API |
| OpenAI   | `pip install 'llm-evals[openai]'`    | `provider.type: openai` | `base_url` lets you point at Azure / vLLM |
| Ollama   | `pip install 'llm-evals[ollama]'`    | `provider.type: ollama` | Local-first; supports both legacy dict and modern object response shapes |
| Fake     | (built-in)                           | `provider.type: fake`   | Deterministic; for tests and offline demos |

### Adding a custom provider

Extend `llm_evals.providers.base.Provider`:

```python
from llm_evals.core.types import Response
from llm_evals.providers.base import Provider

class MyProvider(Provider):
    def __init__(self, ...):
        self.name = "my-provider"
        ...

    def complete(self, prompt: str, *, model: str, **params) -> Response:
        # call your upstream
        return Response(
            text=...,
            model=model,
            provider=self.name,
            prompt_tokens=...,
            completion_tokens=...,
            finish_reason=...,
            latency_ms=...,
            raw={"id": ...},
        )
```

`CachedProvider` and every metric will work transparently. The library does
not require subclasses to register anywhere.

---

## Bundled examples

Each example writes a JSON manifest, summary, samples log, and self-contained
HTML report under `examples/output/<name>/sample/`.

| Config                                                                  | Dataset                                                                | Metrics                            | Demonstrates                                                |
| ----------------------------------------------------------------------- | ---------------------------------------------------------------------- | ---------------------------------- | ----------------------------------------------------------- |
| [`qa_rag_en.yaml`](examples/configs/qa_rag_en.yaml)                     | [QA EN](examples/datasets/qa_rag_en.jsonl)                             | exact_match, hallucination_flag    | Mixed pass/fail with the threshold gate (exit 1)            |
| [`qa_rag_es.yaml`](examples/configs/qa_rag_es.yaml)                     | [QA ES](examples/datasets/qa_rag_es.jsonl)                             | exact_match, hallucination_flag    | Spanish dataset, both thresholds satisfied (exit 0)         |
| [`summarization.yaml`](examples/configs/summarization.yaml)             | [Summarization](examples/datasets/summarization.jsonl)                 | hallucination_flag                 | Near-miss threshold violation (0.889 < 0.900)               |

Browse the committed reports directly in the browser:

- [`examples/output/qa_rag_en/sample/report.html`](examples/output/qa_rag_en/sample/report.html)
- [`examples/output/qa_rag_es/sample/report.html`](examples/output/qa_rag_es/sample/report.html)
- [`examples/output/summarization/sample/report.html`](examples/output/summarization/sample/report.html)

---

## CI integration

A typical PR-gate workflow:

```yaml
name: eval-gate
on:
  pull_request:
    paths:
      - "src/**"
      - "data/**"
      - "eval/**"

jobs:
  evaluate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v4
      - run: uv sync --all-extras

      - name: Run evaluation gate
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
        run: uv run llm-evals run eval/config.yaml

      - name: Upload run artifacts
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: eval-run
          path: runs/
```

The job fails when any metric falls below its threshold. Run artifacts upload
on every job (success or failure) so reviewers can browse the HTML report
straight from the workflow run.

For comparing a PR against `main`:

```yaml
      - name: Compare against baseline
        run: |
          uv run llm-evals diff baseline/sample candidate/sample --max-regression 0.05
```

---

## Development

```bash
git clone https://github.com/camilo-acevedo/llm-evals.git
cd llm-evals
uv sync --all-extras
uv run pre-commit install
```

The project is verified by:

| Tool       | Command                              |
| ---------- | ------------------------------------ |
| Lint       | `uv run ruff check .`                |
| Format     | `uv run ruff format --check .`       |
| Types      | `uv run mypy` (strict)               |
| Tests      | `uv run pytest`                      |
| Coverage   | `uv run pytest --cov`                |

CI runs the full toolchain on Linux, macOS, and Windows across Python 3.11,
3.12, and 3.13.

### Project layout

```
src/llm_evals/
├── core/         # EvalRun, Cache, RunManifest, Response
├── providers/    # Provider ABC, FakeProvider, CachedProvider, real adapters
├── datasets/     # Sample, JSONL loader
├── metrics/      # Metric ABC, ExactMatch, Faithfulness, ...
├── reports/      # JSON and HTML writers
├── config/       # YAML schema + loader + builder
├── cli/          # argparse subcommand dispatcher
└── exceptions.py # LLMEvalsError hierarchy

tests/            # mirror of src/llm_evals/
examples/         # datasets, configs, committed sample outputs
```

---

## Roadmap

- [x] **Phase 0** — Repo skeleton, tooling, CI matrix
- [x] **Phase 1** — Provider/Dataset/Metric abstractions, SQLite cache
- [x] **Phase 2** — MVP metrics: faithfulness, answer relevance, context precision, hallucination flag
- [x] **Phase 3** — CLI (`run`, `diff`) + JSON/HTML reports
- [x] **Phase 4** — Example datasets (EN/ES) and full documentation
- [ ] **Phase 5** — Publish to PyPI with trusted publishing
- [ ] **Post-1.0** — Embedding-based metrics, async provider variants, HuggingFace dataset loader, Markdown report writer

The next breaking-change window will be tagged `v1.0.0`. Until then, treat
the public API as alpha.

---

## Contributing

Bug reports and small PRs are welcome. Larger contributions should open an
issue first to align on direction.

Expected of every PR:

- All tooling green: `ruff check`, `ruff format --check`, `mypy`, `pytest`
- New code paths covered by tests
- Public functions and classes documented with reST-style Sphinx docstrings
- Conventional Commits message style (`feat:`, `fix:`, `docs:`, `chore:`, ...)

Run the full validation locally:

```bash
uv run ruff check . && uv run ruff format --check . && uv run mypy && uv run pytest
```

---

## License

[MIT](LICENSE) © 2026 Bryam Camilo Acevedo
