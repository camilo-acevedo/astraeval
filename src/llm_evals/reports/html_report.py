"""Self-contained HTML report writer for human inspection of run results."""

from __future__ import annotations

import html as _html
import statistics
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path

from llm_evals.core.eval_run import RunResult, SampleResult
from llm_evals.core.manifest import RunManifest
from llm_evals.core.types import Response
from llm_evals.metrics.base import MetricResult

_GOOD = 0.8
_OK = 0.5

_STYLE = """
:root {
  color-scheme: light dark;
  --bg: #0d1017;
  --panel: #161b25;
  --panel-2: #1d2330;
  --panel-3: #252b3a;
  --border: #2a3142;
  --border-soft: rgba(255, 255, 255, 0.06);
  --text: #e8ebf1;
  --text-soft: #c4cad6;
  --muted: #97a2b8;
  --muted-2: #5b6478;
  --accent: #60a5fa;
  --accent-soft: rgba(96, 165, 250, 0.12);
  --good: #22c55e;
  --good-soft: rgba(34, 197, 94, 0.14);
  --ok: #eab308;
  --ok-soft: rgba(234, 179, 8, 0.14);
  --bad: #ef4444;
  --bad-soft: rgba(239, 68, 68, 0.14);
  --shadow: 0 1px 2px rgba(0,0,0,0.2), 0 6px 16px rgba(0,0,0,0.18);
  --mono: ui-monospace, "SF Mono", "JetBrains Mono", Consolas, monospace;
  --sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, system-ui, sans-serif;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font: 14px/1.55 var(--sans);
  -webkit-font-smoothing: antialiased;
}
header.app-header {
  background: linear-gradient(180deg, var(--panel-2), var(--panel));
  border-bottom: 1px solid var(--border);
  padding: 24px 32px 22px;
}
.app-title {
  display: flex;
  align-items: center;
  gap: 14px;
  margin-bottom: 10px;
  flex-wrap: wrap;
}
.brand {
  font-size: 13px;
  font-weight: 600;
  letter-spacing: -0.01em;
  color: var(--muted);
  text-transform: lowercase;
}
.brand::after {
  content: "/";
  color: var(--muted-2);
  margin: 0 8px;
  font-weight: 400;
}
.app-title h1 {
  font-size: 18px;
  font-weight: 600;
  margin: 0;
  letter-spacing: -0.01em;
}
.run-pill {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  background: var(--panel-3);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 3px 4px 3px 10px;
  font-family: var(--mono);
  font-size: 12px;
  color: var(--text-soft);
}
.copy-btn {
  background: transparent;
  border: 0;
  color: var(--muted);
  cursor: pointer;
  font: 11px var(--mono);
  padding: 2px 6px;
  border-radius: 4px;
  transition: color 0.15s, background 0.15s;
}
.copy-btn:hover { color: var(--text); background: var(--border); }
.copy-btn.copied { color: var(--good); }
.app-meta {
  font-family: var(--mono);
  font-size: 12px;
  color: var(--muted);
}
.app-meta .sep {
  color: var(--muted-2);
  margin: 0 8px;
}
.hashes {
  margin-top: 12px;
  display: flex;
  gap: 18px;
  font-family: var(--mono);
  font-size: 11px;
  color: var(--muted);
  flex-wrap: wrap;
}
.hashes .hash-item {
  display: inline-flex;
  align-items: center;
  gap: 6px;
}
.hashes code {
  color: var(--text-soft);
  background: var(--panel-3);
  padding: 1px 6px;
  border-radius: 4px;
  font-size: 11px;
}
main {
  max-width: 1180px;
  margin: 0 auto;
  padding: 28px 32px 60px;
}
section {
  margin-bottom: 28px;
}
section > h2 {
  font-size: 11px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: var(--muted);
  margin: 0 0 12px;
}
.stats {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 12px;
}
.stat-card {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 14px 16px;
  box-shadow: var(--shadow);
}
.stat-card .label {
  font-size: 11px;
  font-weight: 500;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--muted);
  margin-bottom: 6px;
}
.stat-card .value {
  font-family: var(--mono);
  font-size: 22px;
  font-weight: 600;
  color: var(--text);
  letter-spacing: -0.01em;
}
.stat-card .sub {
  margin-top: 4px;
  font-size: 11px;
  color: var(--muted);
  font-family: var(--mono);
}
.summary-card {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 10px;
  overflow: hidden;
  box-shadow: var(--shadow);
}
.summary-card table {
  width: 100%;
  border-collapse: collapse;
}
.summary-card th, .summary-card td {
  padding: 12px 16px;
  text-align: left;
  border-bottom: 1px solid var(--border-soft);
  font-size: 13px;
  vertical-align: middle;
}
.summary-card tr:last-child td {
  border-bottom: none;
}
.summary-card th {
  font-family: var(--sans);
  font-size: 11px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--muted);
  background: var(--panel-2);
}
.summary-card td.metric-name {
  font-family: var(--mono);
  font-weight: 500;
  color: var(--text);
}
.bar {
  width: 220px;
  height: 6px;
  background: var(--panel-3);
  border-radius: 3px;
  overflow: hidden;
  position: relative;
}
.bar .fill {
  height: 100%;
  border-radius: 3px;
  transition: width 0.3s ease;
}
.bar .fill.score-good { background: var(--good); }
.bar .fill.score-ok { background: var(--ok); }
.bar .fill.score-bad { background: var(--bad); }
.score {
  display: inline-block;
  padding: 2px 8px;
  border-radius: 999px;
  font-family: var(--mono);
  font-size: 12px;
  font-weight: 600;
  min-width: 56px;
  text-align: center;
}
.score-good { background: var(--good-soft); color: var(--good); }
.score-ok { background: var(--ok-soft); color: var(--ok); }
.score-bad { background: var(--bad-soft); color: var(--bad); }
.badge {
  display: inline-block;
  padding: 2px 8px;
  border-radius: 4px;
  font-family: var(--mono);
  font-size: 11px;
  font-weight: 500;
  letter-spacing: 0.02em;
  white-space: nowrap;
}
.badge-pass { background: var(--good-soft); color: var(--good); }
.badge-fail { background: var(--bad-soft); color: var(--bad); }
.badge-soft { background: var(--panel-3); color: var(--muted); }
.muted { color: var(--muted-2); font-family: var(--mono); font-size: 12px; }
.samples-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  margin-bottom: 12px;
  flex-wrap: wrap;
}
.samples-header h2 {
  margin: 0;
}
.controls {
  display: flex;
  gap: 6px;
  align-items: center;
}
.controls input[type="search"] {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 6px 10px;
  font: 13px var(--sans);
  color: var(--text);
  width: 220px;
  transition: border-color 0.15s;
}
.controls input[type="search"]:focus {
  outline: none;
  border-color: var(--accent);
}
.controls input[type="search"]::placeholder { color: var(--muted-2); }
.controls button {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 6px 12px;
  font: 12px var(--sans);
  color: var(--text-soft);
  cursor: pointer;
  transition: background 0.15s, color 0.15s;
}
.controls button:hover {
  background: var(--panel-2);
  color: var(--text);
}
details.sample {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 10px;
  margin-bottom: 8px;
  overflow: hidden;
  transition: border-color 0.15s;
}
details.sample[open] {
  background: var(--panel-2);
  border-color: var(--border);
}
details.sample > summary {
  cursor: pointer;
  padding: 14px 18px;
  display: flex;
  align-items: center;
  gap: 12px;
  list-style: none;
  user-select: none;
}
details.sample > summary::-webkit-details-marker { display: none; }
details.sample > summary::before {
  content: "\\203A";
  color: var(--muted);
  font-size: 18px;
  line-height: 1;
  transition: transform 0.2s ease;
  font-family: var(--mono);
  width: 12px;
  display: inline-block;
}
details.sample[open] > summary::before { transform: rotate(90deg); }
.idx {
  color: var(--muted);
  font-family: var(--mono);
  font-size: 12px;
  flex-shrink: 0;
  min-width: 28px;
}
.input-preview {
  flex: 1;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  font-size: 13px;
  color: var(--text-soft);
}
.meta-badges {
  display: flex;
  gap: 6px;
  flex-shrink: 0;
}
.scores {
  display: flex;
  gap: 6px;
  flex-shrink: 0;
}
.body {
  padding: 4px 18px 18px 42px;
  border-top: 1px solid var(--border-soft);
}
.field {
  margin-top: 16px;
}
.field > h3 {
  font-size: 11px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--muted);
  margin: 0 0 8px;
}
.field pre {
  font-family: var(--mono);
  font-size: 12.5px;
  white-space: pre-wrap;
  word-break: break-word;
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 10px 12px;
  margin: 0;
  line-height: 1.55;
}
.context-block {
  font-family: var(--mono);
  font-size: 12px;
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 8px 12px;
  margin-bottom: 6px;
  white-space: pre-wrap;
  line-height: 1.55;
}
.context-index { color: var(--muted); margin-right: 8px; }
.empty {
  color: var(--muted-2);
  font-style: italic;
  font-size: 12.5px;
}
.response-meta {
  display: flex;
  gap: 6px;
  margin-top: 8px;
  flex-wrap: wrap;
}
.metrics-mini {
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: 6px;
  overflow: hidden;
}
.metrics-mini table {
  width: 100%;
  border-collapse: collapse;
}
.metrics-mini th, .metrics-mini td {
  padding: 8px 12px;
  font-size: 12px;
  text-align: left;
  border-bottom: 1px solid var(--border-soft);
  vertical-align: middle;
}
.metrics-mini tr:last-child td { border-bottom: none; }
.metrics-mini th {
  background: var(--panel-3);
  font-weight: 500;
  color: var(--muted);
  font-family: var(--sans);
  font-size: 10.5px;
  text-transform: uppercase;
  letter-spacing: 0.08em;
}
.metrics-mini td.metric-reason {
  color: var(--text-soft);
  font-family: var(--sans);
}
#no-results {
  display: none;
  padding: 24px;
  text-align: center;
  color: var(--muted);
  font-size: 13px;
  background: var(--panel);
  border: 1px dashed var(--border);
  border-radius: 10px;
}
@media (max-width: 720px) {
  header.app-header { padding: 18px 16px; }
  main { padding: 18px 16px 40px; }
  .controls input[type="search"] { width: 160px; }
  .bar { width: 120px; }
  details.sample > summary { padding: 12px 14px; gap: 8px; }
  .body { padding: 4px 14px 14px 28px; }
}
@media (prefers-color-scheme: light) {
  :root {
    --bg: #f7f8fa;
    --panel: #ffffff;
    --panel-2: #f3f5f9;
    --panel-3: #e8ecf1;
    --border: #d8dde6;
    --border-soft: rgba(0, 0, 0, 0.05);
    --text: #1a1c23;
    --text-soft: #2c3038;
    --muted: #5b6478;
    --muted-2: #8a92a4;
    --accent: #2563eb;
    --accent-soft: rgba(37, 99, 235, 0.1);
    --shadow: 0 1px 2px rgba(0,0,0,0.04), 0 4px 12px rgba(0,0,0,0.06);
  }
}
"""

_SCRIPT = """
(() => {
  const search = document.getElementById('sample-search');
  const samples = Array.from(document.querySelectorAll('details.sample'));
  const noResults = document.getElementById('no-results');
  const filter = (q) => {
    const query = q.trim().toLowerCase();
    let visible = 0;
    samples.forEach((s) => {
      const text = (s.dataset.searchText || '').toLowerCase();
      const match = !query || text.includes(query);
      s.style.display = match ? '' : 'none';
      if (match) visible++;
    });
    if (noResults) noResults.style.display = visible === 0 ? 'block' : 'none';
  };
  if (search) search.addEventListener('input', (e) => filter(e.target.value));
  document.querySelectorAll('[data-action]').forEach((btn) => {
    btn.addEventListener('click', () => {
      const action = btn.dataset.action;
      if (action === 'expand-all') samples.forEach((s) => { s.open = true; });
      else if (action === 'collapse-all') samples.forEach((s) => { s.open = false; });
    });
  });
  document.querySelectorAll('[data-copy]').forEach((btn) => {
    btn.addEventListener('click', async () => {
      try {
        await navigator.clipboard.writeText(btn.dataset.copy);
        btn.classList.add('copied');
        setTimeout(() => btn.classList.remove('copied'), 1500);
      } catch (err) { /* clipboard unavailable */ }
    });
  });
})();
"""


def write_html(
    result: RunResult,
    path: Path,
    *,
    thresholds: Mapping[str, float] | None = None,
) -> None:
    """Render ``result`` as a single self-contained HTML file.

    The output embeds CSS and JavaScript inline so the report can be
    emailed, attached to a PR, or hosted on a static page without any
    external assets.

    :param result: Aggregate run result to render.
    :type result: RunResult
    :param path: Destination file path. Existing content is overwritten.
    :type path: pathlib.Path
    :param thresholds: Optional mapping from metric name to minimum
        acceptable score. When provided, the summary table shows a
        per-metric pass/fail badge alongside the threshold value.
    :type thresholds: collections.abc.Mapping[str, float] | None
    """
    path.write_text(_render_html(result, thresholds or {}), encoding="utf-8")


def _render_html(result: RunResult, thresholds: Mapping[str, float]) -> str:
    """Compose the full HTML document for a run result.

    :param result: Aggregate run result to render.
    :type result: RunResult
    :param thresholds: Mapping from metric name to minimum acceptable
        score. May be empty.
    :type thresholds: collections.abc.Mapping[str, float]
    :returns: Complete HTML document as a single string.
    :rtype: str
    """
    header = _render_header(result.manifest)
    stats = _render_stats(result)
    summary = _render_summary_table(result.summary, thresholds)
    samples = _render_samples_section(result.samples)
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en"><head>'
        '<meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>llm-evals run {_e(result.manifest.run_id)}</title>"
        f"<style>{_STYLE}</style>"
        "</head><body>"
        f"{header}"
        "<main>"
        f"{stats}{summary}{samples}"
        "</main>"
        f"<script>{_SCRIPT}</script>"
        "</body></html>"
    )


def _render_header(manifest: RunManifest) -> str:
    """Build the top header with run id, model, timestamps, hashes.

    :param manifest: Run manifest carrying every header field.
    :type manifest: RunManifest
    :returns: Rendered ``<header>`` block.
    :rtype: str
    """
    duration = _format_duration(manifest.started_at, manifest.finished_at)
    duration_html = f'<span class="sep">|</span>{_e(duration)}' if duration else ""
    return (
        '<header class="app-header">'
        '<div class="app-title">'
        '<span class="brand">llm-evals</span>'
        f"<h1>Run {_e(manifest.run_id)}</h1>"
        '<span class="run-pill">'
        f"<span>{_e(manifest.run_id)}</span>"
        f'<button class="copy-btn" data-copy="{_e(manifest.run_id)}" title="Copy run id">copy</button>'
        "</span>"
        "</div>"
        '<div class="app-meta">'
        f"{_e(manifest.provider)}/{_e(manifest.model)}"
        f'<span class="sep">|</span>'
        f"{manifest.sample_count} samples"
        f'<span class="sep">|</span>'
        f"{_e(manifest.started_at)}"
        f"{duration_html}"
        "</div>"
        '<div class="hashes">'
        f"{_render_hash_item('dataset', manifest.dataset_hash)}"
        f"{_render_hash_item('params', manifest.params_hash)}"
        "</div>"
        "</header>"
    )


def _render_hash_item(label: str, value: str) -> str:
    """Render one ``<label>: <code>truncated...</code> [copy]`` row.

    :param label: Short label such as ``"dataset"`` or ``"params"``.
    :type label: str
    :param value: Full hex digest. Displayed truncated to 12 chars; the
        full value is preserved on the copy button.
    :type value: str
    :returns: Rendered hash item span.
    :rtype: str
    """
    short = value[:12] + ("..." if len(value) > 12 else "")
    return (
        f'<span class="hash-item">{label}: <code>{_e(short)}</code>'
        f'<button class="copy-btn" data-copy="{_e(value)}" title="Copy {_e(label)} hash">copy</button>'
        "</span>"
    )


def _render_stats(result: RunResult) -> str:
    """Build the aggregate stat cards (samples, tokens, latency).

    :param result: Aggregate run result.
    :type result: RunResult
    :returns: Rendered ``<section class="stats">`` block.
    :rtype: str
    """
    cards = [
        _stat_card("Samples", str(result.manifest.sample_count), None),
        _token_stat_card(result),
        _latency_stat_card(result),
    ]
    return f'<section class="stats">{"".join(cards)}</section>'


def _stat_card(label: str, value: str, sub: str | None) -> str:
    """Render one stat card with a label, headline value, and optional sub-line.

    :param label: Uppercased label shown above the headline value.
    :type label: str
    :param value: Headline value, rendered in monospace.
    :type value: str
    :param sub: Optional secondary line, used for breakdowns or quantiles.
    :type sub: str | None
    :returns: Rendered ``<div class="stat-card">`` block.
    :rtype: str
    """
    sub_html = f'<div class="sub">{_e(sub)}</div>' if sub else ""
    return (
        '<div class="stat-card">'
        f'<div class="label">{_e(label)}</div>'
        f'<div class="value">{_e(value)}</div>'
        f"{sub_html}"
        "</div>"
    )


def _token_stat_card(result: RunResult) -> str:
    """Render the total-tokens stat card with prompt/completion breakdown.

    :param result: Aggregate run result.
    :type result: RunResult
    :returns: Rendered stat card.
    :rtype: str
    """
    prompt = _sum_optional(s.response.prompt_tokens for s in result.samples)
    completion = _sum_optional(s.response.completion_tokens for s in result.samples)
    if prompt is None and completion is None:
        return _stat_card("Total tokens", "-", "not reported by provider")
    total = (prompt or 0) + (completion or 0)
    sub_parts = []
    if prompt is not None:
        sub_parts.append(f"prompt {prompt:,}")
    if completion is not None:
        sub_parts.append(f"completion {completion:,}")
    sub = "  ".join(sub_parts) if sub_parts else None
    return _stat_card("Total tokens", f"{total:,}", sub)


def _latency_stat_card(result: RunResult) -> str:
    """Render the latency stat card with mean and tail-latency breakdown.

    :param result: Aggregate run result.
    :type result: RunResult
    :returns: Rendered stat card.
    :rtype: str
    """
    latencies = [s.response.latency_ms for s in result.samples if s.response.latency_ms is not None]
    if not latencies:
        return _stat_card("Latency", "-", "not reported by provider")
    mean = statistics.mean(latencies)
    median = statistics.median(latencies)
    peak = max(latencies)
    sub = f"median {_format_latency(median)}  max {_format_latency(peak)}"
    return _stat_card("Latency (avg)", _format_latency(mean), sub)


def _sum_optional(values: object) -> int | None:
    """Sum an iterable that may contain ``None`` entries.

    :param values: Iterable of ``int | None`` values (typed as ``object``
        so it accepts a generator without leaking a complex annotation).
    :type values: object
    :returns: Sum of the non-None integers, or ``None`` if every entry
        was ``None``.
    :rtype: int | None
    """
    total: int | None = None
    for value in values:  # type: ignore[attr-defined]
        if isinstance(value, int) and not isinstance(value, bool):
            total = value if total is None else total + value
    return total


def _render_summary_table(
    summary: Mapping[str, float],
    thresholds: Mapping[str, float],
) -> str:
    """Build the aggregate-summary table with bars, thresholds, and statuses.

    :param summary: Mapping from metric name to aggregate score.
    :type summary: collections.abc.Mapping[str, float]
    :param thresholds: Mapping from metric name to minimum acceptable
        score. Missing keys render as ``-`` in the threshold column and
        omit the status badge.
    :type thresholds: collections.abc.Mapping[str, float]
    :returns: Rendered ``<section id="summary">`` block.
    :rtype: str
    """
    rows = "".join(
        _summary_row(name, summary[name], thresholds.get(name)) for name in sorted(summary)
    )
    return (
        '<section id="summary"><h2>Summary</h2>'
        '<div class="summary-card"><table>'
        "<thead><tr>"
        "<th>Metric</th><th>Score</th><th>Distribution</th>"
        "<th>Threshold</th><th>Status</th>"
        "</tr></thead>"
        f"<tbody>{rows}</tbody>"
        "</table></div></section>"
    )


def _summary_row(name: str, score: float, threshold: float | None) -> str:
    """Build one ``<tr>`` row for the summary table.

    :param name: Metric name.
    :type name: str
    :param score: Aggregate score in ``[0, 1]``.
    :type score: float
    :param threshold: Configured minimum, or ``None`` when the metric has
        no threshold set.
    :type threshold: float | None
    :returns: Rendered table row.
    :rtype: str
    """
    score_class = _score_class(score)
    bar_pct = max(0.0, min(1.0, score)) * 100.0
    if threshold is None:
        status_html = '<span class="muted">-</span>'
        threshold_html = '<span class="muted">-</span>'
    else:
        passed = score >= threshold
        badge_class = "badge-pass" if passed else "badge-fail"
        status_html = f'<span class="badge {badge_class}">{"PASS" if passed else "FAIL"}</span>'
        threshold_html = f'<span class="muted">&gt;= {threshold:.3f}</span>'
    return (
        "<tr>"
        f'<td class="metric-name">{_e(name)}</td>'
        f'<td><span class="{score_class}">{score:.3f}</span></td>'
        f'<td><div class="bar"><div class="fill {score_class.split()[-1]}" '
        f'style="width: {bar_pct:.1f}%"></div></div></td>'
        f"<td>{threshold_html}</td>"
        f"<td>{status_html}</td>"
        "</tr>"
    )


def _render_samples_section(samples: tuple[SampleResult, ...]) -> str:
    """Build the samples section with controls, cards, and empty state.

    :param samples: Per-sample results in run order.
    :type samples: tuple[SampleResult, ...]
    :returns: Rendered ``<section id="samples">`` block.
    :rtype: str
    """
    cards = "".join(_render_sample_card(i, s) for i, s in enumerate(samples))
    return (
        '<section id="samples">'
        '<div class="samples-header">'
        f"<h2>Samples ({len(samples)})</h2>"
        '<div class="controls">'
        '<input type="search" id="sample-search" placeholder="Filter by text...">'
        '<button data-action="expand-all">Expand all</button>'
        '<button data-action="collapse-all">Collapse all</button>'
        "</div>"
        "</div>"
        f"{cards}"
        '<div id="no-results">No samples match the current filter.</div>'
        "</section>"
    )


def _render_sample_card(index: int, sample_result: SampleResult) -> str:
    """Build the collapsible drill-down card for a single sample.

    :param index: Zero-based position of the sample in the run.
    :type index: int
    :param sample_result: Per-sample bundle of sample, response, and
        metric verdicts.
    :type sample_result: SampleResult
    :returns: Rendered ``<details>`` block.
    :rtype: str
    """
    sample = sample_result.sample
    response = sample_result.response
    metrics = sample_result.metrics

    chips = "".join(_score_chip(m) for m in metrics)
    meta_badges = _render_meta_badges(response)
    expected_html = (
        f"<pre>{_e(sample.expected)}</pre>"
        if sample.expected is not None
        else '<div class="empty">no reference answer</div>'
    )
    context_html = (
        "".join(
            f'<div class="context-block"><span class="context-index">[{i}]</span>{_e(chunk)}</div>'
            for i, chunk in enumerate(sample.context)
        )
        if sample.context
        else '<div class="empty">no context provided</div>'
    )
    response_meta_html = _render_response_meta(response)
    metrics_table = _metrics_table(metrics)

    preview = _e(_truncate(sample.input, 120))
    search_text = _e(_searchable_text(sample, response, metrics))
    return (
        f'<details class="sample" data-search-text="{search_text}">'
        "<summary>"
        f'<span class="idx">#{index}</span>'
        f'<span class="input-preview">{preview}</span>'
        f'<span class="meta-badges">{meta_badges}</span>'
        f'<span class="scores">{chips}</span>'
        "</summary>"
        '<div class="body">'
        f'<div class="field"><h3>Input</h3><pre>{_e(sample.input)}</pre></div>'
        f'<div class="field"><h3>Expected</h3>{expected_html}</div>'
        f'<div class="field"><h3>Context ({len(sample.context)})</h3>{context_html}</div>'
        f'<div class="field"><h3>Response</h3>'
        f"<pre>{_e(response.text)}</pre>"
        f"{response_meta_html}"
        "</div>"
        f'<div class="field"><h3>Metrics</h3>{metrics_table}</div>'
        "</div></details>"
    )


def _render_meta_badges(response: Response) -> str:
    """Render the latency and token badges shown in the sample summary row.

    :param response: Response carrying optional latency and token counts.
    :type response: Response
    :returns: Concatenated badge HTML or empty string when nothing to show.
    :rtype: str
    """
    badges: list[str] = []
    if response.latency_ms is not None:
        badges.append(
            f'<span class="badge badge-soft">{_format_latency(response.latency_ms)}</span>'
        )
    total_tokens = (response.prompt_tokens or 0) + (response.completion_tokens or 0)
    if response.prompt_tokens is not None or response.completion_tokens is not None:
        badges.append(f'<span class="badge badge-soft">{total_tokens:,} tok</span>')
    return "".join(badges)


def _render_response_meta(response: Response) -> str:
    """Render the per-response metadata strip below the response text.

    :param response: Response whose finish reason, latency, and token
        counts populate the strip.
    :type response: Response
    :returns: Concatenated badge strip HTML.
    :rtype: str
    """
    items: list[str] = []
    if response.finish_reason:
        items.append(f'<span class="badge badge-soft">finish: {_e(response.finish_reason)}</span>')
    if response.latency_ms is not None:
        items.append(
            f'<span class="badge badge-soft">{_format_latency(response.latency_ms)}</span>'
        )
    if response.prompt_tokens is not None:
        items.append(f'<span class="badge badge-soft">prompt {response.prompt_tokens:,}</span>')
    if response.completion_tokens is not None:
        items.append(
            f'<span class="badge badge-soft">completion {response.completion_tokens:,}</span>'
        )
    if not items:
        return ""
    return f'<div class="response-meta">{"".join(items)}</div>'


def _metrics_table(metrics: tuple[MetricResult, ...]) -> str:
    """Render the per-sample metrics table inside a drill-down card.

    :param metrics: Metric results for one sample, in run order.
    :type metrics: tuple[MetricResult, ...]
    :returns: Rendered table.
    :rtype: str
    """
    rows = "\n".join(
        "<tr>"
        f'<td class="metric-name">{_e(m.metric)}</td>'
        f'<td><span class="{_score_class(m.score)}">{m.score:.3f}</span></td>'
        f'<td class="metric-reason">{_e(m.reason or "")}</td>'
        "</tr>"
        for m in metrics
    )
    return (
        '<div class="metrics-mini"><table>'
        "<thead><tr><th>Metric</th><th>Score</th><th>Reason</th></tr></thead>"
        f"<tbody>{rows}</tbody>"
        "</table></div>"
    )


def _score_chip(metric: MetricResult) -> str:
    """Render the small per-metric chip shown in the sample summary.

    :param metric: Metric result to render.
    :type metric: MetricResult
    :returns: Rendered ``<span>``.
    :rtype: str
    """
    return (
        f'<span class="{_score_class(metric.score)}" '
        f'title="{_e(metric.metric)}">{metric.score:.2f}</span>'
    )


def _score_class(score: float) -> str:
    """Pick the CSS class for a score band.

    :param score: Score in ``[0, 1]``.
    :type score: float
    :returns: One of ``score score-good``, ``score score-ok``,
        ``score score-bad``.
    :rtype: str
    """
    if score >= _GOOD:
        return "score score-good"
    if score >= _OK:
        return "score score-ok"
    return "score score-bad"


def _format_latency(ms: float) -> str:
    """Format a millisecond duration into a compact human-readable string.

    :param ms: Latency in milliseconds.
    :type ms: float
    :returns: Compact representation, e.g. ``"0.5 ms"``, ``"142 ms"``,
        ``"1.5 s"``.
    :rtype: str
    """
    if ms < 1.0:
        return f"{ms:.2f} ms"
    if ms < 1000.0:
        return f"{ms:.0f} ms"
    return f"{ms / 1000.0:.2f} s"


def _format_duration(start_iso: str, end_iso: str) -> str:
    """Compute the duration between two ISO-8601 timestamps.

    :param start_iso: Start timestamp in ISO-8601 format with timezone.
    :type start_iso: str
    :param end_iso: End timestamp in ISO-8601 format with timezone.
    :type end_iso: str
    :returns: Compact duration such as ``"15s"`` or ``"2m 30s"``. Returns
        the empty string if either timestamp is malformed.
    :rtype: str
    """
    try:
        start = datetime.fromisoformat(start_iso)
        end = datetime.fromisoformat(end_iso)
    except ValueError:
        return ""
    delta = (end - start).total_seconds()
    if delta < 0:
        return ""
    if delta < 60:
        return f"{delta:.1f}s" if delta < 10 else f"{delta:.0f}s"
    minutes = int(delta // 60)
    seconds = int(delta % 60)
    return f"{minutes}m {seconds}s"


def _searchable_text(
    sample: object,
    response: Response,
    metrics: tuple[MetricResult, ...],
) -> str:
    """Build the text blob the JS search filter matches against.

    Combines input, response, expected, context chunks, and metric reasons
    so the filter catches any of them.

    :param sample: The :class:`Sample` whose text fields contribute.
    :type sample: object
    :param response: The :class:`Response` text contributes.
    :type response: Response
    :param metrics: Metric results whose names and reasons contribute.
    :type metrics: tuple[MetricResult, ...]
    :returns: Concatenated string used as ``data-search-text``.
    :rtype: str
    """
    parts: list[str] = [
        getattr(sample, "input", "") or "",
        response.text,
        getattr(sample, "expected", "") or "",
    ]
    parts.extend(getattr(sample, "context", ()) or ())
    for metric in metrics:
        parts.append(metric.metric)
        if metric.reason:
            parts.append(metric.reason)
    return " ".join(parts).replace("\n", " ")


def _truncate(text: str, limit: int) -> str:
    """Truncate ``text`` to ``limit`` characters with an ellipsis suffix.

    :param text: Source text.
    :type text: str
    :param limit: Maximum length before truncation.
    :type limit: int
    :returns: Truncated text.
    :rtype: str
    """
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _e(value: str) -> str:
    """Escape ``value`` for embedding in HTML attribute or text content.

    :param value: Untrusted text.
    :type value: str
    :returns: HTML-escaped text.
    :rtype: str
    """
    return _html.escape(value, quote=True)
