"""Self-contained HTML report writer for human inspection of run results."""

from __future__ import annotations

import html as _html
from pathlib import Path

from llm_evals.core.eval_run import RunResult, SampleResult
from llm_evals.metrics.base import MetricResult

_GOOD = 0.8
_OK = 0.5

_STYLE = """
:root {
  color-scheme: light dark;
  --bg: #0f1115;
  --panel: #161922;
  --panel-2: #1d2230;
  --border: #2a3040;
  --text: #e8ebf1;
  --muted: #97a2b8;
  --good: #22c55e;
  --ok: #eab308;
  --bad: #ef4444;
  --link: #60a5fa;
  --mono: ui-monospace, "SF Mono", "JetBrains Mono", Consolas, monospace;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, sans-serif;
}
header {
  border-bottom: 1px solid var(--border);
  padding: 18px 24px;
  background: var(--panel);
}
header h1 { margin: 0 0 4px; font-size: 18px; font-weight: 600; }
header .meta { color: var(--muted); font-size: 13px; font-family: var(--mono); }
main { max-width: 1100px; margin: 0 auto; padding: 24px; }
section { margin-bottom: 32px; }
section > h2 {
  font-size: 13px;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--muted);
  margin: 0 0 12px;
}
table {
  width: 100%;
  border-collapse: collapse;
  background: var(--panel);
  border-radius: 8px;
  overflow: hidden;
  border: 1px solid var(--border);
}
th, td {
  text-align: left;
  padding: 10px 14px;
  border-bottom: 1px solid var(--border);
  font-family: var(--mono);
  font-size: 13px;
}
tr:last-child td { border-bottom: none; }
th { color: var(--muted); font-weight: 500; background: var(--panel-2); }
.score {
  display: inline-block;
  padding: 2px 8px;
  border-radius: 999px;
  font-family: var(--mono);
  font-size: 12px;
  font-weight: 600;
}
.score-good { background: rgba(34,197,94,0.15); color: var(--good); }
.score-ok   { background: rgba(234,179,8,0.15);  color: var(--ok); }
.score-bad  { background: rgba(239,68,68,0.15);  color: var(--bad); }
details.sample {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 8px;
  margin-bottom: 8px;
  overflow: hidden;
}
details.sample[open] { background: var(--panel-2); }
details.sample > summary {
  cursor: pointer;
  padding: 12px 16px;
  display: flex;
  align-items: center;
  gap: 12px;
  list-style: none;
}
details.sample > summary::-webkit-details-marker { display: none; }
details.sample > summary::before {
  content: "\\203A";
  color: var(--muted);
  transition: transform 0.15s;
}
details.sample[open] > summary::before { transform: rotate(90deg); }
.idx {
  color: var(--muted);
  font-family: var(--mono);
  font-size: 12px;
  flex-shrink: 0;
}
.input-preview {
  flex: 1;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.scores { display: flex; gap: 6px; flex-shrink: 0; }
.body { padding: 0 16px 16px 30px; }
.field { margin-top: 14px; }
.field > h3 {
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--muted);
  margin: 0 0 6px;
}
.field pre {
  font-family: var(--mono);
  font-size: 13px;
  white-space: pre-wrap;
  word-break: break-word;
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 10px 12px;
  margin: 0;
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
}
.context-index { color: var(--muted); margin-right: 8px; }
.empty { color: var(--muted); font-style: italic; }
@media (prefers-color-scheme: light) {
  :root {
    --bg: #f7f8fa;
    --panel: #ffffff;
    --panel-2: #f0f2f6;
    --border: #d8dde6;
    --text: #1a1c23;
    --muted: #5b6478;
  }
}
"""


def write_html(result: RunResult, path: Path) -> None:
    """Render ``result`` as a single self-contained HTML file.

    The output embeds CSS inline so the report can be emailed, attached to
    a PR, or hosted on a static page without external assets.

    :param result: Aggregate run result to render.
    :type result: RunResult
    :param path: Destination file path. Existing content is overwritten.
    :type path: pathlib.Path
    """
    path.write_text(_render_html(result), encoding="utf-8")


def _render_html(result: RunResult) -> str:
    """Compose the full HTML document for a run result.

    :param result: Aggregate run result to render.
    :type result: RunResult
    :returns: Complete HTML document.
    :rtype: str
    """
    summary_rows = "\n".join(
        _summary_row(name, score) for name, score in sorted(result.summary.items())
    )
    sample_cards = "\n".join(_sample_card(index, item) for index, item in enumerate(result.samples))
    manifest = result.manifest
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en"><head>'
        '<meta charset="utf-8">'
        f"<title>llm-evals run {_e(manifest.run_id)}</title>"
        f"<style>{_STYLE}</style>"
        "</head><body>"
        "<header>"
        f"<h1>Run {_e(manifest.run_id)}</h1>"
        f'<div class="meta">{_e(manifest.provider)} / {_e(manifest.model)} '
        f"&middot; {manifest.sample_count} samples &middot; "
        f"{_e(manifest.started_at)}</div>"
        "</header>"
        "<main>"
        '<section id="summary"><h2>Summary</h2>'
        "<table><thead><tr><th>Metric</th><th>Score</th></tr></thead>"
        f"<tbody>{summary_rows}</tbody></table></section>"
        '<section id="samples"><h2>Samples</h2>'
        f"{sample_cards}"
        "</section>"
        "</main></body></html>"
    )


def _summary_row(metric_name: str, score: float) -> str:
    """Build one ``<tr>`` for the aggregate summary table.

    :param metric_name: Name of the metric.
    :type metric_name: str
    :param score: Aggregate score in ``[0, 1]``.
    :type score: float
    :returns: Rendered table row.
    :rtype: str
    """
    return (
        f"<tr><td>{_e(metric_name)}</td>"
        f'<td><span class="{_score_class(score)}">{score:.3f}</span></td></tr>'
    )


def _sample_card(index: int, sample_result: SampleResult) -> str:
    """Build the collapsible drill-down card for a single sample.

    :param index: Zero-based position of the sample in the run.
    :type index: int
    :param sample_result: Per-sample result with response and metric verdicts.
    :type sample_result: SampleResult
    :returns: Rendered ``<details>`` block.
    :rtype: str
    """
    sample = sample_result.sample
    response = sample_result.response
    metrics = sample_result.metrics

    chips = "".join(_score_chip(m) for m in metrics)
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
    metrics_table = _metrics_table(metrics)

    preview = _e(_truncate(sample.input, 120))
    return (
        '<details class="sample"><summary>'
        f'<span class="idx">#{index}</span>'
        f'<span class="input-preview">{preview}</span>'
        f'<span class="scores">{chips}</span>'
        "</summary>"
        '<div class="body">'
        f'<div class="field"><h3>Input</h3><pre>{_e(sample.input)}</pre></div>'
        f'<div class="field"><h3>Expected</h3>{expected_html}</div>'
        f'<div class="field"><h3>Context</h3>{context_html}</div>'
        f'<div class="field"><h3>Response</h3><pre>{_e(response.text)}</pre></div>'
        f'<div class="field"><h3>Metrics</h3>{metrics_table}</div>'
        "</div></details>"
    )


def _metrics_table(metrics: tuple[MetricResult, ...]) -> str:
    """Render the per-sample metrics table inside a drill-down card.

    :param metrics: Metric results for one sample, in run order.
    :type metrics: tuple[MetricResult, ...]
    :returns: Rendered table.
    :rtype: str
    """
    rows = "\n".join(
        f"<tr><td>{_e(m.metric)}</td>"
        f'<td><span class="{_score_class(m.score)}">{m.score:.3f}</span></td>'
        f"<td>{_e(m.reason or '')}</td></tr>"
        for m in metrics
    )
    return (
        "<table><thead><tr><th>Metric</th><th>Score</th><th>Reason</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )


def _score_chip(metric: MetricResult) -> str:
    """Render the small per-metric chip shown in the sample summary.

    :param metric: Metric result to render.
    :type metric: MetricResult
    :returns: HTML span.
    :rtype: str
    """
    return (
        f'<span class="score {_score_class(metric.score)}" '
        f'title="{_e(metric.metric)}">{metric.score:.2f}</span>'
    )


def _score_class(score: float) -> str:
    """Pick the CSS class for a score band.

    :param score: Score in ``[0, 1]``.
    :type score: float
    :returns: One of ``score-good``, ``score-ok``, ``score-bad``.
    :rtype: str
    """
    if score >= _GOOD:
        return "score score-good"
    if score >= _OK:
        return "score score-ok"
    return "score score-bad"


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
    return text[: limit - 1] + "…"


def _e(value: str) -> str:
    """Escape ``value`` for embedding in HTML attribute or text content.

    :param value: Untrusted text.
    :type value: str
    :returns: HTML-escaped text.
    :rtype: str
    """
    return _html.escape(value, quote=True)
