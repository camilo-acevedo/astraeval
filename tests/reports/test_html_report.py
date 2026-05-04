"""Tests for :mod:`astraea.reports.html_report`."""

from __future__ import annotations

from pathlib import Path

from astraea.core.eval_run import EvalRun, RunResult
from astraea.datasets.sample import Sample
from astraea.metrics.exact_match import ExactMatch
from astraea.providers.fake import FakeProvider
from astraea.reports.html_report import write_html


def _execute_run() -> RunResult:
    """Run a tiny end-to-end evaluation for the HTML renderer tests.

    :returns: A fully populated :class:`RunResult`.
    :rtype: RunResult
    """
    provider = FakeProvider(["yes", "no"])
    samples = [
        Sample(input="What is the capital of France?", expected="yes", context=("Paris is.",)),
        Sample(input="Q2", expected="yes"),
    ]
    return EvalRun(provider, samples, [ExactMatch()], model="m").execute()


def test_write_html_creates_self_contained_document(tmp_path: Path) -> None:
    """The output is a single HTML file containing inline CSS."""
    out = tmp_path / "report.html"
    result = _execute_run()

    write_html(result, out)

    body = out.read_text(encoding="utf-8")
    assert body.startswith("<!DOCTYPE html>")
    assert "</html>" in body
    assert "<style>" in body
    assert 'href="' not in body or "https://" not in body  # no external CSS


def test_html_report_includes_summary_and_samples(tmp_path: Path) -> None:
    """The summary table and per-sample drill-down both appear in the output."""
    out = tmp_path / "report.html"
    result = _execute_run()

    write_html(result, out)

    body = out.read_text(encoding="utf-8")
    assert 'id="summary"' in body
    assert 'id="samples"' in body
    assert "exact_match" in body
    assert "What is the capital of France?" in body


def test_html_report_escapes_user_input(tmp_path: Path) -> None:
    """Sample text is HTML-escaped to prevent rendering issues or XSS."""
    out = tmp_path / "report.html"
    provider = FakeProvider(["plain"])
    samples = [Sample(input="<script>alert('xss')</script>", expected="plain")]
    result = EvalRun(provider, samples, [ExactMatch()], model="m").execute()

    write_html(result, out)

    body = out.read_text(encoding="utf-8")
    assert "<script>alert" not in body
    assert "&lt;script&gt;alert" in body


def test_score_classes_chosen_by_band(tmp_path: Path) -> None:
    """Good/ok/bad scores receive distinct CSS classes."""
    out = tmp_path / "report.html"
    provider = FakeProvider(["yes", "no", "no", "no", "no"])
    samples = [Sample(input=f"Q{i}", expected="yes") for i in range(5)]
    result = EvalRun(provider, samples, [ExactMatch()], model="m").execute()

    write_html(result, out)

    body = out.read_text(encoding="utf-8")
    assert "score-bad" in body


def test_html_report_handles_sample_without_context(tmp_path: Path) -> None:
    """Samples without context render an explicit empty-state, not a blank section."""
    out = tmp_path / "report.html"
    result = _execute_run()

    write_html(result, out)

    body = out.read_text(encoding="utf-8")
    assert "no context provided" in body


def test_html_report_handles_sample_without_expected(tmp_path: Path) -> None:
    """Samples without ``expected`` render an explicit empty-state."""
    out = tmp_path / "report.html"
    provider = FakeProvider(["whatever"])
    samples = [Sample(input="Q")]

    result = EvalRun(provider, samples, [], model="m").execute()
    write_html(result, out)

    body = out.read_text(encoding="utf-8")
    assert "no reference answer" in body


def test_html_report_includes_inline_script(tmp_path: Path) -> None:
    """The report embeds a vanilla JS block for search and expand controls."""
    out = tmp_path / "report.html"
    result = _execute_run()

    write_html(result, out)

    body = out.read_text(encoding="utf-8")
    assert "<script>" in body
    assert 'data-action="expand-all"' in body
    assert 'id="sample-search"' in body


def test_html_report_renders_threshold_badges(tmp_path: Path) -> None:
    """When thresholds are passed, the summary table shows pass/fail badges."""
    out = tmp_path / "report.html"
    provider = FakeProvider(["yes", "yes"])
    samples = [Sample(input="Q1", expected="yes"), Sample(input="Q2", expected="yes")]
    result = EvalRun(provider, samples, [ExactMatch()], model="m").execute()

    write_html(result, out, thresholds={"exact_match": 0.5})

    body = out.read_text(encoding="utf-8")
    assert "PASS" in body
    assert "&gt;= 0.500" in body


def test_html_report_renders_failed_threshold(tmp_path: Path) -> None:
    """Scores below the configured threshold render with a FAIL badge."""
    out = tmp_path / "report.html"
    provider = FakeProvider(["wrong", "wrong"])
    samples = [Sample(input="Q1", expected="right"), Sample(input="Q2", expected="right")]
    result = EvalRun(provider, samples, [ExactMatch()], model="m").execute()

    write_html(result, out, thresholds={"exact_match": 0.9})

    body = out.read_text(encoding="utf-8")
    assert "FAIL" in body


def test_html_report_renders_run_id_copy_button(tmp_path: Path) -> None:
    """The header carries a copy button bound to the run id."""
    out = tmp_path / "report.html"
    result = _execute_run()

    write_html(result, out)

    body = out.read_text(encoding="utf-8")
    assert f'data-copy="{result.manifest.run_id}"' in body


def test_html_report_includes_aggregate_stats(tmp_path: Path) -> None:
    """The stats section reports sample count, tokens, and latency cards."""
    out = tmp_path / "report.html"
    result = _execute_run()

    write_html(result, out)

    body = out.read_text(encoding="utf-8")
    assert 'class="stats"' in body
    assert ">Samples<" in body
    assert ">Total tokens<" in body
    assert "Latency" in body


def test_html_report_includes_sample_meta_badges(tmp_path: Path) -> None:
    """Each sample card shows latency in its summary row when reported."""
    out = tmp_path / "report.html"
    result = _execute_run()

    write_html(result, out)

    body = out.read_text(encoding="utf-8")
    assert "ms" in body
    assert 'class="meta-badges"' in body
