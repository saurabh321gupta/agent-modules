"""Tests for `agent_modules.timing` — the timings-only view of a run."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from agent_modules.timing import convert_jsonl_to_timing, render_timing

BASE = datetime(2026, 9, 19, 12, 0, 0, tzinfo=timezone.utc)


def record(event: str, offset_s: float, **data) -> dict:
    return {
        "event_number": 1,
        "timestamp": (BASE + timedelta(seconds=offset_s)).isoformat(),
        "run_id": "run-abc",
        "event": event,
        "data": data,
    }


def run_with(*records) -> str:
    return render_timing(list(records))


# ----------------------------------------------------------------------------------------- basics


def test_an_empty_run_is_handled():
    assert "no events recorded" in render_timing([])


def test_the_header_names_the_run_and_its_configuration():
    output = run_with(
        record("run_started", 0.0, url="https://x", planner="staged", model="deepseek-flash",
               max_run_seconds=300),
        record("run_finished", 5.0, result={"status": "success", "steps": 2}),
    )
    assert "TIMING  run-abc" in output
    assert "planner=staged" in output
    assert "model=deepseek-flash" in output
    assert "budget=300s" in output
    assert "outcome=success" in output


def test_every_event_appears_in_order_with_its_elapsed_time():
    """This is the whole point: the sequence, with a clock against it."""
    output = run_with(
        record("run_started", 0.0, url="https://x"),
        record("page_snapshot", 1.5, step=1, snapshot={"elements": []}),
        record("run_finished", 4.0, result={"status": "success", "steps": 1}),
    )
    lines = [line for line in output.splitlines() if line.strip().startswith(("0.", "1.", "4."))]
    assert lines[0].split()[1] == "0.000"
    assert "page_snapshot" in lines[1] and "1.500" in lines[1]
    assert "run_finished" in lines[2] and "4.000" in lines[2]


def test_the_delta_column_shows_the_gap_from_the_previous_event():
    output = run_with(
        record("run_started", 0.0, url="https://x"),
        record("page_snapshot", 2.0, step=1, snapshot={}),
    )
    row = next(line for line in output.splitlines() if "page_snapshot" in line)
    columns = row.split()
    assert columns[0] == "2.000"
    assert columns[1] == "2.000", "the first gap is the whole elapsed time"


# ------------------------------------------------------------------------------------ detail text


def test_a_snapshot_row_reports_the_step_and_control_count():
    output = run_with(record("page_snapshot", 0.0, step=3, snapshot={"elements": [1, 2, 3]}))
    assert "step 3 · 3 controls" in output


def test_a_classification_row_reports_the_verdict_and_confidence():
    output = run_with(
        record(
            "page_classified",
            0.0,
            decision={"page_class": "job_listing", "confidence": 0.95},
        )
    )
    assert "job_listing (0.95)" in output


def test_a_model_response_row_reports_tokens():
    output = run_with(
        record("llm_response", 0.0, latency_s=12.5, prompt_tokens=100, completion_tokens=20)
    )
    assert "100+20 tokens" in output


def test_action_rows_show_what_was_done():
    output = run_with(
        record("action_start", 0.0, action={"type": "fill", "target": "e1", "value": "Bangalore"}),
        record(
            "action_result",
            0.5,
            result={"ok": True, "action": {"type": "fill", "target": "e1", "value": "Bangalore"}},
        ),
    )
    assert "FILL -> e1" in output
    assert "SUCCESS FILL -> e1" in output


def test_a_failed_action_is_visible_at_a_glance():
    output = run_with(
        record(
            "action_result",
            0.0,
            result={"ok": False, "message": "boom", "action": {"type": "click", "target": "e9"}},
        )
    )
    assert "FAILED CLICK -> e9" in output


def test_a_slow_step_is_flagged():
    """The reason to open this file is usually one step that took far too long."""
    output = run_with(record("step_timing", 0.0, step=2, total_ms=31000, plan_ms=29000))
    assert "<- slow" in output


def test_a_normal_step_is_not_flagged():
    output = run_with(record("step_timing", 0.0, step=1, total_ms=2000, observe_ms=200))
    assert "<- slow" not in output


def test_a_long_url_is_truncated_so_the_table_stays_a_table():
    output = run_with(record("run_started", 0.0, url="https://example.com/" + "x" * 300))
    row = next(line for line in output.splitlines() if "run_started" in line)
    assert row.endswith("…")


# ----------------------------------------------------------------------------------------- totals


def test_model_time_sums_the_reported_latencies():
    output = run_with(
        record("llm_response", 0.0, latency_s=10.0, prompt_tokens=1, completion_tokens=1),
        record("jev_response", 17.0, latency_s=1.0),
        record("run_finished", 20.0, result={"status": "success"}),
    )
    assert "model          11.00s" in output
    assert "2 call(s)" in output


def test_browser_time_is_measured_from_action_start_to_result():
    """The event stream brackets exactly that, so it is the honest figure rather than an estimate."""
    output = run_with(
        record("run_started", 0.0, url="https://x"),
        record("action_start", 1.0, action={"type": "click", "target": "e1"}),
        record("action_result", 3.5, result={"ok": True, "action": {"type": "click", "target": "e1"}}),
        record("run_finished", 4.0, result={"status": "success"}),
    )
    assert "browser         2.50s" in output
    assert "1 action(s)" in output


def test_the_three_buckets_add_up_to_the_run():
    output = run_with(
        record("run_started", 0.0, url="https://x"),
        record("llm_response", 0.0, latency_s=4.0, prompt_tokens=1, completion_tokens=1),
        record("action_start", 4.0, action={"type": "click", "target": "e1"}),
        record("action_result", 6.0, result={"ok": True, "action": {"type": "click", "target": "e1"}}),
        record("run_finished", 10.0, result={"status": "success"}),
    )
    assert "model           4.00s" in output
    assert "browser         2.00s" in output
    assert "unaccounted     4.00s" in output
    assert "run            10.00s" in output


def test_percentages_are_shown_against_the_run():
    output = run_with(
        record("run_started", 0.0, url="https://x"),
        record("llm_response", 0.0, latency_s=6.0, prompt_tokens=1, completion_tokens=1),
        record("run_finished", 10.0, result={"status": "success"}),
    )
    assert "60.0%" in output


def test_a_single_event_run_does_not_divide_by_zero():
    output = render_timing([record("run_started", 0.0, url="https://x")])
    assert "n/a" in output


def test_an_unfinished_run_is_labelled_as_incomplete():
    output = run_with(record("run_started", 0.0, url="https://x"), record("page_snapshot", 1.0, snapshot={}))
    assert "outcome=incomplete" in output


# ------------------------------------------------------------------------------------ the steps


def test_the_step_table_breaks_each_phase_out():
    output = run_with(
        record(
            "step_timing",
            0.0,
            step=1,
            total_ms=4200,
            settle_ms=300,
            observe_ms=400,
            plan_ms=3300,
            validate_ms=0,
            execute_ms=500,
        )
    )
    assert "step  1" in output
    assert "total   4.20s" in output
    assert "settle   0.30s" in output
    assert "observe   0.40s" in output
    assert "plan   3.30s" in output
    assert "execute   0.50s" in output


def test_settle_is_reported_because_it_is_usually_the_unattributed_time():
    """Without it, the largest single wait on a slow page shows up as unexplained time."""
    output = run_with(record("step_timing", 0.0, step=1, total_ms=6000, settle_ms=5100))
    assert "settle   5.10s" in output


def test_a_trace_that_failed_but_wrote_something_says_so():
    """A partial trace from a crashed run is exactly when you want to look at one."""
    output = run_with(record("trace_failed", 0.0, phase="stop", error="driver died", bytes_written=65536))
    assert "stop · 64 KB kept" in output


def test_a_trace_that_wrote_nothing_says_so_plainly():
    output = run_with(record("trace_failed", 0.0, phase="stop", error="driver died", bytes_written=0))
    assert "nothing written" in output


def test_no_step_timings_says_so_rather_than_printing_an_empty_table():
    assert "(none recorded)" in run_with(record("run_started", 0.0))


# -------------------------------------------------------------------------------------- the file


def test_the_report_can_be_regenerated_from_an_existing_trace(tmp_path):
    """Same contract as the readable log: the JSONL is the truth and every view is derived."""
    trace = tmp_path / "j.jsonl"
    trace.write_text(
        "\n".join(
            json.dumps(r)
            for r in (
                record("run_started", 0.0, url="https://x", planner="staged"),
                record("run_finished", 3.0, result={"status": "success", "steps": 1}),
            )
        )
        + "\n",
        encoding="utf-8",
    )
    produced = convert_jsonl_to_timing(str(trace))
    assert produced == str(tmp_path / "j.timing.log")
    content = (tmp_path / "j.timing.log").read_text(encoding="utf-8")
    assert "TIMING  run-abc" in content
    assert "outcome=success" in content


def test_rendering_is_deterministic():
    """A pure transform over recorded events: the same records must give the same report."""
    records = [
        record("run_started", 0.0, url="https://x"),
        record("llm_response", 1.0, latency_s=2.0, prompt_tokens=3, completion_tokens=4),
        record("run_finished", 5.0, result={"status": "blocked"}),
    ]
    assert render_timing(records) == render_timing(records)


def test_no_secret_can_appear_because_only_whitelisted_fields_are_rendered():
    """Detail text is built from named fields, never by dumping the payload, so a token cannot leak.

    This is the structural reason the timings file is safe to share: it never echoes a request body.
    """
    output = run_with(
        record(
            "run_started",
            0.0,
            url="https://x",
            api_key="sk-should-never-appear",
            request={"messages": [{"role": "user", "content": "sk-also-secret"}]},
        )
    )
    assert "sk-should-never-appear" not in output
    assert "sk-also-secret" not in output


if __name__ == "__main__":  # pragma: no cover - a convenience, not a test path
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
