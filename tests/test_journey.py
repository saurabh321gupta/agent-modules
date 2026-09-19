"""Tests for `agent_modules.journey` — the single run artefact."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from helpers import element, snapshot

from agent_modules.config import AutomationDefaults
from agent_modules.journey import JourneyLogger, convert_jsonl_to_human, JourneyRenderer


def read_records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_records_carry_the_expected_envelope(tmp_path):
    logger = JourneyLogger(str(tmp_path / "j.jsonl"))
    logger.log("run_started", url="https://example.com")
    logger.close()
    records = read_records(tmp_path / "j.jsonl")
    assert len(records) == 1
    record = records[0]
    assert set(record) == {"event_number", "timestamp", "run_id", "event", "data"}
    assert record["event"] == "run_started"
    assert record["event_number"] == 1
    assert record["data"]["url"] == "https://example.com"


def test_event_numbers_increase(tmp_path):
    logger = JourneyLogger(str(tmp_path / "j.jsonl"))
    logger.log("a")
    logger.log("b")
    logger.log("c")
    logger.close()
    assert [r["event_number"] for r in read_records(tmp_path / "j.jsonl")] == [1, 2, 3]


def test_secret_values_are_redacted(tmp_path):
    logger = JourneyLogger(str(tmp_path / "j.jsonl"), secret_values=["hunter2-secret"])
    logger.log("note", text="the key is hunter2-secret ok")
    logger.close()
    assert "hunter2-secret" not in (tmp_path / "j.jsonl").read_text(encoding="utf-8")
    assert "[REDACTED]" in read_records(tmp_path / "j.jsonl")[0]["data"]["text"]


def test_api_key_named_fields_are_redacted_whatever_the_value(tmp_path):
    logger = JourneyLogger(str(tmp_path / "j.jsonl"))
    logger.log("cfg", api_key="anything", authorization="Bearer x", nested={"token": "z"})
    logger.close()
    data = read_records(tmp_path / "j.jsonl")[0]["data"]
    assert data["api_key"] == "[REDACTED]"
    assert data["authorization"] == "[REDACTED]"
    assert data["nested"]["token"] == "[REDACTED]"


def test_sk_style_keys_are_redacted_without_being_registered(tmp_path):
    logger = JourneyLogger(str(tmp_path / "j.jsonl"))
    logger.log("note", text="using sk-abcdefghijklmnop now")
    logger.close()
    assert "[REDACTED]" in read_records(tmp_path / "j.jsonl")[0]["data"]["text"]


def test_usage_accumulates_across_llm_responses(tmp_path):
    logger = JourneyLogger(str(tmp_path / "j.jsonl"))
    logger.log("llm_response", prompt_tokens=100, completion_tokens=10, cached_tokens=5, reasoning_tokens=2)
    logger.log("llm_response", prompt_tokens=50, completion_tokens=4)
    logger.close()
    assert logger._usage == {
        "prompt_tokens": 150,
        "completion_tokens": 14,
        "cached_tokens": 5,
        "reasoning_tokens": 2,
    }


def test_usage_ignores_non_numeric_and_other_events(tmp_path):
    logger = JourneyLogger(str(tmp_path / "j.jsonl"))
    logger.log("llm_response", prompt_tokens="nonsense")
    logger.log("normalise_response", prompt_tokens=999)
    logger.close()
    assert logger._usage["prompt_tokens"] == 0


def test_pydantic_and_dataclass_payloads_are_serialised(tmp_path):
    logger = JourneyLogger(str(tmp_path / "j.jsonl"))
    logger.log("page_snapshot", snapshot=snapshot([element("e1", label="City")]))
    logger.log("run_started", defaults=AutomationDefaults())
    logger.close()
    records = read_records(tmp_path / "j.jsonl")
    assert records[0]["data"]["snapshot"]["elements"][0]["label"] == "City"
    assert records[1]["data"]["defaults"]["citizenship_country"] == "India"


def test_a_readable_log_is_written_beside_the_jsonl(tmp_path):
    logger = JourneyLogger(str(tmp_path / "j.jsonl"))
    logger.log("run_started", url="https://example.com", model="m", planner="staged")
    logger.close()
    human = (tmp_path / "j.log").read_text(encoding="utf-8")
    assert "EVENT 001" in human
    assert "RUN STARTED" in human
    assert "https://example.com" in human


def test_log_path_with_log_suffix_still_writes_both_files(tmp_path):
    logger = JourneyLogger(str(tmp_path / "run.log"))
    logger.log("run_started", url="u")
    logger.close()
    assert (tmp_path / "run.log").exists()
    assert (tmp_path / "run.jsonl").exists()


def test_run_started_reports_the_field_pause(tmp_path):
    """Regression: without this line a run's artefacts cannot show whether the pause was applied."""
    logger = JourneyLogger(str(tmp_path / "j.jsonl"))
    logger.log("run_started", url="u", field_pause_s=1.0)
    logger.close()
    human = (tmp_path / "j.log").read_text(encoding="utf-8")
    assert "Field pause" in human
    assert "1.0s" in human


def test_run_started_reports_the_hardening_settings(tmp_path):
    logger = JourneyLogger(str(tmp_path / "j.jsonl"))
    logger.log(
        "run_started",
        url="u",
        reasoning_effort="low",
        request_timeout_s=60.0,
        hedge_after_s=30.0,
        max_run_seconds=300.0,
    )
    logger.close()
    human = (tmp_path / "j.log").read_text(encoding="utf-8")
    assert "Reasoning effort  : low" in human
    assert "hedged after 30.0s" in human
    assert "Run budget        : 300.0s" in human


def test_hedge_event_has_a_readable_section(tmp_path):
    logger = JourneyLogger(str(tmp_path / "j.jsonl"))
    logger.log("llm_hedge", model="m", after_s=30.0, note="raced a duplicate")
    logger.close()
    human = (tmp_path / "j.log").read_text(encoding="utf-8")
    assert "SECOND REQUEST RACED" in human
    assert "30.0s" in human


def test_timeout_and_budget_events_have_readable_sections(tmp_path):
    logger = JourneyLogger(str(tmp_path / "j.jsonl"))
    logger.log("llm_timeout", timeout_s=60)
    logger.log("run_timeout", max_run_seconds=300, elapsed_s=301.2)
    logger.close()
    human = (tmp_path / "j.log").read_text(encoding="utf-8")
    assert "TIMED OUT" in human
    assert "EXHAUSTED - DISCARDING WITHOUT SUBMITTING" in human


def test_snapshot_section_lists_the_controls(tmp_path):
    logger = JourneyLogger(str(tmp_path / "j.jsonl"))
    logger.log("page_snapshot", step=1, snapshot=snapshot([element("e1", label="City", required=True)]))
    logger.close()
    human = (tmp_path / "j.log").read_text(encoding="utf-8")
    assert "[e1] TEXTBOX City (required)" in human


def test_hidden_controls_are_not_listed_in_the_readable_snapshot(tmp_path):
    """The readable log shows what a human sees; the JSONL keeps the hidden control."""
    logger = JourneyLogger(str(tmp_path / "j.jsonl"))
    logger.log(
        "page_snapshot",
        step=1,
        snapshot=snapshot(
            [
                element("e9", role="textbox", label="", input_type="file", visible=False),
                element("e10", role="button", label="Upload Resume", input_type="button"),
            ]
        ),
    )
    logger.close()
    human = (tmp_path / "j.log").read_text(encoding="utf-8")
    assert "[e10] BUTTON Upload Resume" in human
    assert "[e9]" not in human


def test_unknown_events_fall_back_to_raw_json(tmp_path):
    logger = JourneyLogger(str(tmp_path / "j.jsonl"))
    logger.log("something_new", alpha=1, beta="two")
    logger.close()
    assert '"alpha": 1' in (tmp_path / "j.log").read_text(encoding="utf-8")


def test_run_started_reports_whether_a_trace_is_being_recorded(tmp_path):
    logger = JourneyLogger(str(tmp_path / "j.jsonl"))
    logger.log("run_started", url="u", trace_path="/tmp/run.zip")
    logger.log("run_started", url="u")
    logger.close()
    human = (tmp_path / "j.log").read_text(encoding="utf-8")
    assert "Playwright trace  : /tmp/run.zip" in human
    assert "Playwright trace  : off" in human


def test_run_started_reports_which_form_payload_is_in_use(tmp_path):
    """The two payloads behave differently, so the trace has to say which one ran."""
    logger = JourneyLogger(str(tmp_path / "j.jsonl"))
    logger.log("run_started", url="u", normalise=True)
    logger.log("run_started", url="u", normalise=False)
    logger.close()
    human = (tmp_path / "j.log").read_text(encoding="utf-8")
    assert "Form payload      : normalised questions" in human
    assert "Form payload      : raw snapshot (normalisation off)" in human


def test_trace_events_have_readable_sections(tmp_path):
    logger = JourneyLogger(str(tmp_path / "j.jsonl"))
    logger.log("trace_started", path="/tmp/run.zip", note="recording everything")
    logger.log("trace_written", path="/tmp/run.zip")
    logger.log("trace_failed", phase="stop", error="disk full")
    logger.close()
    human = (tmp_path / "j.log").read_text(encoding="utf-8")
    assert "Playwright trace   : RECORDING" in human
    assert "Playwright trace   : WRITTEN" in human
    assert "playwright show-trace" in human
    assert "Playwright trace   : FAILED" in human
    assert "disk full" in human


# ----------------------------------------------------------------------------------- timing report


def test_closing_writes_a_timings_report_beside_the_trace(tmp_path):
    logger = JourneyLogger(str(tmp_path / "j.jsonl"))
    logger.log("run_started", url="https://x", planner="staged")
    logger.log("run_finished", result={"status": "success", "steps": 1})
    logger.close()
    report = tmp_path / "j.timing.log"
    assert report.exists()
    content = report.read_text(encoding="utf-8")
    assert "run_started" in content
    assert "outcome=success" in content


def test_the_timing_path_sits_beside_the_other_two_artefacts(tmp_path):
    logger = JourneyLogger(str(tmp_path / "j.jsonl"))
    logger.log("run_started", url="u")
    logger.close()
    assert logger.path.endswith("j.jsonl")
    assert logger.human_path.endswith("j.log")
    assert logger.timing_path.endswith("j.timing.log")


def test_a_custom_log_path_still_yields_a_matching_timing_path(tmp_path):
    logger = JourneyLogger(str(tmp_path / "custom.log"))
    logger.log("run_started", url="u")
    logger.close()
    assert logger.timing_path == str(tmp_path / "custom.timing.log")
    assert (tmp_path / "custom.timing.log").exists()


def test_the_timings_report_is_not_world_readable(tmp_path):
    logger = JourneyLogger(str(tmp_path / "j.jsonl"))
    logger.log("run_started", url="u")
    logger.close()
    assert (tmp_path / "j.timing.log").stat().st_mode & 0o077 == 0


def test_the_timings_report_contains_only_timings(tmp_path):
    """The point of the file: events and durations, with no payloads to wade through."""
    logger = JourneyLogger(str(tmp_path / "j.jsonl"))
    logger.log(
        "llm_request",
        request={"model": "m", "messages": [{"role": "user", "content": "a long prompt " * 50}]},
    )
    logger.log("run_finished", result={"status": "success"})
    logger.close()
    report = (tmp_path / "j.timing.log").read_text(encoding="utf-8")
    assert "a long prompt" not in report
    assert "llm_request" in report


def test_a_failed_timing_render_still_closes_the_logger(tmp_path, monkeypatch):
    """Losing the trace because a derived report failed would be the wrong trade."""
    logger = JourneyLogger(str(tmp_path / "j.jsonl"))
    logger.log("run_started", url="u")

    def boom() -> str:
        raise RuntimeError("cannot render")

    monkeypatch.setattr(logger, "write_timing_report", boom)
    logger.close()
    assert logger._handle.closed
    assert (tmp_path / "j.jsonl").exists()


def test_jsonl_files_are_not_world_readable(tmp_path):
    logger = JourneyLogger(str(tmp_path / "j.jsonl"))
    logger.log("a")
    logger.close()
    assert (tmp_path / "j.jsonl").stat().st_mode & 0o077 == 0


def test_convert_jsonl_to_human_replays_an_existing_trace(tmp_path):
    logger = JourneyLogger(str(tmp_path / "j.jsonl"))
    logger.log("run_started", url="https://example.com")
    logger.log("run_finished", result={"status": "success", "message": "ok", "steps": 3})
    logger.close()
    (tmp_path / "j.log").unlink()

    produced = convert_jsonl_to_human(str(tmp_path / "j.jsonl"))
    assert produced == str(tmp_path / "j.log")
    human = (tmp_path / "j.log").read_text(encoding="utf-8")
    assert "RUN STARTED" in human
    assert "RUN FINISHED" in human
    assert "Run status        : success" in human


def test_renderer_can_be_used_without_a_logger():
    renderer = JourneyRenderer(human_path="h.log", jsonl_path="j.jsonl")
    output = renderer.render(
        {"event_number": 7, "timestamp": "2026-09-18T12:00:00+00:00", "run_id": "x", "event": "plan_rejected", "data": {"error": "expired"}}
    )
    assert "EVENT 007" in output
    assert "expired" in output


def test_run_started_reports_the_thinking_settings(tmp_path):
    """Whether reasoning was on changes both the latency and the quality, so the trace must say."""
    logger = JourneyLogger(str(tmp_path / "j.jsonl"))
    logger.log("run_started", url="u", normaliser_thinking=False, planner_thinking=None)
    logger.log("run_started", url="u", normaliser_thinking=True, planner_thinking=False)
    logger.close()
    human = (tmp_path / "j.log").read_text(encoding="utf-8")
    assert "Thinking          : normaliser off, planner provider default" in human
    assert "Thinking          : normaliser on, planner off" in human


if __name__ == "__main__":  # pragma: no cover - a convenience, not a test path
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
