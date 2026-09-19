"""A timings-only view of a run. Layer 0.

Derived from the same JSONL the run already writes rather than instrumented separately, for one
reason: it cannot drift. Anything measured twice eventually disagrees, and a timing report that
disagrees with the trace is worse than no report.

Deliberately narrow. No payloads, no prompts, no reasons - just what happened and how long it took,
in order, so the answer to "where did the run spend its time" is readable at a glance.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

#: Steps that took longer than this are called out, because they are where a run is usually losing.
SLOW_MS = 5_000

#: Detail for these is pulled from the tokens the provider reported, not from wall-clock deltas.
_MODEL_RESPONSE_EVENTS = ("llm_response", "normalise_response", "jev_response")


def _short(value: Any, limit: int = 58) -> str:
    text = "" if value is None else str(value)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _action_label(action: dict[str, Any]) -> str:
    """The same shape the readable log uses, so the two files agree on what an action looks like."""
    kind = str(action.get("type", "")).upper()
    target = action.get("target") or "(no target)"
    details = [
        f"{key}={action[key]!r}"
        for key in ("value_ref", "option_value", "asset_id", "checked", "value")
        if action.get(key) is not None
    ]
    suffix = f"; {', '.join(details)}" if details else ""
    return f"{kind} -> {target}{suffix}"


def _detail(event: str, data: dict[str, Any]) -> str:
    """One short phrase per event. Long enough to identify it, short enough to stay a table."""
    if event == "run_started":
        return _short(data.get("url", ""))
    if event == "page_snapshot":
        count = len((data.get("snapshot") or {}).get("elements") or [])
        return f"step {data.get('step')} · {count} controls"
    if event == "overlay_cleanup":
        return f"{len(data.get('actions') or [])} closed"
    if event == "classify_request":
        return f"{data.get('button_count', 0)} button(s)"
    if event == "classify_failed":
        return _short(data.get("error"))
    if event == "page_classified":
        decision = data.get("decision") or {}
        return f"{decision.get('page_class', '')} ({float(decision.get('confidence') or 0):.2f})"
    if event == "normalise_request":
        request = data.get("request") or {}
        messages = request.get("messages") or []
        controls = 0
        if messages:
            try:
                controls = len(json.loads(messages[-1]["content"]).get("controls") or [])
            except Exception:
                controls = 0
        return f"{controls} control(s)"
    if event == "normalised_form":
        cached = "cached" if data.get("cached") else "fresh"
        return f"{data.get('field_count', 0)} field(s) · {cached}"
    if event in _MODEL_RESPONSE_EVENTS:
        return f"{data.get('prompt_tokens', '?')}+{data.get('completion_tokens', '?')} tokens"
    if event == "llm_request":
        return _short(data.get("request", {}).get("model", ""))
    if event == "plan_received":
        plan = data.get("plan") or {}
        return f"{plan.get('status', '')} · {len(plan.get('actions') or [])} action(s)"
    if event == "plan_validated":
        return "accepted"
    if event == "plan_rejected":
        return _short(data.get("error"))
    if event == "plan_retry":
        return f"attempt {data.get('attempt', '')}: {_short(data.get('error'), 38)}"
    if event == "plan_reduced":
        return f"{data.get('remaining', '')} action(s) kept"
    if event == "plan_stale":
        return f"contradiction #{data.get('stale_count', '')}"
    if event == "action_start":
        return _action_label(data.get("action") or {})
    if event == "action_result":
        result = data.get("result") or {}
        outcome = "SUCCESS" if result.get("ok") else "FAILED"
        return f"{outcome} {_action_label(result.get('action') or {})}"
    if event == "step_timing":
        decision = data.get("decision")
        return f"step {data.get('step', '')}" + (f" · {decision}" if decision else "")
    if event == "llm_hedge":
        return f"after {data.get('after_s', '')}s"
    if event == "llm_timeout":
        return f"{data.get('timeout_s', '')}s"
    if event == "trace_failed":
        written = data.get("bytes_written")
        kept = f"{int(written) / 1024:.0f} KB kept" if written else "nothing written"
        return f"{data.get('phase', '')} · {kept}"
    if event in ("trace_started", "trace_written"):
        return _short(data.get("path"))
    if event == "field_reverted":
        return _short(data.get("label") or data.get("target"))
    if event == "run_finished":
        result = data.get("result") or {}
        return f"{result.get('status', '')} · {result.get('steps', '')} step(s)"
    if event in ("run_timeout", "run_exception"):
        return _short(data.get("error") or data.get("max_run_seconds"))
    return ""


def _moment(record: dict[str, Any]) -> datetime:
    return datetime.fromisoformat(record["timestamp"])


def _seconds_between(start: datetime, end: datetime) -> float:
    return max(0.0, (end - start).total_seconds())


def _totals(records: list[dict[str, Any]]) -> tuple[float, float, int, int, float]:
    """Model seconds, browser seconds and their counts, plus the run's own duration.

    Model time comes from the latency the provider call itself reported. Browser time is measured
    from the gap between an action starting and its result, which is the honest figure because the
    event stream brackets exactly that.
    """
    model_s = 0.0
    model_calls = 0
    browser_s = 0.0
    actions = 0
    pending = None
    for record in records:
        event = record["event"]
        data = record.get("data") or {}
        if event in _MODEL_RESPONSE_EVENTS:
            latency = data.get("latency_s")
            if isinstance(latency, (int, float)):
                model_s += float(latency)
                model_calls += 1
        elif event == "action_start":
            pending = _moment(record)
        elif event == "action_result":
            if pending is not None:
                browser_s += _seconds_between(pending, _moment(record))
                actions += 1
                pending = None
    run_s = 0.0
    if len(records) >= 2:
        run_s = _seconds_between(_moment(records[0]), _moment(records[-1]))
    return model_s, browser_s, model_calls, actions, run_s


def _step_rows(records: list[dict[str, Any]]) -> list[str]:
    rows = []
    for record in records:
        if record["event"] != "step_timing":
            continue
        data = record.get("data") or {}
        parts = []
        for key, label in (
            ("settle_ms", "settle"),
            ("observe_ms", "observe"),
            ("plan_ms", "plan"),
            ("validate_ms", "validate"),
            ("execute_ms", "execute"),
        ):
            if key in data:
                parts.append(f"{label} {float(data[key]) / 1000:6.2f}s")
        total_ms = float(data.get("total_ms") or 0)
        marker = "  <- slow" if total_ms > SLOW_MS else ""
        rows.append(
            f"  step {str(data.get('step', '')):>2}   total {total_ms / 1000:6.2f}s   "
            + "   ".join(parts)
            + marker
        )
    return rows


def render_timing(records: list[dict[str, Any]]) -> str:
    """Render a timings-only report from recorded events. Pure: same input, same output."""
    if not records:
        return "TIMING  (no events recorded)\n"

    started = next((r["data"] for r in records if r["event"] == "run_started"), {}) or {}
    finished = next((r["data"] for r in reversed(records) if r["event"] == "run_finished"), {}) or {}
    result = finished.get("result") or {}
    model_s, browser_s, model_calls, actions, run_s = _totals(records)

    header = [
        f"TIMING  {records[0].get('run_id', '')}",
        "  planner={planner}   model={model}   budget={budget}s   steps={steps}   "
        "outcome={outcome}".format(
            planner=started.get("planner", ""),
            model=started.get("model", ""),
            budget=started.get("max_run_seconds", ""),
            steps=result.get("steps", ""),
            outcome=result.get("status", "incomplete"),
        ),
        "",
    ]

    base = _moment(records[0])
    rows = ["   t+      delta   event                 detail"]
    previous = 0.0
    for record in records:
        elapsed = _seconds_between(base, _moment(record))
        rows.append(
            f"{elapsed:7.3f} {elapsed - previous:8.3f}   {record['event']:<20}  "
            f"{_detail(record['event'], record.get('data') or {})}"
        )
        previous = elapsed

    body = ["", "  ── steps " + "─" * 66]
    body.extend(_step_rows(records) or ["  (none recorded)"])

    other = max(0.0, run_s - model_s - browser_s)
    share = (lambda value: f"{value / run_s * 100:5.1f}%" if run_s else "   n/a")
    totals = [
        "",
        "  ── where the time went " + "─" * 55,
        f"  model       {model_s:8.2f}s  {share(model_s)}   {model_calls} call(s)",
        f"  browser     {browser_s:8.2f}s  {share(browser_s)}   {actions} action(s)",
        f"  unaccounted {other:8.2f}s  {share(other)}",
        f"  {'run':<11} {run_s:8.2f}s",
        "",
    ]

    return "\n".join(header + rows + body + totals)


def convert_jsonl_to_timing(jsonl_path: str, timing_path: str | None = None) -> str:
    """Regenerate the timing report for an existing run, the same way the readable log can be."""
    source = Path(jsonl_path)
    destination = Path(timing_path) if timing_path else source.with_suffix(".timing.log")
    destination.parent.mkdir(parents=True, exist_ok=True)
    records = [
        json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    destination.write_text(render_timing(records), encoding="utf-8")
    return str(destination)
