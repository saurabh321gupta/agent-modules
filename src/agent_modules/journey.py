"""The run trace: an exact JSONL record plus a readable narrative. Layer 0.

Every module may log here. Nothing may log anywhere else, so a run has exactly one artefact and the
readable log can always be regenerated from the JSONL.
"""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from .timing import render_timing

SECRET_PATTERN = re.compile(r"(?i)(?:sk-[A-Za-z0-9_-]{10,}|bearer\s+[A-Za-z0-9._~+/=-]+)")
SECRET_KEYS = {
    "api_key",
    "apikey",
    "authorization",
    "password",
    "secret",
    "token",
}

#: Events whose payload means the same thing rendered in the same way.
_FIELD_ASSIST_EVENTS = frozenset(
    {
        "large_option_request",
        "large_option_resolved",
        "large_option_rejected",
        "large_option_skipped",
        "large_option_failed",
        "field_repair_request",
        "field_repaired",
        "field_repair_failed",
        "field_repair_skipped",
    }
)


class JourneyRenderer:
    """Renders one recorded event as a readable section.

    Split out from the logger so an existing JSONL can be rendered without constructing a logger.
    """

    def __init__(
        self,
        *,
        human_path: str,
        jsonl_path: str,
        usage: dict[str, int] | None = None,
        started_at: float | None = None,
        elapsed_s: float | None = None,
    ) -> None:
        self.human_path = human_path
        self.path = jsonl_path
        self._usage = usage if usage is not None else {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cached_tokens": 0,
            "reasoning_tokens": 0,
        }
        self._started_at = started_at if started_at is not None else time.monotonic()
        self._elapsed_s = elapsed_s

    # ------------------------------------------------------------------ entry point

    def render(self, record: dict[str, Any]) -> str:
        event_number = record["event_number"]
        event = record["event"]
        data = record["data"]
        timestamp = datetime.fromisoformat(record["timestamp"]).astimezone(ZoneInfo("Asia/Kolkata"))
        stamp = timestamp.strftime("%Y-%m-%d %H:%M:%S %Z")
        title = event.replace("_", " ").upper()
        lines = ["=" * 88, f"EVENT {event_number:03d}  |  {stamp}  |  {title}", "=" * 88]

        lines.extend(self._section(event, data))
        return "\n".join(lines) + "\n\n"

    def _section(self, event: str, data: dict[str, Any]) -> list[str]:
        """Render one event's body. Events without a bespoke view fall through to raw JSON."""
        if event == "run_started":
            return self._run_started(data)
        if event == "page_snapshot":
            return self._format_snapshot(data.get("snapshot", {}), data.get("step"))
        if event == "llm_request":
            return self._format_llm_request(data.get("request", {}))
        if event == "llm_response":
            return self._llm_response(data)
        if event == "plan_received":
            return self._plan(data, validated=False)
        if event == "plan_validated":
            return self._plan(data, validated=True)
        if event == "action_start":
            return ["Action about to run:"] + self._format_action(data.get("action", {}), 1)
        if event == "action_result":
            return self._action_result(data)
        if event == "run_finished":
            return self._run_finished(data)
        if event == "step_timing":
            return self._step_timing(data)
        if event == "page_classified":
            return self._page_classified(data)
        if event == "normalised_form":
            return self._normalised_form(data)
        if event == "overlay_cleanup":
            return self._overlay_cleanup(data)
        if event == "jev_response":
            return self._jev_response(data)
        if event == "jev_request":
            return [
                f"Model            : {data.get('model', '')}",
                f"Questions        : {data.get('question_count', '')}",
                f"State size       : {data.get('state_bytes', '')} bytes",
                "Questions asked:",
                *[f"  - {name}" for name in (data.get("questions") or [])],
            ]
        if event in _FIELD_ASSIST_EVENTS:
            return self._field_assist(data)
        if event == "plan_rejected":
            return [
                "Validation result  : REJECTED",
                f"Reason             : {data.get('error', '')}",
            ]
        if event == "plan_retry":
            return [
                "Validation result  : REJECTED - RETRYING",
                f"Reason             : {data.get('error', '')}",
                f"Recovery attempt   : {data.get('attempt', '')}",
                str(data.get("reason", "")),
            ]
        if event == "plan_reduced":
            return [
                "Validation result  : PARTIALLY ACCEPTED",
                f"Reason             : {data.get('reason', '')}",
                f"Executing          : {data.get('remaining', '')} of the planned actions",
            ]
        if event == "plan_stale":
            return [
                "Planner response   : IGNORED AS STALE/CONTRADICTORY",
                f"Reason             : {data.get('reason', '')}",
                f"Recovery attempt   : {data.get('stale_count', '')}",
                "The live page already matched every requested action; the runner re-observed instead of stopping.",
            ]
        if event == "run_exception":
            return [f"Runner exception   : {data.get('error', '')}"]
        if event == "field_reverted":
            return [
                "Field rewritten   : THE PAGE CHANGED THIS FIELD AFTER TYPING",
                f"Field             : {data.get('label') or data.get('target')} ({data.get('target')})",
                f"We wrote          : {data.get('wrote')!r}",
                f"Now reads         : {data.get('after_blur')!r}",
                str(data.get("note", "")),
            ]
        if event == "classify_request":
            return [
                f"Controls offered : {data.get('button_count', '')} button(s) plus page context",
                "Classification request (exact payload):",
            ] + self._indented(data.get("payload"), 2)
        if event == "classify_failed":
            return [
                "Classification    : FAILED",
                f"Error             : {data.get('error', '')}",
                str(data.get("note", "")),
            ]
        if event == "submitted_verdict_rejected":
            return [
                "Completion        : REJECTED (not certain enough)",
                f"Verdict confidence: {data.get('confidence')}",
                f"Submitted evidence: {data.get('submitted_evidence')}",
                str(data.get("note", "")),
            ]
        if event == "normalise_request":
            return ["Normaliser request (exact payload):"] + self._indented(data.get("request"), 2)
        if event == "normalise_response":
            return [
                f"Model             : {data.get('model', '')}",
                f"Latency           : {data.get('latency_s', '')}s",
                f"Tokens            : prompt={data.get('prompt_tokens')} "
                f"completion={data.get('completion_tokens')} cached={data.get('cached_tokens')}",
                "Raw response (verbatim):",
                str(data.get("raw_response", "")),
            ]
        if event == "schema_fallback":
            return [
                "Response format    : STRICT JSON SCHEMA UNAVAILABLE - FALLING BACK",
                f"Provider said      : {data.get('error', '')}",
                str(data.get("note", "")),
            ]
        if event == "plan_malformed":
            return [
                "Planner response   : MALFORMED - RETRYING",
                f"Reason             : {data.get('error', '')}",
            ]
        if event == "empty_snapshot":
            return [
                "Page render        : NOTHING YET - RE-OBSERVING",
                f"Recovery attempt   : {data.get('attempt', '')}",
                str(data.get("reason", "")),
            ]
        if event == "llm_timeout":
            return [
                "Planner call       : TIMED OUT",
                f"Timeout            : {data.get('timeout_s', '')}s",
                str(data.get("note", "")),
            ]
        if event == "llm_hedge":
            return [
                "Slow call          : SECOND REQUEST RACED",
                f"Triggered after    : {data.get('after_s', '')}s with no response",
                str(data.get("note", "")),
            ]
        if event == "run_timeout":
            return [
                "Run budget         : EXHAUSTED - DISCARDING WITHOUT SUBMITTING",
                f"Budget             : {data.get('max_run_seconds', '')}s",
                f"Elapsed            : {data.get('elapsed_s', '')}s",
            ]
        if event == "resume_upload_chosen":
            return [
                "Resume upload      : ATTACHING BEFORE ANSWERING",
                f"Target             : {data.get('field') or data.get('target')} ({data.get('target')})",
                f"Asset              : {data.get('asset_id', '')}",
                f"Jev confidence     : {data.get('confidence', '')}",
                f"Page class         : {data.get('page_class', '')}",
                str(data.get("note", "")),
            ]
        if event == "resume_upload_skipped":
            return [
                "Resume upload      : SKIPPED",
                f"Target             : {data.get('field', '')}",
                f"Reason             : {data.get('reason', '')}",
            ]
        if event == "trace_started":
            return [
                "Playwright trace   : RECORDING",
                f"Path               : {data.get('path', '')}",
                str(data.get("note", "")),
            ]
        if event == "trace_written":
            return [
                "Playwright trace   : WRITTEN",
                f"Path               : {data.get('path', '')}",
                "Replay it with 'playwright show-trace <path>'.",
            ]
        if event == "trace_failed":
            return [
                "Playwright trace   : FAILED",
                f"Phase              : {data.get('phase', '')}",
                f"Error              : {data.get('error', '')}",
            ]
        return self._indented(data, 0)

    # -------------------------------------------------------------------- sections

    def _run_started(self, data: dict[str, Any]) -> list[str]:
        planner = data.get("planner") or "llm"
        lines = [
            f"Target URL       : {data.get('url', '')}",
            f"Planner          : {planner}"
            + (f" ({data.get('jev_model')})" if planner == "jev" else f" ({data.get('model', '')})"),
            f"Model            : {data.get('jev_model') or data.get('model', '')}",
            f"Provider endpoint : {data.get('base_url', '')}",
            f"Browser           : {'headless' if data.get('headless') else 'visible'}",
            f"Page zoom         : {data.get('zoom', 1.0)}",
            f"Step limit        : {data.get('max_steps', '')}",
            f"Final submission  : {'ALLOWED' if data.get('allow_submission') else 'BLOCKED'}",
            f"Upload assets     : {', '.join(data.get('asset_ids', [])) or 'none'}",
            f"Answer bank       : {data.get('answers', 0)} answers ({data.get('answers_path') or 'none'})",
            # Without this, a run's artefacts cannot confirm whether the pause was applied at all.
            f"Field pause       : {data.get('field_pause_s', 0.0)}s after typing",
            f"Form payload      : "
            f"{'normalised questions' if data.get('normalise', True) else 'raw snapshot (normalisation off)'}",
            f"Reasoning effort  : {data.get('reasoning_effort') or 'provider default'}",
            f"Thinking          : normaliser "
            f"{'on' if data.get('normaliser_thinking', False) else 'off'}, "
            f"planner "
            + (
                "on"
                if data.get("planner_thinking") is True
                else "off"
                if data.get("planner_thinking") is False
                else "provider default"
            ),
            f"Call timeout      : {data.get('request_timeout_s', '')}s "
            f"(hedged after {data.get('hedge_after_s', '')}s)",
            f"Run budget        : {data.get('max_run_seconds', '')}s",
            f"Playwright trace  : {data.get('trace_path') or 'off'}",
            "Defaults:",
        ]
        lines.extend(self._indented(data.get("defaults"), 2))
        return lines

    def _llm_response(self, data: dict[str, Any]) -> list[str]:
        lines = [
            f"Model            : {data.get('model', '')}",
            f"Finish reason    : {data.get('finish_reason', '')}",
            f"Latency          : {data.get('latency_s', '')}s",
            f"Tokens           : prompt={data.get('prompt_tokens')} completion={data.get('completion_tokens')} "
            f"cached={data.get('cached_tokens')} reasoning={data.get('reasoning_tokens')}",
            "Model response:",
        ]
        lines.extend(self._indented(self._pretty(data.get("response_text", "")), 2))
        return lines

    def _run_finished(self, data: dict[str, Any]) -> list[str]:
        result = data.get("result") or {}
        elapsed = self._elapsed_s if self._elapsed_s is not None else time.monotonic() - self._started_at
        return [
            f"Run status        : {result.get('status', '')}",
            f"Message           : {result.get('message', '')}",
            f"Page URL          : {result.get('url', '')}",
            f"Steps completed   : {result.get('steps', '')}",
            f"Total run time    : {elapsed:.1f}s",
            f"Model tokens      : prompt={self._usage['prompt_tokens']} "
            f"completion={self._usage['completion_tokens']} "
            f"cached={self._usage['cached_tokens']} reasoning={self._usage['reasoning_tokens']}",
            f"Readable log      : {result.get('journey_log') or self.human_path}",
            f"Detailed JSONL    : {result.get('journey_jsonl') or self.path}",
            f"Timings           : {result.get('journey_timing') or ''}",
        ]

    def _plan(self, data: dict[str, Any], *, validated: bool) -> list[str]:
        plan = data.get("plan", {})
        lines = [
            f"Application step  : {data.get('step')}",
            f"Plan status       : {plan.get('status', '')}",
            f"Snapshot used     : {plan.get('snapshot_id', '')}",
            f"Reason            : {plan.get('reason', '')}",
            "Planned actions:",
        ]
        actions = plan.get("actions", [])
        if not actions:
            lines.append("  (none)")
        for index, action in enumerate(actions, 1):
            lines.extend(self._format_action(action, 1, index))
        if validated:
            lines.append("Validation result    : accepted by the safety validator")
        return lines

    def _action_result(self, data: dict[str, Any]) -> list[str]:
        result = data.get("result", {})
        lines = [
            f"Action outcome    : {'SUCCESS' if result.get('ok') else 'FAILED'}",
            f"Message           : {result.get('message', '')}",
        ]
        lines.extend(self._format_action(result.get("action", {}), 0))
        return lines

    def _field_assist(self, data: dict[str, Any]) -> list[str]:
        lines = [f"Field            : {data.get('field') or ', '.join(data.get('fields') or [])}"]
        if data.get("option_count") is not None:
            lines.append(f"Options offered  : {data.get('option_count')} (too many for a Jev Choice)")
        if data.get("option_value"):
            lines.append(f"Chosen option    : {data.get('option_value')}")
        if data.get("replacement"):
            lines.append(f"Repaired value   : {data.get('replacement')!r}  (was {data.get('was')!r})")
        if data.get("error"):
            lines.append(f"Site error       : {data.get('error')}")
        if data.get("model_returned") is not None:
            lines.append(f"Model returned   : {data.get('model_returned')}")
        if data.get("latency_s") is not None:
            lines.append(f"Latency          : {data.get('latency_s')}s")
        if data.get("reason"):
            lines.append(f"Reason           : {data.get('reason')}")
        return lines

    def _jev_response(self, data: dict[str, Any]) -> list[str]:
        lines = [
            f"Model            : {data.get('model', '')}",
            f"Latency          : {data.get('latency_s', '')}s",
            f"Usage            : {data.get('usage')}",
            "Answers:",
        ]
        for name, answer in (data.get("answers") or {}).items():
            if not isinstance(answer, dict):
                continue
            if answer.get("type") == "choice":
                lines.append(f"  {name}: {answer.get('choice')}  (confidence {answer.get('confidence')})")
            elif answer.get("type") == "noul":
                lines.append(f"  {name}: {answer.get('noul')}")
            else:
                lines.append(f"  {name}: {answer.get('score')}")
        return lines

    def _step_timing(self, data: dict[str, Any]) -> list[str]:
        lines = [f"Step              : {data.get('step', '')}"]
        lines.append(f"  {'total':<12}: {data.get('total_ms', '')} ms")
        for name, value in data.items():
            if name in {"step", "total_ms"}:
                continue
            if isinstance(value, (int, float)):
                lines.append(f"  {name.removesuffix('_ms'):<12}: {value} ms")
            else:
                lines.append(f"  {name.replace('_', ' '):<12}: {value}")
        return lines

    def _page_classified(self, data: dict[str, Any]) -> list[str]:
        decision = data.get("decision") or {}
        probabilities = decision.get("probabilities") or {}
        runner_up = sorted(
            ((name, value) for name, value in probabilities.items() if name != decision.get("page_class")),
            key=lambda item: -float(item[1]),
        )[:2]
        return [
            f"Verdict           : {decision.get('page_class', '')} "
            f"(confidence {float(decision.get('confidence') or 0):.2f})",
            f"Runner-up         : {', '.join(f'{n} {float(v):.2f}' for n, v in runner_up) or 'none'}",
            f"Next control      : {decision.get('control_label') or 'none'} "
            f"({float(decision.get('control_confidence') or 0):.2f})",
            f"Resume upload     : "
            + (
                f"{decision.get('resume_control_id')} "
                f"(confidence {float(decision.get('resume_confidence') or 0):.2f})"
                if decision.get("resume_control_id")
                else "none offered"
            ),
            f"Submitted evidence: {float(decision.get('submitted_evidence') or 0):.2f}",
            f"Latency           : {data.get('latency_s', '')}s",
            f"Usage             : {data.get('usage')}",
        ]

    def _normalised_form(self, data: dict[str, Any]) -> list[str]:
        if data.get("cached"):
            lines = [f"Normalised form   : reused from cache ({data.get('field_count', '')} fields)"]
        else:
            lines = [
                f"Page kind         : {data.get('page_kind', '')}",
                f"Fields            : {data.get('field_count', '')}",
                f"Invented ids      : {data.get('dropped_ids') or 'none'}",
            ]
        lines.append(f"{'id':<6}{'type':<13}{'req':<5}{'group':<26}question")
        for field in data.get("form") or []:
            lines.append(
                f"{field.get('id', ''):<6}{str(field.get('field_type', '')):<13}"
                f"{('yes' if field.get('required') else 'no'):<5}"
                f"{('[' + str(field['group']) + ']') if field.get('group') else '':<26}"
                f"{str(field.get('question', ''))[:78]}"
            )
        return lines

    def _overlay_cleanup(self, data: dict[str, Any]) -> list[str]:
        lines = ["Automatically closed non-essential UI overlays before observing the page:"]
        for action in data.get("actions", []):
            lines.append(f"  - {action.get('kind', 'overlay')}: {action.get('label', '')}")
        return lines

    # ---------------------------------------------------------------------- helpers

    @staticmethod
    def _pretty(value: Any) -> Any:
        if not isinstance(value, str):
            return value
        try:
            return json.dumps(json.loads(value), indent=2, ensure_ascii=False)
        except (TypeError, json.JSONDecodeError):
            return value

    @staticmethod
    def _indented(value: Any, amount: int) -> list[str]:
        rendered = (
            value
            if isinstance(value, str)
            else json.dumps(value, ensure_ascii=False, indent=2, default=str)
        )
        prefix = " " * amount
        return [prefix + line if line else line for line in rendered.splitlines()]

    def _format_snapshot(self, snapshot: dict[str, Any], step: Any) -> list[str]:
        elements = snapshot.get("elements", [])
        lines = [
            f"Application step  : {step}",
            f"Page title        : {snapshot.get('title', '')}",
            f"URL               : {snapshot.get('url', '')}",
            f"Snapshot ID       : {snapshot.get('snapshot_id', '')}",
            f"Visible controls  : {len(elements)}",
        ]
        if snapshot.get("validation_errors"):
            lines.append(f"Validation errors : {', '.join(snapshot['validation_errors'])}")
        if snapshot.get("blockers"):
            lines.append(f"Blockers          : {', '.join(snapshot['blockers'])}")
        visible_text = [str(item) for item in snapshot.get("visible_text", []) if str(item).strip()]
        if visible_text:
            lines.append("Visible page text:")
            for item in visible_text[:18]:
                lines.append(f"  - {item}")
            if len(visible_text) > 18:
                lines.append(
                    f"  ... {len(visible_text) - 18} more lines (see JSONL for the complete snapshot)"
                )
        lines.append("Interactive controls:")
        for element in elements:
            if not element.get("visible", True):
                continue
            state = []
            if element.get("required"):
                state.append("required")
            if element.get("value") not in (None, ""):
                state.append(f"value={element['value']!r}")
            if element.get("checked") is not None:
                state.append(f"checked={element['checked']}")
            if element.get("options"):
                state.append(f"{len(element['options'])} options")
            if element.get("file_attached"):
                state.append("file attached")
            suffix = f" ({'; '.join(state)})" if state else ""
            label = str(element.get("label") or "").replace("\n", " ")
            lines.append(f"  [{element.get('id')}] {str(element.get('role', '')).upper()} {label}{suffix}")
        return lines

    def _format_llm_request(self, request: dict[str, Any]) -> list[str]:
        lines = [
            "Exact model input (rendered as readable sections; the complete request is also in the JSONL):",
            f"Model            : {request.get('model', '')}",
            f"Reasoning effort : {request.get('reasoning_effort') or 'provider default'}",
            "Messages:",
        ]
        for index, message in enumerate(request.get("messages", []), 1):
            lines.append(f"  MESSAGE {index} [{message.get('role', 'unknown').upper()}]")
            content = message.get("content", "")
            if isinstance(content, str):
                try:
                    content = json.dumps(json.loads(content), ensure_ascii=False, indent=2)
                except json.JSONDecodeError:
                    pass
            lines.extend(self._indented(content, 4))
        response_format = request.get("response_format")
        if response_format:
            schema = response_format.get("json_schema", {}) if isinstance(response_format, dict) else {}
            lines.extend(
                [
                    f"Response format   : {response_format.get('type', '') if isinstance(response_format, dict) else ''}",
                    f"Schema name       : {schema.get('name', '')}",
                    "Full response schema is preserved in the JSONL trace.",
                ]
            )
        return lines

    @staticmethod
    def _format_action(action: dict[str, Any], indent: int, index: int | None = None) -> list[str]:
        prefix = " " * (indent * 2)
        number = f"{index}. " if index is not None else ""
        kind = str(action.get("type", "")).upper()
        target = action.get("target") or "(no target)"
        details = []
        for key in ("value_ref", "option_value", "asset_id", "checked", "value"):
            if action.get(key) is not None:
                details.append(f"{key}={action[key]!r}")
        suffix = f"; {', '.join(details)}" if details else ""
        return [f"{prefix}{number}{kind} -> {target}{suffix}"]


class JourneyLogger:
    """Write a detailed JSONL trace and a readable narrative for each run."""

    def __init__(self, path: str | None = None, secret_values: list[str] | None = None) -> None:
        run_id = uuid.uuid4().hex[:12]
        self.run_id = run_id
        self._started_at = time.monotonic()
        self._usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cached_tokens": 0,
            "reasoning_tokens": 0,
        }
        if path:
            requested = Path(path)
            if requested.suffix.lower() == ".log":
                self.human_path = str(requested)
                self.path = str(requested.with_suffix(".jsonl"))
            else:
                self.path = str(requested)
                self.human_path = str(requested.with_suffix(".log"))
        else:
            stem = Path("run-artifacts") / f"journey-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{run_id}"
            self.path = str(stem.with_suffix(".jsonl"))
            self.human_path = str(stem.with_suffix(".log"))
        # The timings sit beside the other two artefacts and are derived from the JSONL at close,
        # so they always agree with the trace.
        self.timing_path = str(Path(self.path).with_suffix("")) + ".timing.log"

        jsonl_path = Path(self.path)
        human_path = Path(self.human_path)
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        human_path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = jsonl_path.open("a", encoding="utf-8")
        self._human_handle = human_path.open("a", encoding="utf-8")
        os.chmod(jsonl_path, 0o600)
        os.chmod(human_path, 0o600)
        self._secret_values = [value for value in (secret_values or []) if value]
        self._event_number = 0
        self._renderer = JourneyRenderer(
            human_path=self.human_path,
            jsonl_path=self.path,
            usage=self._usage,
            started_at=self._started_at,
        )

    def log(self, event: str, **data: Any) -> None:
        self._event_number += 1
        if event == "llm_response":
            for name in self._usage:
                value = data.get(name)
                if isinstance(value, (int, float)):
                    self._usage[name] += value
        sanitized = self._sanitize(data)
        record = {
            "event_number": self._event_number,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run_id": self.run_id,
            "event": event,
            "data": sanitized,
        }
        self._handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        self._handle.flush()
        self._human_handle.write(self._renderer.render(record))
        self._human_handle.flush()

    def close(self) -> None:
        # Render the timings from the JSONL before closing it: the file is the single source of
        # truth for every artefact, and reading it back keeps the report impossible to drift.
        try:
            self.write_timing_report()
        except Exception:
            # A report that cannot be produced must never be the reason a run's trace is lost.
            pass
        if not self._handle.closed:
            self._handle.close()
        if not self._human_handle.closed:
            self._human_handle.close()

    def write_timing_report(self) -> str:
        """Write the timings-only report beside the trace, and return its path."""
        self._handle.flush()
        records = [
            json.loads(line)
            for line in Path(self.path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        destination = Path(self.timing_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(render_timing(records), encoding="utf-8")
        os.chmod(destination, 0o600)
        return self.timing_path

    def _sanitize(self, value: Any, key: str | None = None) -> Any:
        if isinstance(value, dict):
            return {
                name: "[REDACTED]" if name.lower() in SECRET_KEYS else self._sanitize(item, name)
                for name, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [self._sanitize(item, key) for item in value]
        if isinstance(value, str):
            sanitized = value
            for secret in self._secret_values:
                sanitized = sanitized.replace(secret, "[REDACTED]")
            return SECRET_PATTERN.sub("[REDACTED]", sanitized)
        if hasattr(value, "model_dump"):
            return self._sanitize(value.model_dump(mode="json"), key)
        if is_dataclass(value):
            return self._sanitize(asdict(value), key)
        return value


def convert_jsonl_to_human(jsonl_path: str, human_path: str | None = None) -> str:
    """Convert an existing JSONL journey into the same readable format used by new runs."""
    source = Path(jsonl_path)
    destination = Path(human_path) if human_path else source.with_suffix(".log")
    destination.parent.mkdir(parents=True, exist_ok=True)
    records = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    elapsed = None
    if len(records) >= 2:
        first = datetime.fromisoformat(records[0]["timestamp"])
        last = datetime.fromisoformat(records[-1]["timestamp"])
        elapsed = (last - first).total_seconds()
    renderer = JourneyRenderer(
        human_path=str(destination),
        jsonl_path=str(source),
        elapsed_s=elapsed,
    )
    with destination.open("w", encoding="utf-8") as output_handle:
        for record in records:
            output_handle.write(renderer.render(record))
    os.chmod(destination, 0o600)
    return str(destination)
