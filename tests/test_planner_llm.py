"""Tests for `agent_modules.planner_llm`."""

from __future__ import annotations

import json

import pytest
from fakes import FakeModelClient, thinking_of
from helpers import element, profile, snapshot

from agent_modules.config import AutomationDefaults
from agent_modules.journey import JourneyLogger
from agent_modules.planner_llm import LLMPlanner

APPLICANT = profile()
ASSETS = {"resume_primary": "/tmp/resume.pdf"}


def action_json(kind: str = "fill", target: str = "e1", **extra) -> dict:
    base = {
        "type": kind,
        "target": target,
        "value": None,
        "value_ref": None,
        "option_value": None,
        "checked": None,
        "asset_id": None,
    }
    base.update(extra)
    return base


def plan_json(actions=None, status: str = "continue", **extra) -> str:
    payload = {
        "status": status,
        "actions": actions or [],
        "reason": "because",
        "completion_evidence": None,
    }
    payload.update(extra)
    return json.dumps(payload)


def make(replies, **kwargs) -> tuple[FakeModelClient, LLMPlanner]:
    client = FakeModelClient(replies)
    planner = LLMPlanner(
        client,
        "model",
        ASSETS,
        AutomationDefaults(),
        answer_bank={"Expected CTC": "8000000"},
        **kwargs,
    )
    return client, planner


async def test_a_plan_is_returned_from_the_model_reply():
    client, planner = make(plan_json([action_json(value_ref="identity.first_name")]))
    plan = await planner.next_step(APPLICANT, snapshot([element("e1", label="Given Name(s)")]))
    assert plan.status == "continue"
    assert plan.actions[0].value_ref == "identity.first_name"


async def test_last_step_reports_the_action_count():
    client, planner = make(plan_json([action_json(), action_json(target="e2")]))
    await planner.next_step(APPLICANT, snapshot([element("e1", label="A"), element("e2", label="B")]))
    assert planner.last_step["actions"] == 2


# --------------------------------------------------------------------------------------- payload


def test_the_payload_is_exactly_the_three_things_it_should_be():
    """Profile, approved answers, page. Everything else that used to ride along is gone, and this is
    the test that keeps it gone."""
    _, planner = make(plan_json())
    payload = planner.payload(APPLICANT, snapshot([element("e1", label="City")]))
    assert set(payload) == {"candidate_profile", "user_provided_answers", "current_page"}


def test_the_payload_carries_no_history():
    _, planner = make(plan_json())
    payload = planner.payload(APPLICANT, snapshot([element("e1", label="City")]))
    assert "recent_history" not in payload


def test_the_payload_carries_no_snapshot_id():
    """Nothing the model has to echo back, so nothing it can get wrong by echoing."""
    _, planner = make(plan_json())
    payload = planner.payload(APPLICANT, snapshot([element("e1", label="City")]))
    assert "snapshot_id" not in payload
    assert "snapshot_id" not in payload["current_page"]


def test_the_payload_carries_no_normalised_form():
    _, planner = make(plan_json())
    payload = planner.payload(APPLICANT, snapshot([element("e1", label="City")]))
    assert "form" not in payload


def test_the_page_is_sent_as_the_reader_saw_it():
    _, planner = make(plan_json())
    snap = snapshot(
        [element("e1", label="Phone", value="91-0000000000", invalid=True)],
        visible_text=["My Information"],
        validation_errors=["Phone Number is not in a valid format"],
    )
    payload = planner.payload(APPLICANT, snap)
    assert payload["current_page"]["validation_errors"] == [
        "Phone Number is not in a valid format"
    ]
    assert payload["current_page"]["elements"][0]["invalid"] is True
    assert payload["current_page"]["elements"][0]["value"] == "91-0000000000"


async def test_the_payload_carries_the_profile_and_the_answer_bank():
    client, planner = make(plan_json())
    await planner.next_step(APPLICANT, snapshot([element("e1", label="City")]))
    sent = json.loads(client.calls[0]["messages"][1]["content"])
    assert sent["candidate_profile"]["identity"]["first_name"] == "Alex"
    assert sent["user_provided_answers"] == {"Expected CTC": "8000000"}
    assert sent["current_page"]["elements"][0]["label"] == "City"


def test_local_paths_never_appear_in_the_payload():
    _, planner = make(plan_json())
    encoded = json.dumps(planner.payload(APPLICANT, snapshot([element("e1", label="City")])))
    assert "/tmp/resume.pdf" not in encoded
    assert "resume_primary" in encoded


def test_the_payload_is_json_serialisable():
    _, planner = make(plan_json())
    json.dumps(planner.payload(APPLICANT, snapshot([element("e1", label="City")])))


# ------------------------------------------------------------------------------------ schema mode


async def test_a_native_schema_prompt_does_not_restate_the_schema():
    client, planner = make(plan_json())
    await planner.next_step(APPLICANT, snapshot([element("e1", label="City")]))
    system = client.calls[0]["messages"][0]["content"]
    assert '"title": "ApplicationPlan"' not in system


async def test_json_mode_restates_the_schema_in_the_prompt():
    """A provider that refuses json_schema still has to be told the exact shape."""
    client, planner = make(plan_json())
    client.schema_mode = "json_object"
    await planner.next_step(APPLICANT, snapshot([element("e1", label="City")]))
    system = client.calls[0]["messages"][0]["content"]
    assert "no markdown fences" in system
    assert '"title": "ApplicationPlan"' in system


# ------------------------------------------------------------------------------------ malformed


async def test_a_malformed_plan_is_retried_once():
    client, planner = make(["not json", plan_json([action_json()])])
    plan = await planner.next_step(APPLICANT, snapshot([element("e1", label="City")]))
    assert len(plan.actions) == 1
    assert len(client.calls) == 2


async def test_a_malformed_plan_is_logged(tmp_path):
    logger = JourneyLogger(str(tmp_path / "j.jsonl"))
    client, planner = make(["nonsense", plan_json()], journey=logger)
    await planner.next_step(APPLICANT, snapshot([element("e1", label="City")]))
    logger.close()
    events = [json.loads(line)["event"] for line in (tmp_path / "j.jsonl").read_text().splitlines() if line]
    assert "plan_malformed" in events
    assert "MALFORMED - RETRYING" in (tmp_path / "j.log").read_text()


async def test_a_second_malformed_plan_raises():
    client, planner = make(["nonsense", "still nonsense"])
    with pytest.raises(Exception):
        await planner.next_step(APPLICANT, snapshot([element("e1", label="City")]))


async def test_a_transport_failure_is_surfaced_rather_than_swallowed():
    """A structural error is not a parse error; the run must not loop on it."""
    client = FakeModelClient(raises=RuntimeError("connection reset"))
    planner = LLMPlanner(client, "model", ASSETS, AutomationDefaults())
    with pytest.raises(RuntimeError, match="connection reset"):
        await planner.next_step(APPLICANT, snapshot([element("e1", label="City")]))
    assert len(client.calls) == 1, "a transport failure must not be retried as a malformed plan"


# -------------------------------------------------------------------------------------- logging


async def test_the_request_and_response_are_logged(tmp_path):
    logger = JourneyLogger(str(tmp_path / "j.jsonl"))
    client, planner = make(plan_json([action_json()]), journey=logger)
    await planner.next_step(APPLICANT, snapshot([element("e1", label="City")]))
    logger.close()
    events = [json.loads(line)["event"] for line in (tmp_path / "j.jsonl").read_text().splitlines() if line]
    assert events == ["llm_request", "llm_response"]
    human = (tmp_path / "j.log").read_text(encoding="utf-8")
    assert "MESSAGE 1 [SYSTEM]" in human
    assert "Tokens           : prompt=11 completion=22" in human


async def test_a_plan_is_not_logged_as_received_by_the_planner():
    """The orchestrator owns plan_received; logging it here too would double every plan in the trace."""
    logger_calls: list[str] = []

    class RecordingLogger:
        def log(self, event, **data):
            logger_calls.append(event)

    client, planner = make(plan_json(), journey=RecordingLogger())
    await planner.next_step(APPLICANT, snapshot([element("e1", label="City")]))
    assert "plan_received" not in logger_calls


# ---------------------------------------------------------------------------------------- thinking


async def test_the_answering_call_leaves_thinking_to_the_provider_by_default():
    """Unset is not the same as off.

    Answering a form involves real judgement - matching a fact to a question, choosing between
    similar options - so it is left alone until someone has measured that turning it off holds up.
    """
    client, planner = make(plan_json())
    await planner.next_step(APPLICANT, snapshot([element("e1", label="City")]))
    assert thinking_of(client.calls[0]) is None


async def test_the_answering_call_can_switch_thinking_off():
    """For measuring the trade: the same run with reasoning off, compared on the same page."""
    client = FakeModelClient(plan_json())
    planner = LLMPlanner(client, "model", ASSETS, AutomationDefaults(), thinking=False)
    await planner.next_step(APPLICANT, snapshot([element("e1", label="City")]))
    assert thinking_of(client.calls[0]) is False


if __name__ == "__main__":  # pragma: no cover - a convenience, not a test path
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
