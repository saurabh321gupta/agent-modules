"""Tests for `agent_modules.planner_llm`."""

from __future__ import annotations

import json

import pytest
from fakes import FakeModelClient
from helpers import element, profile, snapshot

from agent_modules.config import AutomationDefaults
from agent_modules.journey import JourneyLogger
from agent_modules.planner_llm import LLMPlanner
from agent_modules.types import NormalisedField, NormalisedForm

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


def plan_json(actions=None, status: str = "continue", snapshot_id: str = "s-1-abc12345", **extra) -> str:
    payload = {
        "snapshot_id": snapshot_id,
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
    plan = await planner.next_step(APPLICANT, snapshot([element("e1", label="Given Name(s)")]), [])
    assert plan.status == "continue"
    assert plan.actions[0].value_ref == "identity.first_name"


async def test_last_step_reports_the_action_count():
    client, planner = make(plan_json([action_json(), action_json(target="e2")]))
    await planner.next_step(APPLICANT, snapshot([element("e1", label="A"), element("e2", label="B")]), [])
    assert planner.last_step["actions"] == 2


# --------------------------------------------------------------------------------------- payload


async def test_the_payload_carries_profile_answers_and_page():
    client, planner = make(plan_json())
    await planner.next_step(APPLICANT, snapshot([element("e1", label="City")]), [])
    sent = json.loads(client.calls[0]["messages"][1]["content"])
    assert sent["candidate_profile"]["identity"]["first_name"] == "Alex"
    assert sent["user_provided_answers"] == {"Expected CTC": "8000000"}
    assert sent["current_page"]["elements"][0]["label"] == "City"


async def test_local_paths_never_appear_in_the_payload():
    client, planner = make(plan_json())
    await planner.next_step(APPLICANT, snapshot([element("e1", label="City")]), [])
    assert "/tmp/resume.pdf" not in client.calls[0]["messages"][1]["content"]
    assert "resume_primary" in client.calls[0]["messages"][1]["content"]


async def test_history_is_trimmed_to_the_last_six_entries():
    client, planner = make(plan_json())
    history = [{"snapshot_id": f"s{n}"} for n in range(20)]
    await planner.next_step(APPLICANT, snapshot([element("e1", label="City")]), history)
    sent = json.loads(client.calls[0]["messages"][1]["content"])
    assert [h["snapshot_id"] for h in sent["recent_history"]] == [f"s{n}" for n in range(14, 20)]


async def test_a_normalised_form_replaces_the_raw_page_in_the_payload():
    """The staged path sends questions rather than controls: smaller, and easier to answer."""
    client, planner = make(plan_json())
    form = NormalisedForm(
        page_kind="form",
        fields=[
            NormalisedField(
                id="e1", question="Which city do you live in?", field_type="text", required=True
            )
        ],
        submit_controls=["e9"],
    )
    snap = snapshot([element("e1", label="City", value="Bangalore")])
    await planner.next_step(APPLICANT, snap, [], form=form)
    sent = json.loads(client.calls[0]["messages"][1]["content"])
    assert "current_page" not in sent
    assert sent["snapshot_id"] == snap.snapshot_id
    assert sent["form"]["questions"][0]["question"] == "Which city do you live in?"
    assert sent["form"]["questions"][0]["current_value"] == "Bangalore"
    assert sent["form"]["submit_controls"] == ["e9"]


async def test_current_values_are_looked_up_from_the_live_snapshot():
    """The form came from a cache, so its values are stale; the snapshot is the truth."""
    client, planner = make(plan_json())
    form = NormalisedForm(
        page_kind="form",
        fields=[NormalisedField(id="e1", question="City?", field_type="text", required=True)],
        submit_controls=[],
    )
    await planner.next_step(APPLICANT, snapshot([element("e1", label="City", value="Mumbai")]), [], form=form)
    sent = json.loads(client.calls[0]["messages"][1]["content"])
    assert sent["form"]["questions"][0]["current_value"] == "Mumbai"


async def test_include_all_options_reaches_the_payload():
    client, planner = make(plan_json())
    planner.include_all_options = True
    options = [(f"C{i}", f"Choice {i}") for i in range(400)]
    await planner.next_step(
        APPLICANT,
        snapshot([element("e1", role="combobox", label="Country", input_type="select-one",
                          options=options, value="C200")]),
        [],
    )
    sent = json.loads(client.calls[0]["messages"][1]["content"])
    assert len(sent["current_page"]["elements"][0]["options"]) == 400


# ------------------------------------------------------------------------------------ schema mode


async def test_a_native_schema_prompt_does_not_restate_the_schema():
    client, planner = make(plan_json())
    await planner.next_step(APPLICANT, snapshot([element("e1", label="City")]), [])
    system = client.calls[0]["messages"][0]["content"]
    assert '"title": "ApplicationPlan"' not in system


async def test_json_mode_restates_the_schema_in_the_prompt():
    """A provider that refuses json_schema still has to be told the exact shape."""
    client, planner = make(plan_json())
    client.schema_mode = "json_object"
    await planner.next_step(APPLICANT, snapshot([element("e1", label="City")]), [])
    system = client.calls[0]["messages"][0]["content"]
    assert "no markdown fences" in system
    assert '"title": "ApplicationPlan"' in system


# ------------------------------------------------------------------------------------ malformed


async def test_a_malformed_plan_is_retried_once():
    client, planner = make(["not json", plan_json([action_json()])])
    plan = await planner.next_step(APPLICANT, snapshot([element("e1", label="City")]), [])
    assert len(plan.actions) == 1
    assert len(client.calls) == 2


async def test_a_malformed_plan_is_logged(tmp_path):
    logger = JourneyLogger(str(tmp_path / "j.jsonl"))
    client, planner = make(["nonsense", plan_json()], journey=logger)
    await planner.next_step(APPLICANT, snapshot([element("e1", label="City")]), [])
    logger.close()
    events = [json.loads(line)["event"] for line in (tmp_path / "j.jsonl").read_text().splitlines() if line]
    assert "plan_malformed" in events
    assert "MALFORMED - RETRYING" in (tmp_path / "j.log").read_text()


async def test_a_second_malformed_plan_raises():
    client, planner = make(["nonsense", "still nonsense"])
    with pytest.raises(Exception):
        await planner.next_step(APPLICANT, snapshot([element("e1", label="City")]), [])


async def test_a_transport_failure_is_surfaced_rather_than_swallowed():
    """A structural error is not a parse error; the run must not loop on it."""
    client = FakeModelClient(raises=RuntimeError("connection reset"))
    planner = LLMPlanner(client, "model", ASSETS, AutomationDefaults())
    with pytest.raises(RuntimeError, match="connection reset"):
        await planner.next_step(APPLICANT, snapshot([element("e1", label="City")]), [])
    assert len(client.calls) == 1, "a transport failure must not be retried as a malformed plan"


# -------------------------------------------------------------------------------------- logging


async def test_the_request_and_response_are_logged(tmp_path):
    logger = JourneyLogger(str(tmp_path / "j.jsonl"))
    client, planner = make(plan_json([action_json()]), journey=logger)
    await planner.next_step(APPLICANT, snapshot([element("e1", label="City")]), [])
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
    await planner.next_step(APPLICANT, snapshot([element("e1", label="City")]), [])
    assert "plan_received" not in logger_calls


if __name__ == "__main__":  # pragma: no cover - a convenience, not a test path
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
