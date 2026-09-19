"""Tests for `agent_modules.planner_staged`.

Each branch is exercised on its own, because the point of staging is that most pages never reach an
answering call at all.
"""

from __future__ import annotations

import json

import pytest
from fakes import FakeClassifier, FakeModelClient, classification
from helpers import element, profile, snapshot

from agent_modules.config import AutomationDefaults
from agent_modules.journey import JourneyLogger
from agent_modules.normalizer import Normalizer
from agent_modules.planner_llm import LLMPlanner
from agent_modules.planner_staged import StagedPlanner

APPLICANT = profile()

FORM_REPLY = json.dumps(
    {
        "page_kind": "form",
        "fields": [{"id": "e1", "question": "Which city?", "field_type": "text", "required": True}],
        "submit_controls": [],
    }
)

PLAN_REPLY = json.dumps(
    {
        "snapshot_id": "s-1-abc12345",
        "status": "continue",
        "actions": [
            {
                "type": "fill",
                "target": "e1",
                "value": None,
                "value_ref": "location.city",
                "option_value": None,
                "checked": None,
                "asset_id": None,
            }
        ],
        "reason": "filled the city",
        "completion_evidence": None,
    }
)


def make(answers, *, form_reply=FORM_REPLY, plan_reply=PLAN_REPLY, raises=None, journey=None):
    classifier = FakeClassifier(answers, raises=raises)
    model = FakeModelClient([form_reply, plan_reply])
    normalizer = Normalizer(model, "model", journey=journey)
    planner = LLMPlanner(model, "model", defaults=AutomationDefaults(), journey=journey)
    staged = StagedPlanner(classifier, normalizer, planner, journey=journey)
    return classifier, model, staged


FORM_PAGE = snapshot([element("e1", label="City", required=True)])


# ------------------------------------------------------------------------------- simple branches


@pytest.mark.parametrize("kind", ["captcha", "login"])
async def test_a_wall_is_reported_as_blocked(kind):
    _, model, staged = make(classification(kind))
    plan = await staged.next_step(APPLICANT, FORM_PAGE, [])
    assert plan.status == "blocked"
    assert model.calls == [], "a wall must not cost an answering call"


async def test_a_confirmation_page_with_strong_evidence_completes():
    _, model, staged = make(
        classification("application_submitted", 0.95, submitted_evidence={"type": "noul", "noul": 0.95})
    )
    plan = await staged.next_step(APPLICANT, FORM_PAGE, [])
    assert plan.status == "complete"
    assert plan.completion_evidence is not None
    assert model.calls == []


async def test_a_weak_confirmation_verdict_is_refused():
    """A false 'submitted' ends the run early, so the bar for it is higher than for the rest."""
    _, model, staged = make(
        classification("application_submitted", 0.8, submitted_evidence={"type": "noul", "noul": 0.5})
    )
    plan = await staged.next_step(APPLICANT, FORM_PAGE, [])
    assert plan.status == "needs_input"


async def test_a_confirmation_verdict_without_visible_evidence_is_refused():
    _, model, staged = make(
        classification("application_submitted", 0.9, submitted_evidence={"type": "noul", "noul": 0.1})
    )
    plan = await staged.next_step(APPLICANT, FORM_PAGE, [])
    assert plan.status == "needs_input"


# ---------------------------------------------------------------------------------- click branch


JOB_PAGE = snapshot(
    [
        element("e1", role="button", label="Apply Now", input_type="button"),
        element("e2", role="button", label="Job alerts", input_type="button"),
    ]
)


async def test_a_job_listing_is_advanced_by_a_click_and_no_answering_call():
    _, model, staged = make(
        classification("job_listing", 0.95, next_control={"type": "choice", "choice": "e1", "confidence": 0.9})
    )
    plan = await staged.next_step(APPLICANT, JOB_PAGE, [])
    assert plan.status == "continue"
    assert [a.target for a in plan.actions] == ["e1"]
    assert model.calls == [], "classification alone is enough for an advert"
    assert "No planner call was needed" in plan.reason


async def test_a_review_page_is_advanced_by_a_click():
    _, model, staged = make(
        classification("job_review", 0.9, next_control={"type": "choice", "choice": "e1", "confidence": 0.9})
    )
    plan = await staged.next_step(APPLICANT, JOB_PAGE, [])
    assert [a.target for a in plan.actions] == ["e1"]


async def test_a_job_listing_without_a_usable_control_stops():
    _, _, staged = make(classification("job_listing", 0.95))
    plan = await staged.next_step(APPLICANT, JOB_PAGE, [])
    assert plan.status == "needs_input"


async def test_a_job_listing_control_that_does_not_apply_is_refused():
    """A 'Job alerts' button is not how an application starts."""
    _, model, staged = make(
        classification("job_listing", 0.95, next_control={"type": "choice", "choice": "e2", "confidence": 0.9})
    )
    plan = await staged.next_step(APPLICANT, JOB_PAGE, [])
    assert plan.status == "needs_input"
    assert "does not start an application" in plan.reason


async def test_a_control_id_not_on_the_page_is_discarded():
    _, _, staged = make(
        classification("job_listing", 0.95, next_control={"type": "choice", "choice": "e404", "confidence": 0.9})
    )
    plan = await staged.next_step(APPLICANT, JOB_PAGE, [])
    assert plan.status == "needs_input"


# ------------------------------------------------------------------------------------ form branch


async def test_a_form_page_is_normalised_then_answered():
    _, model, staged = make(classification("application_form", 0.95))
    plan = await staged.next_step(APPLICANT, FORM_PAGE, [])
    assert plan.status == "continue"
    assert plan.actions[0].value_ref == "location.city"
    kinds = [call["kind"] for call in model.calls]
    assert kinds == ["complete", "complete_json"]


async def test_the_answering_call_receives_the_normalised_questions():
    _, model, staged = make(classification("application_form", 0.95))
    await staged.next_step(APPLICANT, FORM_PAGE, [])
    sent = json.loads(model.calls[1]["messages"][1]["content"])
    assert sent["form"]["questions"][0]["question"] == "Which city?"
    assert "current_page" not in sent


async def test_a_low_confidence_verdict_falls_through_to_the_form_path():
    """Below the floor the verdict is 'cannot tell', and answering a form is the safe default."""
    _, model, staged = make(classification("job_listing", 0.1))
    plan = await staged.next_step(APPLICANT, FORM_PAGE, [])
    assert plan.status == "continue"
    assert any(call["kind"] == "complete_json" for call in model.calls)


async def test_an_unknown_page_class_falls_through_to_the_form_path():
    _, model, staged = make({"page_class": {"type": "choice", "choice": "nonsense", "confidence": 0.99}})
    await staged.next_step(APPLICANT, FORM_PAGE, [])
    assert any(call["kind"] == "complete_json" for call in model.calls)


async def test_an_unrecognised_page_stops_when_the_verdict_is_confident():
    _, model, staged = make(classification("other", 0.95))
    plan = await staged.next_step(APPLICANT, FORM_PAGE, [])
    assert plan.status == "needs_input"
    assert model.calls == []


async def test_a_classification_failure_falls_back_to_the_form_path():
    """Without classification the behaviour is the plain planner, which is the pre-staging default."""
    _, model, staged = make(None, raises=RuntimeError("jev is down"))
    plan = await staged.next_step(APPLICANT, FORM_PAGE, [])
    assert plan.status == "continue"
    assert plan.actions[0].value_ref == "location.city"


async def test_a_denied_submission_notice_is_not_treated_as_a_form():
    """A page that merely mentions submission without evidence is not proof of anything."""
    _, _, staged = make(classification("application_form", 0.9))
    plan = await staged.next_step(APPLICANT, FORM_PAGE, [])
    assert plan.status == "continue"


# ------------------------------------------------------------------------------------- reporting


async def test_last_step_reports_the_classification_for_a_form_page():
    """Regression: without this the STEP TIMING line attributed every form step to the first page."""
    _, _, staged = make(classification("application_form", 0.88))
    await staged.next_step(APPLICANT, FORM_PAGE, [])
    assert staged.last_step["class"] == "application_form"
    assert staged.last_step["confidence"] == 0.88
    assert staged.last_step["actions"] == 1


async def test_last_step_reports_the_classification_for_a_click_branch():
    _, _, staged = make(
        classification("job_listing", 0.95, next_control={"type": "choice", "choice": "e1", "confidence": 0.9})
    )
    await staged.next_step(APPLICANT, JOB_PAGE, [])
    assert staged.last_step["class"] == "job_listing"
    assert staged.last_step["actions"] == 1


# --------------------------------------------------------------------------------------- logging


async def test_the_classification_is_logged(tmp_path):
    logger = JourneyLogger(str(tmp_path / "j.jsonl"))
    _, _, staged = make(classification("application_form", 0.9), journey=logger)
    await staged.next_step(APPLICANT, FORM_PAGE, [])
    logger.close()
    events = [json.loads(line)["event"] for line in (tmp_path / "j.jsonl").read_text().splitlines() if line]
    assert "classify_request" in events
    assert "page_classified" in events
    human = (tmp_path / "j.log").read_text(encoding="utf-8")
    assert "Verdict           : application_form (confidence 0.90)" in human


async def test_a_classification_failure_is_logged(tmp_path):
    logger = JourneyLogger(str(tmp_path / "j.jsonl"))
    _, _, staged = make(None, raises=RuntimeError("jev down"), journey=logger)
    await staged.next_step(APPLICANT, FORM_PAGE, [])
    logger.close()
    assert "Classification    : FAILED" in (tmp_path / "j.log").read_text(encoding="utf-8")


async def test_a_refused_submitted_verdict_is_logged(tmp_path):
    logger = JourneyLogger(str(tmp_path / "j.jsonl"))
    _, _, staged = make(
        classification("application_submitted", 0.9, submitted_evidence={"type": "noul", "noul": 0.1}),
        journey=logger,
    )
    await staged.next_step(APPLICANT, FORM_PAGE, [])
    logger.close()
    assert "REJECTED (not certain enough)" in (tmp_path / "j.log").read_text(encoding="utf-8")


if __name__ == "__main__":  # pragma: no cover - a convenience, not a test path
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
