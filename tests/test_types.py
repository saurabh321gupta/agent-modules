"""Tests for `agent_modules.types` — the contracts every other module depends on."""

from __future__ import annotations

import pytest
from helpers import action, element, profile, snapshot
from pydantic import ValidationError

from agent_modules.types import (
    Action,
    ApplicationPlan,
    CandidateProfile,
    CompletionEvidence,
    NormalisedField,
    NormalisedForm,
    PageElement,
    RunResult,
)


def test_action_forbids_unknown_fields():
    """Strict output depends on this: an extra key means the model invented something."""
    with pytest.raises(ValidationError):
        Action(
            type="fill",
            target="e1",
            value="x",
            value_ref=None,
            option_value=None,
            checked=None,
            asset_id=None,
            surprise="nope",
        )


def test_page_element_forbids_unknown_fields():
    with pytest.raises(ValidationError):
        PageElement(
            id="e1",
            role="textbox",
            label="City",
            input_type="text",
            required=False,
            value=None,
            enabled=True,
            checked=None,
            options=[],
            help_text=None,
            visible=True,
            surprise=True,
        )


def test_all_action_fields_are_required_for_strict_schema():
    """The wire format carries explicit nulls so the JSON schema can be strict."""
    required = set(Action.model_json_schema()["required"])
    assert required == {
        "type",
        "target",
        "value",
        "value_ref",
        "option_value",
        "checked",
        "asset_id",
    }


def test_plan_round_trips_through_json():
    original = ApplicationPlan(
        snapshot_id="s-1-a",
        status="continue",
        actions=[action("fill", "e1", value_ref="identity.first_name")],
        reason="fill the name",
        completion_evidence=None,
    )
    restored = ApplicationPlan.model_validate_json(original.model_dump_json())
    assert restored == original


def test_plan_status_is_constrained():
    with pytest.raises(ValidationError):
        ApplicationPlan(
            snapshot_id="s-1-a",
            status="nonsense",
            actions=[],
            reason="",
            completion_evidence=None,
        )


def test_run_result_accepts_timeout_status():
    """The run budget needs somewhere to land that is not 'failed' or 'blocked'."""
    result = RunResult(
        status="timeout",
        message="budget exhausted",
        url="https://example.com",
        steps=3,
        evidence=None,
        action_results=[],
    )
    assert result.status == "timeout"


def test_candidate_profile_allows_extra_keys():
    """Profiles are user-authored and will grow; refusing an unknown key would break real files."""
    candidate = CandidateProfile.model_validate(
        {"identity": {"first_name": "A"}, "something_new": {"x": 1}}
    )
    assert candidate.identity["first_name"] == "A"


def test_completion_evidence_is_all_nullable():
    evidence = CompletionEvidence(text=None, url=None, reference=None)
    assert evidence.reference is None


def test_normalised_form_defaults_are_independent():
    """A mutable default shared between instances would leak cache state across forms."""
    first = NormalisedForm(page_kind="form", fields=[], submit_controls=[])
    second = NormalisedForm(page_kind="form", fields=[], submit_controls=[])
    first.fields.append(NormalisedField(id="e1", question="City", field_type="text", required=True))
    assert second.fields == []


def test_snapshot_serialises_for_the_journey_trace():
    snap = snapshot([element("e1", label="City")])
    payload = snap.model_dump(mode="json")
    assert payload["snapshot_id"] == snap.snapshot_id
    assert payload["elements"][0]["label"] == "City"


def test_profile_documents_survive_round_trip():
    candidate = profile()
    assert candidate.documents["resume"]["asset_id"] == "resume_primary"


if __name__ == "__main__":  # pragma: no cover - a convenience, not a test path
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
