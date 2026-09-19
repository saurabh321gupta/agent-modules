"""Tests for `agent_modules.policy` — what the run may answer without asking."""

from __future__ import annotations

from helpers import element, snapshot

from agent_modules.config import AutomationDefaults
from agent_modules.policy import default_consent_actions, is_matching_consent


def test_matching_consent_labels_are_recognised():
    for label in (
        "I consent to the processing of my data",
        "I agree to the processing of my personal data",
        "I have read the recruitment notice",
        "I accept the privacy notice",
    ):
        assert is_matching_consent(element("e1", role="checkbox", label=label)), label


def test_non_consent_checkboxes_are_left_alone():
    for label in (
        "I have a disability",
        "I am a veteran",
        "I agree to the terms of service",
        "Subscribe to job alerts",
    ):
        assert not is_matching_consent(element("e1", role="checkbox", label=label)), label


def test_only_checkboxes_are_consent():
    """A button labelled 'I consent' is not a checkbox and must not be auto-clicked."""
    assert not is_matching_consent(element("e1", role="button", label="I consent", input_type="button"))


def test_actions_are_emitted_for_unchecked_consents():
    snap = snapshot(
        [
            element("e1", role="checkbox", label="I accept the privacy notice", checked=False),
            element("e2", role="checkbox", label="I have a disability", checked=False),
        ]
    )
    actions = default_consent_actions(snap, AutomationDefaults())
    assert len(actions) == 1
    assert actions[0].target == "e1"
    assert actions[0].type == "set_checked"
    assert actions[0].checked is True


def test_already_checked_consents_are_not_repeated():
    snap = snapshot([element("e1", role="checkbox", label="I consent", checked=True)])
    assert default_consent_actions(snap, AutomationDefaults()) == []


def test_disabled_consents_are_skipped():
    snap = snapshot([element("e1", role="checkbox", label="I consent", checked=False, enabled=False)])
    assert default_consent_actions(snap, AutomationDefaults()) == []


def test_opt_out_produces_no_actions():
    snap = snapshot([element("e1", role="checkbox", label="I consent", checked=False)])
    assert default_consent_actions(snap, AutomationDefaults(accept_matching_consents=False)) == []


def test_emitted_action_carries_explicit_nulls():
    """The action travels into a strict-schema payload, so every field must be present."""
    snap = snapshot([element("e1", role="checkbox", label="I consent", checked=False)])
    payload = default_consent_actions(snap, AutomationDefaults())[0].model_dump()
    assert payload == {
        "type": "set_checked",
        "target": "e1",
        "value": None,
        "value_ref": None,
        "option_value": None,
        "checked": True,
        "asset_id": None,
    }


if __name__ == "__main__":  # pragma: no cover - a convenience, not a test path
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
