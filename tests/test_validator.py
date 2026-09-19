"""Tests for `agent_modules.validator` — the gate every plan passes through before the browser."""

from __future__ import annotations

import pytest
from helpers import action, element, plan, profile, snapshot

from agent_modules.validator import (
    PlanValidationError,
    looks_like_submit,
    resolve_profile_ref,
    unresolved_required,
    validate_plan,
)

ASSETS = {"resume_primary": "/tmp/resume.pdf"}


def form_snapshot(**overrides):
    elements = overrides.pop(
        "elements",
        [
            element("e1", role="textbox", label="City", required=True),
            element(
                "e2",
                role="combobox",
                label="Country",
                input_type="select-one",
                options=[("IND", "India"), ("USA", "United States")],
            ),
            element("e3", role="button", label="Next", input_type="submit"),
            element("e4", role="textbox", label="", input_type="file", visible=False),
            element("e5", role="checkbox", label="I agree to the privacy notice", checked=False),
            element("e6", role="textbox", label="Locked", enabled=False),
        ],
    )
    return snapshot(elements, **overrides)


# ------------------------------------------------------------------------------- profile refs


def test_resolve_profile_ref_walks_dotted_paths(sample_profile):
    assert resolve_profile_ref(sample_profile, "identity.first_name") == "Alex"
    assert resolve_profile_ref(sample_profile, "location.city") == "Bangalore"


def test_resolve_profile_ref_rejects_unknown_path(sample_profile):
    with pytest.raises(PlanValidationError, match="Unknown candidate profile reference"):
        resolve_profile_ref(sample_profile, "identity.middle_name")


def test_resolve_profile_ref_rejects_unknown_value(sample_profile):
    """A missing fact is unknown, never empty and never no."""
    with pytest.raises(PlanValidationError, match="value is unknown"):
        resolve_profile_ref(sample_profile, "preferences.expected_salary")


def test_resolve_profile_ref_rejects_containers(sample_profile):
    with pytest.raises(PlanValidationError, match="must resolve to a scalar"):
        resolve_profile_ref(sample_profile, "identity")


# ------------------------------------------------------------------------------- batch limits


def test_expired_snapshot_is_rejected(sample_profile):
    with pytest.raises(PlanValidationError, match="expired snapshot"):
        validate_plan(plan("s-OLD", []), form_snapshot(), sample_profile, ASSETS, True)


def test_batch_over_limit_is_rejected(sample_profile):
    actions = [action("scroll", "e3") for _ in range(13)]
    with pytest.raises(PlanValidationError, match="maximum of 12 actions"):
        validate_plan(plan("s-1-abc12345", actions), form_snapshot(), sample_profile, ASSETS, True)


def test_twelve_actions_is_allowed(sample_profile):
    actions = [action("scroll", "e3") for _ in range(12)]
    validate_plan(plan("s-1-abc12345", actions), form_snapshot(), sample_profile, ASSETS, True)


# ------------------------------------------------------------------------------- target rules


def test_unknown_target_is_rejected(sample_profile):
    with pytest.raises(PlanValidationError, match="not in snapshot"):
        validate_plan(
            plan("s-1-abc12345", [action("fill", "e99", value="x")]),
            form_snapshot(),
            sample_profile,
            ASSETS,
            True,
        )


def test_missing_target_is_rejected(sample_profile):
    with pytest.raises(PlanValidationError, match="requires a target"):
        validate_plan(
            plan("s-1-abc12345", [action("fill", None, value="x")]),
            form_snapshot(),
            sample_profile,
            ASSETS,
            True,
        )


def test_disabled_target_is_rejected(sample_profile):
    """The message is parsed by the orchestrator to reduce the batch, so it must name the target."""
    with pytest.raises(PlanValidationError, match="Target e6 is disabled"):
        validate_plan(
            plan("s-1-abc12345", [action("fill", "e6", value="x")]),
            form_snapshot(),
            sample_profile,
            ASSETS,
            True,
        )


# ----------------------------------------------------------------------------------- fill rules


def test_fill_on_file_input_is_rejected(sample_profile):
    with pytest.raises(PlanValidationError, match="is not a text field"):
        validate_plan(
            plan("s-1-abc12345", [action("fill", "e4", value="/tmp/x.pdf")]),
            form_snapshot(),
            sample_profile,
            ASSETS,
            True,
        )


def test_fill_with_unknown_value_ref_is_rejected(sample_profile):
    with pytest.raises(PlanValidationError, match="Unknown candidate profile reference"):
        validate_plan(
            plan("s-1-abc12345", [action("fill", "e1", value_ref="identity.nope")]),
            form_snapshot(),
            sample_profile,
            ASSETS,
            True,
        )


def test_fill_with_neither_value_nor_ref_is_rejected(sample_profile):
    with pytest.raises(PlanValidationError, match="requires value or value_ref"):
        validate_plan(
            plan("s-1-abc12345", [action("fill", "e1")]),
            form_snapshot(),
            sample_profile,
            ASSETS,
            True,
        )


# --------------------------------------------------------------------------------- select rules


def test_select_requires_an_option_value(sample_profile):
    with pytest.raises(PlanValidationError, match="requires option_value"):
        validate_plan(
            plan("s-1-abc12345", [action("select", "e2")]),
            form_snapshot(),
            sample_profile,
            ASSETS,
            True,
        )


def test_select_with_absent_option_is_rejected(sample_profile):
    """A model that invents an option must not reach the page."""
    with pytest.raises(PlanValidationError, match="is unavailable for e2"):
        validate_plan(
            plan("s-1-abc12345", [action("select", "e2", option_value="DEU")]),
            form_snapshot(),
            sample_profile,
            ASSETS,
            True,
        )


def test_select_with_real_option_is_allowed(sample_profile):
    validate_plan(
        plan("s-1-abc12345", [action("select", "e2", option_value="IND")]),
        form_snapshot(),
        sample_profile,
        ASSETS,
        True,
    )


def test_select_on_a_textbox_is_rejected(sample_profile):
    with pytest.raises(PlanValidationError, match="is not a dropdown"):
        validate_plan(
            plan("s-1-abc12345", [action("select", "e1", option_value="x")]),
            form_snapshot(),
            sample_profile,
            ASSETS,
            True,
        )


# -------------------------------------------------------------------------------- check rules


def test_set_checked_on_textbox_is_rejected(sample_profile):
    with pytest.raises(PlanValidationError, match="not a checkbox or radio"):
        validate_plan(
            plan("s-1-abc12345", [action("set_checked", "e1", checked=True)]),
            form_snapshot(),
            sample_profile,
            ASSETS,
            True,
        )


def test_set_checked_without_state_is_rejected(sample_profile):
    with pytest.raises(PlanValidationError, match="requires checked"):
        validate_plan(
            plan("s-1-abc12345", [action("set_checked", "e5")]),
            form_snapshot(),
            sample_profile,
            ASSETS,
            True,
        )


# --------------------------------------------------------------------------------- upload rules


def test_upload_to_a_file_input_is_allowed(sample_profile):
    validate_plan(
        plan("s-1-abc12345", [action("upload", "e4", asset_id="resume_primary")]),
        form_snapshot(),
        sample_profile,
        ASSETS,
        True,
    )


def test_upload_to_a_non_file_input_is_rejected(sample_profile):
    with pytest.raises(PlanValidationError, match="is not a file input"):
        validate_plan(
            plan("s-1-abc12345", [action("upload", "e1", asset_id="resume_primary")]),
            form_snapshot(),
            sample_profile,
            ASSETS,
            True,
        )


def test_upload_of_unregistered_asset_is_rejected(sample_profile):
    with pytest.raises(PlanValidationError, match="Unregistered upload asset"):
        validate_plan(
            plan("s-1-abc12345", [action("upload", "e4", asset_id="cover_letter")]),
            form_snapshot(),
            sample_profile,
            ASSETS,
            True,
        )


def test_upload_of_asset_not_in_profile_is_rejected(sample_profile):
    """An asset the run registered but the profile never mentions is not the candidate's document."""
    with pytest.raises(PlanValidationError, match="not referenced by the candidate profile"):
        validate_plan(
            plan("s-1-abc12345", [action("upload", "e4", asset_id="stray")]),
            form_snapshot(),
            sample_profile,
            {"resume_primary": "/tmp/r.pdf", "stray": "/tmp/s.pdf"},
            True,
        )


# ------------------------------------------------------------------------------ submission gate


def test_submit_is_refused_when_submission_is_off(sample_profile):
    with pytest.raises(PlanValidationError, match="submission is disabled"):
        validate_plan(
            plan("s-1-abc12345", [action("click", "e3")]),
            form_snapshot(elements=[element("e3", role="button", label="Submit Application")]),
            sample_profile,
            ASSETS,
            False,
        )


def test_submit_is_refused_while_validation_errors_remain(sample_profile):
    with pytest.raises(PlanValidationError, match="validation errors remain"):
        validate_plan(
            plan("s-1-abc12345", [action("click", "e3")]),
            form_snapshot(
                elements=[element("e3", role="button", label="Submit")],
                validation_errors=["Email is invalid"],
            ),
            sample_profile,
            ASSETS,
            True,
        )


def test_submit_is_refused_while_a_required_field_is_unresolved(sample_profile):
    with pytest.raises(PlanValidationError, match="Required fields are unresolved: City"):
        validate_plan(
            plan("s-1-abc12345", [action("click", "e3")]),
            form_snapshot(
                elements=[
                    element("e1", role="textbox", label="City", required=True),
                    element("e3", role="button", label="Submit Application"),
                ]
            ),
            sample_profile,
            ASSETS,
            True,
        )


def test_submit_is_refused_while_a_required_file_is_unattached(sample_profile):
    with pytest.raises(PlanValidationError, match="Required fields are unresolved"):
        validate_plan(
            plan("s-1-abc12345", [action("click", "e3")]),
            form_snapshot(
                elements=[
                    element("e4", role="textbox", label="Resume", input_type="file",
                            required=True, visible=False, file_attached=False),
                    element("e3", role="button", label="Submit Application"),
                ]
            ),
            sample_profile,
            ASSETS,
            True,
        )


def test_submit_is_allowed_once_everything_is_resolved(sample_profile):
    validate_plan(
        plan("s-1-abc12345", [action("click", "e3")]),
        form_snapshot(
            elements=[
                element("e1", role="textbox", label="City", required=True, value="Bangalore"),
                element("e3", role="button", label="Submit Application"),
            ]
        ),
        sample_profile,
        ASSETS,
        True,
    )


def test_ordinary_next_button_is_never_gated(sample_profile):
    """Only a real submission control is gated; Next must stay clickable."""
    validate_plan(
        plan("s-1-abc12345", [action("click", "e3")]),
        form_snapshot(),
        sample_profile,
        ASSETS,
        True,
    )


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("Submit Application", True),
        ("submit", True),
        ("Final Submit", True),
        ("Complete and submit", True),
        ("Next", False),
        ("Save and continue", False),
        ("Apply Now", False),
        ("", False),
    ],
)
def test_submit_label_detection(label, expected):
    assert looks_like_submit(label) is expected


def test_unresolved_required_names_fields_readably():
    snap = form_snapshot(
        elements=[
            element("e1", role="textbox", label="City", required=True),
            element("e2", role="checkbox", label="I agree", required=True, checked=False),
            element("e3", role="textbox", label="Filled", required=True, value="ok"),
        ]
    )
    assert unresolved_required(snap) == ["City", "I agree"]


def test_scroll_without_target_is_allowed(sample_profile):
    validate_plan(
        plan("s-1-abc12345", [action("scroll", None)]), form_snapshot(), sample_profile, ASSETS, True
    )


if __name__ == "__main__":  # pragma: no cover - a convenience, not a test path
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
