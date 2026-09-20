"""Tests for `agent_modules.prompts_llm` — what the answering model is told and given."""

from __future__ import annotations

import json

from helpers import element, profile, snapshot

from agent_modules.config import AutomationDefaults
from agent_modules.prompts_llm import (
    BATCH_RULE,
    FILLED_OPTION_WINDOW,
    FILLED_RULE,
    MAIN_SYSTEM_PROMPT,
    MAX_OPTIONS_SENT,
    REJECTION_RULE,
    UPLOAD_RULE,
    profile_for_llm,
    response_format_for,
    scrub_local_paths,
    snapshot_for_llm,
    with_schema,
)

ASSETS = {"resume_primary": "/tmp/resume.pdf"}


def flat(prompt: str) -> str:
    """Collapse whitespace so an assertion does not depend on where the prompt happens to wrap."""
    return " ".join(prompt.split())


# ------------------------------------------------------------------------------------------ rules


def test_the_prompt_carries_all_four_rules():
    for rule in (UPLOAD_RULE, BATCH_RULE, FILLED_RULE, REJECTION_RULE):
        assert rule in MAIN_SYSTEM_PROMPT


def test_the_upload_rule_says_an_offered_file_field_needs_a_file():
    assert "asset_id" in UPLOAD_RULE
    assert "never skip an empty one" in UPLOAD_RULE
    assert "not offered to you at all" in UPLOAD_RULE, "an attached field never reaches the model"


def test_the_batch_rule_explains_only_one_page_changing_action_can_land():
    """A batch ends at a click, an upload or a scroll, so pairing an upload with a click means only
    the upload happens and the page never advances - which is exactly what closed a live run."""
    assert "at most one of them can take effect" in BATCH_RULE
    assert "Never pair an upload with a click" in BATCH_RULE


def test_the_filled_rule_names_the_keys_the_page_actually_carries():
    """It used to name `current_value`, which only existed in a payload we no longer send."""
    text = flat(FILLED_RULE)
    assert "value says what a field contains" in text
    assert "checked says whether a checkbox or radio is already set" in text
    assert "current_value" not in text


def test_the_rejection_rule_names_the_keys_the_page_actually_carries():
    """Same trap: `rejected` was a normalised-form key. The page marks a control `invalid`."""
    text = flat(REJECTION_RULE)
    assert '"invalid": true' in text
    assert "validation_errors" in text
    assert '"rejected"' not in text
    assert "without one" in text, "the phone-number case is named explicitly"


def test_the_prompt_does_not_promise_a_history_that_is_not_sent():
    """Anything the prompt refers to has to be in the payload, or it describes a capability the code
    does not have - which has bitten this project repeatedly."""
    assert "recent_history" not in MAIN_SYSTEM_PROMPT


def test_the_prompt_does_not_promise_an_answer_bank_reference_that_nothing_resolves():
    """It used to say "or an answers.* path", and nothing resolves that."""
    assert "answers.*" not in MAIN_SYSTEM_PROMPT


def test_the_prompt_carries_no_unsubstituted_placeholder():
    assert "%s" not in MAIN_SYSTEM_PROMPT


# ------------------------------------------------------------------------------------- schema mode


def test_with_schema_leaves_a_native_schema_prompt_alone():
    assert with_schema("base", "json_schema") == "base"


def test_with_schema_appends_the_schema_in_json_mode():
    rendered = with_schema("base", "json_object")
    assert rendered.startswith("base")
    assert "no markdown fences" in rendered
    payload = json.loads(rendered.split("exactly:\n", 1)[1])
    assert payload["title"] == "ApplicationPlan"


def test_the_plan_schema_no_longer_asks_for_a_snapshot_id():
    properties = response_format_for("json_schema")["json_schema"]["schema"]["properties"]
    assert "snapshot_id" not in properties


def test_response_format_for_marks_the_schema_strict():
    strict = response_format_for("json_schema")
    assert strict["type"] == "json_schema"
    assert strict["json_schema"]["strict"] is True
    assert strict["json_schema"]["name"] == "application_plan"


def test_response_format_for_json_mode_is_bare():
    assert response_format_for("json_object") == {"type": "json_object"}


# --------------------------------------------------------------------------------- profile payload


def test_local_paths_never_reach_the_payload():
    """The documents block is reduced to an asset view, so the path is removed rather than scrubbed."""
    payload = profile_for_llm(profile(), ASSETS, AutomationDefaults())
    assert "path" not in payload["documents"]["resume"]
    assert "/secret/resume.pdf" not in json.dumps(payload)


def test_scrub_handles_nested_lists_and_suffixes():
    scrubbed = scrub_local_paths(
        {"a": [{"file_path": "/x"}, {"other": "/keep"}], "save_filepath": "/y"}
    )
    assert scrubbed["a"][0]["file_path"] == "[local path redacted]"
    assert scrubbed["a"][1]["other"] == "/keep"
    assert scrubbed["save_filepath"] == "[local path redacted]"


def test_documents_are_reduced_to_an_asset_view():
    payload = profile_for_llm(profile(), ASSETS, AutomationDefaults())
    assert payload["documents"]["resume"] == {
        "asset_id": "resume_primary",
        "filename": "resume.pdf",
        "available": True,
    }


def test_an_unregistered_document_is_marked_unavailable():
    payload = profile_for_llm(profile(), {}, AutomationDefaults())
    assert payload["documents"]["resume"]["available"] is False


def test_defaults_are_injected_under_a_clear_key():
    payload = profile_for_llm(profile(), ASSETS, AutomationDefaults(citizenship_country="India"))
    assert payload["automation_defaults"]["citizenship_country"] == "India"
    assert payload["automation_defaults"]["accept_matching_consent_checkboxes"] is True


def test_profile_payload_is_json_serialisable():
    json.dumps(profile_for_llm(profile(), ASSETS, AutomationDefaults()))


# ---------------------------------------------------------------------------------- page payload


def test_the_page_payload_is_the_whole_page():
    """The model sees the page as a person does: text, controls, errors, blockers."""
    snap = snapshot([element("e1", label="City")], visible_text=["Apply for this job"])
    payload = snapshot_for_llm(snap)
    assert set(payload) >= {"url", "title", "visible_text", "elements", "validation_errors"}
    assert payload["elements"][0]["label"] == "City"


def test_the_page_payload_carries_the_sites_own_errors():
    """The rule about rejected values is only obeyable because these travel with the page."""
    payload = snapshot_for_llm(snapshot([], validation_errors=["Email is invalid"]))
    assert payload["validation_errors"] == ["Email is invalid"]


def test_an_element_carries_its_state_and_its_rejection():
    payload = snapshot_for_llm(
        snapshot(
            [
                element("e1", role="checkbox", label="I consent", checked=True),
                element("e2", label="Phone", value="91-8386", invalid=True),
            ]
        )
    )
    first, second = payload["elements"]
    assert first["checked"] is True
    assert second["invalid"] is True
    assert second["value"] == "91-8386"


def test_an_unset_dropdown_keeps_every_option():
    """option_value must be copied exactly, so the planner needs the list it is choosing from."""
    options = [(f"C{i}", f"Choice {i}") for i in range(40)]
    payload = snapshot_for_llm(
        snapshot([element("e1", role="combobox", label="Country", input_type="select-one", options=options)])
    )
    assert len(payload["elements"][0]["options"]) == 40
    assert "options_truncated" not in payload["elements"][0]


def test_an_already_set_dropdown_is_windowed():
    options = [(f"C{i}", f"Choice {i}") for i in range(400)]
    payload = snapshot_for_llm(
        snapshot(
            [
                element(
                    "e1",
                    role="combobox",
                    label="Country",
                    input_type="select-one",
                    options=options,
                    value="C200",
                )
            ]
        )
    )
    kept = payload["elements"][0]["options"]
    assert payload["elements"][0]["options_truncated"] is True
    assert len(kept) == 1 + 2 * FILLED_OPTION_WINDOW
    assert kept[0]["value"] == "C200", "the current selection must always be included"


def test_an_enormous_unset_dropdown_is_capped_but_keeps_head_and_tail():
    options = [(f"C{i}", f"Choice {i}") for i in range(MAX_OPTIONS_SENT + 500)]
    payload = snapshot_for_llm(
        snapshot([element("e1", role="combobox", label="Country", input_type="select-one", options=options)])
    )
    kept = payload["elements"][0]["options"]
    assert payload["elements"][0]["options_truncated"] is True
    assert len(kept) == 999
    assert kept[0]["value"] == "C0"
    assert kept[-1]["value"] == f"C{MAX_OPTIONS_SENT + 499}"


def test_visible_text_is_capped_for_the_payload():
    payload = snapshot_for_llm(snapshot([], visible_text=[f"line {i}" for i in range(400)]))
    assert len(payload["visible_text"]) == 180


def test_snapshot_payload_is_json_serialisable():
    json.dumps(snapshot_for_llm(snapshot([element("e1", label="City")])))


if __name__ == "__main__":  # pragma: no cover - a convenience, not a test path
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
