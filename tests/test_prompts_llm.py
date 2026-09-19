"""Tests for `agent_modules.prompts_llm` — what the main model is told and given."""

from __future__ import annotations

import json

from helpers import element, profile, snapshot

from agent_modules.config import AutomationDefaults
from agent_modules.prompts_llm import (
    FILLED_OPTION_WINDOW,
    FORM_SYSTEM_PROMPT,
    MAIN_SYSTEM_PROMPT,
    MAX_OPTIONS_SENT,
    UPLOAD_RULE,
    profile_for_llm,
    response_format_for,
    scrub_local_paths,
    snapshot_for_llm,
    with_schema,
)

ASSETS = {"resume_primary": "/tmp/resume.pdf"}


# ------------------------------------------------------------------------------ upload instruction


def test_both_prompts_state_the_upload_rule():
    """Regression: the model was told to skip buttons, and saw only an 'Upload Resume' button."""
    assert UPLOAD_RULE in MAIN_SYSTEM_PROMPT
    assert UPLOAD_RULE in FORM_SYSTEM_PROMPT
    assert "upload action" in UPLOAD_RULE
    assert "asset_id" in UPLOAD_RULE


def test_form_prompt_keeps_its_other_rules():
    for fragment in (
        "snapshot_id",
        "up to 12 actions",
        "field_type button or unknown",
        "return needs_input naming it",
        "Click a submit control only once",
        "visible evidence that the application was received",
    ):
        assert fragment in FORM_SYSTEM_PROMPT, fragment


def test_form_prompt_has_no_unsubstituted_placeholder():
    assert "%s" not in FORM_SYSTEM_PROMPT


# ------------------------------------------------------------------------------------- schema mode


def test_with_schema_leaves_a_native_schema_prompt_alone():
    assert with_schema("base", "json_schema") == "base"


def test_with_schema_appends_the_schema_in_json_mode():
    rendered = with_schema("base", "json_object")
    assert rendered.startswith("base")
    assert "no markdown fences" in rendered
    payload = json.loads(rendered.split("exactly:\n", 1)[1])
    assert payload["title"] == "ApplicationPlan"


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
    payload = profile_for_llm(profile(), ASSETS, AutomationDefaults())
    json.dumps(payload)


# ---------------------------------------------------------------------------------- page payload


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


def test_include_all_options_is_the_escape_hatch():
    """Set after a plan is rejected for choosing an option that was not in the window we sent."""
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
        ),
        include_all_options=True,
    )
    assert len(payload["elements"][0]["options"]) == 400
    assert "options_truncated" not in payload["elements"][0]


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
