"""Tests for `agent_modules.prompts_jev` — what Jev is asked, and how."""

from __future__ import annotations

from helpers import element, profile, snapshot

from agent_modules.config import MAX_CHOICES
from agent_modules.prompts_jev import (
    CLASSIFY_PROMPT,
    LARGE_OPTION_SYSTEM,
    REPAIR_SYSTEM,
    SHORTLIST,
    build_field_questions,
    classify_controls,
    classify_questions,
    classify_state,
    option_criteria,
    profile_facts,
    shortlist,
    similarity,
)

BANK = {
    "Current CTC (fixed + variable + other components)": "6000000",
    "Expected CTC": "8000000",
    "Current notice period": "30 days",
    "How did you hear about this job?": "social media",
}

APPLICANT = profile()


# ------------------------------------------------------------------------------- classification


def test_classification_offers_every_page_class():
    questions = classify_questions([])
    assert set(questions["page_class"]["criteria"]) == {
        "job_listing",
        "job_review",
        "application_submitted",
        "application_form",
        "captcha",
        "login",
        "other",
    }


def test_classification_offers_submitted_evidence_but_no_control_when_there_are_no_buttons():
    questions = classify_questions([])
    assert questions["submitted_evidence"]["type"] == "noul"
    assert "next_control" not in questions


def test_classification_lists_only_enabled_buttons_as_controls():
    questions = classify_questions(
        [{"id": "e1", "label": "Apply Now"}, {"id": "e2", "label": "Job alerts"}]
    )
    assert set(questions["next_control"]["criteria"]) == {"e1", "e2"}
    assert "never choose" in questions["next_control"]["instructions"].lower()


def test_classify_controls_narrows_to_the_relevant_roles():
    controls = classify_controls(
        snapshot(
            [
                element("e1", role="button", label="Apply", input_type="button"),
                element("e2", role="combobox", label="Country", input_type="select-one"),
                element("e3", role="textbox", label="City"),
                element("e4", role="link", label="Help", input_type=None),
                element("e5", role="checkbox", label="Alert me", input_type="checkbox"),
            ]
        )
    )
    assert [c["id"] for c in controls] == ["e1", "e2", "e3"]


def test_classify_state_stays_small():
    """Classification runs on every page, so it must not carry the payload an answering call does."""
    big = snapshot([], visible_text=[f"line {i}" for i in range(200)])
    state = classify_state(big, [])
    assert len(state["visible_text"]) == 20
    assert set(state) == {"url", "title", "visible_text", "controls", "validation_errors"}


def test_classify_prompt_lists_every_class_name():
    for name in ("job_listing", "job_review", "application_submitted", "application_form", "captcha", "login"):
        assert name in CLASSIFY_PROMPT


# ------------------------------------------------------------------------------------ fuzzy match


def test_similarity_ignores_stopwords_and_case():
    assert similarity("Current CTC", "current ctc") == 1.0
    assert similarity("Expected CTC", "Current notice period") < 0.2


def test_similarity_of_unrelated_text_is_zero():
    assert similarity("", "anything") == 0.0
    assert similarity("the and for", "anything") == 0.0


def test_shortlist_returns_the_most_similar_entries():
    """Candidates are ranked by how well their question text matches the field's label."""
    candidates = {
        "answers.Current CTC (fixed + variable + other components)": "Current CTC (fixed + variable + other components)",
        "answers.Expected CTC": "Expected CTC",
        "answers.Current notice period": "Current notice period",
    }
    chosen = shortlist("Expected CTC", candidates, limit=1)
    assert list(chosen) == ["answers.Expected CTC"]


def test_shortlist_is_bounded():
    wide = {f"question {i}": f"answer {i}" for i in range(50)}
    assert len(shortlist("question 1", wide, limit=SHORTLIST)) == SHORTLIST


# --------------------------------------------------------------------------------- profile facts


def test_profile_facts_flatten_to_dotted_paths():
    facts = profile_facts(APPLICANT)
    assert facts["identity.first_name"] == "Alex"
    assert facts["location.city"] == "Bangalore"
    assert facts["documents.resume.asset_id"] == "resume_primary"


def test_profile_facts_skip_empty_values():
    """A null fact is unknown. Offering it as a Choice criterion would invite the model to pick it."""
    facts = profile_facts(APPLICANT)
    assert "preferences.expected_salary" not in facts


# ------------------------------------------------------------------------------------ criteria


def test_option_criteria_drops_the_empty_placeholder():
    """'Please Select' is not an answer, so it must never be offered as one."""
    criteria = option_criteria(
        element(
            "e1",
            role="combobox",
            label="Country",
            input_type="select-one",
            options=[("", "Please Select"), ("IND", "India"), ("USA", "United States")],
        )
    )
    assert "" not in criteria
    assert criteria["IND"] == "India"
    assert criteria["none"] == "No option fits, or this field should be left alone"


def test_option_criteria_is_capped_at_the_choice_limit():
    options = [(f"C{i}", f"Choice {i}") for i in range(MAX_CHOICES + 100)]
    criteria = option_criteria(
        element("e1", role="combobox", label="Country", input_type="select-one", options=options)
    )
    assert len(criteria) == MAX_CHOICES + 1  # options plus the "none" sentinel


# ------------------------------------------------------------------------------------- questions


def form_snapshot():
    return snapshot(
        [
            element("e1", role="textbox", label="City", required=True),
            element(
                "e2",
                role="combobox",
                label="Country",
                input_type="select-one",
                options=[("IND", "India"), ("USA", "United States")],
            ),
            element("e3", role="checkbox", label="I consent", checked=False),
            element("e4", role="checkbox", label="Already ticked", checked=True),
            element("e5", role="button", label="Next", input_type="submit"),
            element("e6", role="textbox", label="", input_type="file", visible=False),
            element("e7", role="textbox", label="Locked", enabled=False),
            element("e8", role="textbox", label="Phone", invalid=True),
        ]
    )


def test_every_answerable_control_becomes_a_question():
    questions, mapping, _, _ = build_field_questions(APPLICANT, form_snapshot(), BANK)
    assert set(mapping) == {"state_e1", "state_e2", "state_e3", "advance"}
    assert mapping["state_e1"] == "e1"
    assert mapping["advance"] == "advance"


def test_page_kind_and_submit_ready_are_always_asked():
    questions, _, _, _ = build_field_questions(APPLICANT, form_snapshot(), BANK)
    assert questions["page_kind"]["type"] == "choice"
    assert questions["submit_ready"]["type"] == "noul"


def test_file_password_disabled_and_invisible_controls_are_never_asked():
    """Jev answers 'which fact?', which a file upload is not - and it must not see passwords at all."""
    questions, mapping, _, _ = build_field_questions(APPLICANT, form_snapshot(), BANK)
    asked = set(mapping)
    assert "state_e6" not in asked  # file input
    assert "state_e7" not in asked  # disabled
    assert "state_e8" not in asked  # invalid, diverted below


def test_a_rejected_field_is_diverted_and_not_re_asked():
    """Re-asking which fact to use cannot fix a format the site already rejected."""
    _, _, _, invalid = build_field_questions(APPLICANT, form_snapshot(), BANK)
    assert [e.id for e in invalid] == ["e8"]


def test_an_already_checked_box_is_not_re_asked():
    _, mapping, _, _ = build_field_questions(APPLICANT, form_snapshot(), BANK)
    assert "state_e4" not in mapping


def test_oversized_dropdowns_are_diverted_to_the_fallback():
    """A Choice cannot carry more than 255 criteria, so the responder handles it instead."""
    options = [(f"C{i}", f"Choice {i}") for i in range(MAX_CHOICES + 1)]
    snap = snapshot(
        [element("e1", role="combobox", label="Major", input_type="select-one", options=options)]
    )
    questions, mapping, large, _ = build_field_questions(APPLICANT, snap, BANK)
    assert [e.id for e in large] == ["e1"]
    assert "state_e1" not in mapping
    assert questions["page_kind"] is not None


def test_a_textbox_offers_every_profile_fact_and_a_shortlist_of_answers():
    questions, _, _, _ = build_field_questions(APPLICANT, form_snapshot(), BANK)
    criteria = questions["state_e1"]["criteria"]
    assert "identity.first_name" in criteria
    assert any(key.startswith("answers.") for key in criteria)
    assert criteria["none"] == "No candidate fact answers this field"
    assert len(criteria) <= len(profile_facts(APPLICANT)) + SHORTLIST + 1


def test_advance_offers_every_button():
    questions, _, _, _ = build_field_questions(APPLICANT, form_snapshot(), BANK)
    assert set(questions["advance"]["criteria"]) == {"e5"}


def test_no_advance_question_when_there_are_no_buttons():
    snap = snapshot([element("e1", role="textbox", label="City")])
    questions, mapping, _, _ = build_field_questions(APPLICANT, snap, BANK)
    assert "advance" not in questions
    assert "advance" not in mapping


def test_fallback_prompts_insist_on_a_value_from_the_list():
    assert "Never invent a value" in LARGE_OPTION_SYSTEM
    assert "option_value must be copied exactly" in LARGE_OPTION_SYSTEM
    assert "staying faithful to the candidate's real data" in REPAIR_SYSTEM


if __name__ == "__main__":  # pragma: no cover - a convenience, not a test path
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
