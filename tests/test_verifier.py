"""Tests for `agent_modules.verifier` — the only thing allowed to declare success."""

from __future__ import annotations

import pytest
from helpers import snapshot

from agent_modules.verifier import verify_completion

APPLIED_FORM = ["My Information", "Submit", "Country", "India"]


@pytest.mark.parametrize(
    "text",
    [
        "Your application has been submitted.",
        "Application submitted successfully",
        "We have received your application",
        "Thank you for applying!",
        "You have successfully applied for this role",
        "Your application has been received",
    ],
)
def test_confirmation_phrases_are_recognised(text):
    evidence = verify_completion(snapshot([], visible_text=[text]))
    assert evidence is not None


def test_a_filled_form_is_not_a_confirmation():
    """The expensive false positive: a form that merely mentions 'application' must not end the run."""
    assert verify_completion(snapshot([], visible_text=APPLIED_FORM)) is None


def test_clicking_submit_is_not_evidence():
    assert verify_completion(snapshot([], visible_text=["Submit application", "Next", "Back"])) is None


def test_empty_page_is_not_evidence():
    assert verify_completion(snapshot([], visible_text=[])) is None


def test_reference_number_is_extracted():
    evidence = verify_completion(
        snapshot(
            [],
            visible_text=[
                "Thank you for applying",
                "Your reference number: ABC-12345",
            ],
        )
    )
    assert evidence is not None
    assert evidence.reference == "ABC-12345"


def test_evidence_records_the_matched_text_and_url():
    evidence = verify_completion(
        snapshot(
            [],
            visible_text=["Application submitted successfully"],
            url="https://example.com/done",
        )
    )
    assert evidence is not None
    assert evidence.url == "https://example.com/done"
    assert "submitted" in (evidence.text or "").lower()


def test_evidence_without_a_reference_is_still_evidence():
    evidence = verify_completion(snapshot([], visible_text=["Thank you for applying"]))
    assert evidence is not None
    assert evidence.reference is None


def test_phrases_are_matched_case_insensitively():
    assert verify_completion(snapshot([], visible_text=["THANK YOU FOR APPLYING"])) is not None


if __name__ == "__main__":  # pragma: no cover - a convenience, not a test path
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
