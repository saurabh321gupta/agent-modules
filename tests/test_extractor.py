"""Tests for `agent_modules.extractor` — which controls a model may be shown."""

from __future__ import annotations

from helpers import EBAY_LIKE_ELEMENTS, element, snapshot

from agent_modules import extractor


def test_links_are_never_capturable():
    """Anchors are navigation. Sending them costs payload and was never once acted on."""
    assert extractor.is_capturable(element("e1", role="link", label="Apply Now", input_type=None)) is False


def test_an_anchor_declared_as_a_button_is_capturable():
    """An apply control styled as a link but declaring role=button is a real control."""
    assert extractor.is_capturable(element("e1", role="button", label="Apply Now", input_type=None)) is True


def test_an_invisible_text_field_is_not_capturable():
    assert extractor.is_capturable(element("e1", label="City", visible=False)) is False


def test_a_hidden_file_input_is_capturable():
    """The real resume input is hidden behind a styled button, as eBay does it.

    Dropping it here is what stopped the required upload from ever happening: the field never reached
    the planner, no upload was ever planned, and the form refused to advance.
    """
    assert extractor.is_capturable(
        element("e9", role="textbox", label="", input_type="file", visible=False)
    ) is True


def test_an_attached_file_input_is_not_actionable():
    """Regression, found on a live run.

    An application wizard keeps its resume widget mounted in the page shell, so the same hidden input
    appears on every step, already attached after the first upload. Offering it invites the model to
    upload the resume again on page four - and because re-uploading does not change the page, the
    identical plan repeats until the run is declared stuck on a page it never tried to leave.
    """
    attached = element("e6", role="textbox", label="", input_type="file",
                       visible=False, file_attached=True)
    assert extractor.is_capturable(attached) is True
    assert extractor.is_actionable(attached) is False


def test_an_empty_hidden_file_input_is_actionable():
    """The page-one case the visibility exemption exists for."""
    assert extractor.is_actionable(
        element("e9", role="textbox", label="", input_type="file", visible=False, file_attached=False)
    ) is True


def test_the_two_predicates_agree_on_everything_else():
    plain = element("e3", label="City")
    invisible = element("e4", label="Hidden", visible=False)
    link = element("e5", role="link", label="Help", input_type=None)
    for item in (plain, invisible, link):
        assert extractor.is_capturable(item) == extractor.is_actionable(item)


def test_the_ebay_shape_survives_the_capture_rules():
    """The recorded page's real controls, checked against the rules that decide what is offered."""
    capturable = [e.id for e in EBAY_LIKE_ELEMENTS if extractor.is_capturable(e)]
    actionable = [e.id for e in EBAY_LIKE_ELEMENTS if extractor.is_actionable(e)]
    assert "e9" in capturable, "the hidden resume input must survive"
    assert capturable == actionable, "nothing on that page is attached yet"


def test_a_snapshot_is_still_serialisable_for_the_trace():
    payload = snapshot([element("e1", label="City")]).model_dump(mode="json")
    assert payload["elements"][0]["label"] == "City"
