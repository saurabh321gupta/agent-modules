"""Tests for `agent_modules.reader`.

These run against a real headless Chromium loading `tests/fixtures/form.html`, so the injected
JavaScript is genuinely exercised. No network is involved.
"""

from __future__ import annotations

import pytest
from playwright.async_api import Page

from agent_modules.reader import (
    PageReader,
    find_blockers,
    parse_accessible_name,
    snapshot_fingerprint,
)


async def read(page: Page):
    return await PageReader().capture(page)


def by_id(snapshot, element_id):
    found = next((e for e in snapshot.elements if e.id == element_id), None)
    assert found is not None, f"no element {element_id}; got {[e.id for e in snapshot.elements]}"
    return found


def by_label(snapshot, label):
    found = next((e for e in snapshot.elements if e.label == label), None)
    assert found is not None, (
        f"no element labelled {label!r}; labels were {[e.label for e in snapshot.elements]}"
    )
    return found


# ---------------------------------------------------------------------------------- basic capture


async def test_snapshot_records_page_context(loaded_page):
    snapshot = await read(loaded_page)
    assert snapshot.title == "Application form"
    assert snapshot.snapshot_id.startswith("s-1-")
    assert snapshot.visible_text


async def test_controls_are_tagged_with_a_snapshot_and_id(loaded_page):
    snapshot = await read(loaded_page)
    count = await loaded_page.evaluate(
        "document.querySelectorAll('[data-agent-snapshot][data-agent-id]').length"
    )
    assert count >= len(snapshot.elements)
    first = await loaded_page.get_attribute(
        f'[data-agent-snapshot="{snapshot.snapshot_id}"][data-agent-id="e1"]', "data-agent-id"
    )
    assert first == "e1"


async def test_ids_are_positional_and_start_at_one(loaded_page):
    snapshot = await read(loaded_page)
    assert [e.id for e in snapshot.elements] == [f"e{i}" for i in range(1, len(snapshot.elements) + 1)]


# ------------------------------------------------------------------------------------ visibility


async def test_visible_controls_are_marked_visible(loaded_page):
    snapshot = await read(loaded_page)
    assert by_label(snapshot, "City").visible is True


async def test_a_hidden_file_input_is_still_captured(loaded_page):
    """The real resume control is display:none behind a styled button, as eBay does it."""
    snapshot = await read(loaded_page)
    resume = next(e for e in snapshot.elements if e.input_type == "file")
    assert resume.visible is False
    assert resume.file_attached is False


async def test_disabled_controls_are_reported_as_disabled(loaded_page):
    snapshot = await read(loaded_page)
    assert by_label(snapshot, "Next").enabled is False


# ------------------------------------------------------------------------------------- labelling


async def test_label_from_a_for_attribute(loaded_page):
    snapshot = await read(loaded_page)
    assert by_label(snapshot, "City").label_source in {"aria", "heuristic"}


async def test_aria_labelledby_is_resolved_by_the_accessibility_tree(loaded_page):
    """The heuristic chain cannot see aria-labelledby; the accessibility tree can."""
    snapshot = await read(loaded_page)
    phone = next(e for e in snapshot.elements if e.input_type == "tel")
    assert phone.label == "Phone Number"
    assert phone.label_source == "aria"


async def test_a_neighbouring_div_label_is_not_resolved(loaded_page):
    """A known limitation of the accessible-name rules.

    The heuristic chain is aria-label -> label[for] -> wrapping label -> placeholder -> name ->
    innerText. A label sitting in a sibling div with no `for` is invisible to every one of those, so
    the control comes back named after its `name` attribute. A model reading the surrounding DOM can
    still work out the question; a longer if-else chain cannot, which is the whole argument for
    normalising rather than extending these rules.
    """
    snapshot = await read(loaded_page)
    postal = next((e for e in snapshot.elements if (e.label or "").lower() == "postal"), None)
    assert postal is not None, "the field should fall back to its name attribute"
    assert postal.label != "Postal Code"
    assert postal.label_source == "heuristic"


async def test_icon_glyphs_are_stripped_from_labels(loaded_page):
    snapshot = await read(loaded_page)
    glyph = next(e for e in snapshot.elements if "Given Name" in (e.label or ""))
    assert glyph.label == "Given Name(s)"


def test_parse_accessible_name_strips_private_use_glyphs():
    assert parse_accessible_name('- textbox "\ue001 Given Name(s)"') == "Given Name(s)"


def test_parse_accessible_name_handles_escaped_quotes():
    assert parse_accessible_name('- textbox "Say \\"hi\\""') == 'Say "hi"'


def test_parse_accessible_name_returns_empty_without_a_name():
    assert parse_accessible_name("- textbox") == ""
    assert parse_accessible_name("") == ""


# -------------------------------------------------------------------------------------- grouping


async def test_radio_group_question_comes_from_the_legend(loaded_page):
    """Each radio says only 'Yes' or 'No'; the question lives in the legend."""
    snapshot = await read(loaded_page)
    radios = [e for e in snapshot.elements if e.role == "radio"]
    assert radios, "the fixture has a radio group"
    assert all(
        e.group == "Are you legally authorised to work in India?" for e in radios
    )


async def test_group_is_not_captured_for_ordinary_controls(loaded_page):
    """Portals wrap plain sections in a fieldset whose legend is a framework id, which would be
    misread as the question the applicant is answering."""
    snapshot = await read(loaded_page)
    assert by_label(snapshot, "City").group is None


# ----------------------------------------------------------------------------------- form state


async def test_required_is_read_from_the_attribute(loaded_page):
    snapshot = await read(loaded_page)
    assert by_label(snapshot, "City").required is True


async def test_required_is_read_from_aria_required(loaded_page):
    snapshot = await read(loaded_page)
    assert by_label(snapshot, "Email Address").required is True


async def test_optional_controls_are_not_required(loaded_page):
    snapshot = await read(loaded_page)
    assert by_label(snapshot, "Additional information").required is False


async def test_select_options_are_captured_with_values_and_labels(loaded_page):
    snapshot = await read(loaded_page)
    country = by_label(snapshot, "Country")
    assert country.role == "combobox"
    assert [(o.value, o.label) for o in country.options] == [
        ("", "Please Select"),
        ("IND", "India"),
        ("USA", "United States"),
        ("GBR", "United Kingdom"),
    ]


async def test_current_values_are_read_back(loaded_page):
    await loaded_page.fill("#city", "Bangalore")
    snapshot = await read(loaded_page)
    assert by_label(snapshot, "City").value == "Bangalore"


async def test_password_values_are_never_read(loaded_page):
    await loaded_page.set_content('<input type="password" aria-label="Password" value="hunter2" />')
    snapshot = await read(loaded_page)
    assert by_label(snapshot, "Password").value is None


# -------------------------------------------------------------------------------------- anchors


async def test_plain_anchors_are_not_captured(loaded_page):
    """They are navigation, they dominate payloads, and no run has ever acted on one."""
    snapshot = await read(loaded_page)
    assert not [e for e in snapshot.elements if e.role == "link"]


async def test_an_anchor_declared_a_button_is_captured(loaded_page):
    snapshot = await read(loaded_page)
    assert by_label(snapshot, "Apply Now").role == "button"


# ------------------------------------------------------------------------------------- blockers


async def test_captcha_is_reported_as_a_blocker():
    assert "CAPTCHA or human verification is present" in find_blockers("Please verify you are human")


async def test_login_wall_is_reported_as_a_blocker():
    assert find_blockers("Sign in to apply") == ["Authentication or account creation is required"]


async def test_email_verification_is_reported_as_a_blocker():
    assert find_blockers("We need to verify your email address") == [
        "Email verification is required"
    ]


async def test_an_ordinary_form_has_no_blockers(loaded_page):
    snapshot = await read(loaded_page)
    assert snapshot.blockers == []


async def test_aria_invalid_fields_are_surfaced(loaded_page):
    await loaded_page.evaluate(
        "document.getElementById('email').setAttribute('aria-invalid', 'true')"
    )
    snapshot = await read(loaded_page)
    assert any("Email" in error for error in snapshot.validation_errors)


async def test_visible_error_elements_are_surfaced(loaded_page):
    await loaded_page.evaluate(
        "const d=document.createElement('div'); d.setAttribute('role','alert');"
        "d.textContent='This field is required'; document.body.appendChild(d);"
    )
    snapshot = await read(loaded_page)
    assert "This field is required" in snapshot.validation_errors


# ------------------------------------------------------------------------------------ sequencing


async def test_snapshot_ids_increase_per_capture(loaded_page):
    reader = PageReader()
    first = await reader.capture(loaded_page)
    second = await reader.capture(loaded_page)
    assert first.snapshot_id != second.snapshot_id


async def test_fingerprint_changes_when_a_value_changes(loaded_page):
    reader = PageReader()
    before = snapshot_fingerprint(await reader.capture(loaded_page))
    await loaded_page.fill("#city", "Bangalore")
    after = snapshot_fingerprint(await reader.capture(loaded_page))
    assert before != after


async def test_fingerprint_ignores_the_snapshot_id(loaded_page):
    """Otherwise every capture would differ from the last and no run would ever look stuck."""
    reader = PageReader()
    first = await reader.capture(loaded_page)
    second = await reader.capture(loaded_page)
    assert first.snapshot_id != second.snapshot_id
    assert snapshot_fingerprint(first) == snapshot_fingerprint(second)


if __name__ == "__main__":  # pragma: no cover - a convenience, not a test path
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
