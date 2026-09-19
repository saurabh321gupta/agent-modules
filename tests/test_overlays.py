"""Tests for `agent_modules.overlays`.

The rule under test is asymmetric: missing a banner costs a click that might be intercepted, while
closing the wrong thing destroys the application the run was there to complete.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from playwright.async_api import Page

from agent_modules.overlays import dismiss_nonessential_overlays

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
async def overlay_page(page: Page) -> Page:
    await page.set_content((FIXTURES / "overlays.html").read_text(encoding="utf-8"))
    return page


async def clicked(overlay_page: Page) -> list[str]:
    raw = await overlay_page.get_attribute("body", "data-clicked") or ""
    return sorted(part for part in raw.split(",") if part)


async def test_the_cookie_banner_is_accepted(overlay_page):
    await dismiss_nonessential_overlays(overlay_page)
    assert "cookie" in await clicked(overlay_page)


async def test_the_chat_widget_is_closed(overlay_page):
    await dismiss_nonessential_overlays(overlay_page)
    assert "chat" in await clicked(overlay_page)


async def test_the_application_dialog_is_never_dismissed(overlay_page):
    """Its Close button would throw away the form the run is there to fill in."""
    await dismiss_nonessential_overlays(overlay_page)
    assert "form-close" not in await clicked(overlay_page)


async def test_an_unrelated_modal_is_left_alone(overlay_page):
    """Only dialogs marked as non-essential are touched, and only via a close-style control."""
    await dismiss_nonessential_overlays(overlay_page)
    assert "notice-close" not in await clicked(overlay_page)


async def test_reported_actions_describe_what_was_closed(overlay_page):
    actions = await dismiss_nonessential_overlays(overlay_page)
    kinds = {action["kind"] for action in actions}
    assert kinds <= {"chat", "cookie", "dialog"}
    assert "chat" in kinds
    assert "cookie" in kinds


async def test_a_page_with_no_overlays_reports_nothing(page: Page):
    await page.set_content("<h1>Plain form</h1><input aria-label='City' />")
    assert await dismiss_nonessential_overlays(page) == []


if __name__ == "__main__":  # pragma: no cover - a convenience, not a test path
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
