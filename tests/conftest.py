"""Shared pytest configuration and fixtures.

`helpers` is importable from any test file because the tests directory is placed on sys.path here,
so a test can construct its data inline without a fixture indirection.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import AsyncIterator

import pytest
from playwright.async_api import Browser, Page, async_playwright

TESTS_DIR = Path(__file__).parent
FIXTURES_DIR = TESTS_DIR / "fixtures"
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

from helpers import profile  # noqa: E402

from agent_modules.models import CandidateProfile  # noqa: E402


@pytest.fixture(scope="session")
async def browser() -> AsyncIterator[Browser]:
    """One Chromium for the whole session.

    Launching per test dominates the suite runtime, and the page-level fixtures already isolate each
    test through a fresh context.
    """
    async with async_playwright() as playwright:
        launched = await playwright.chromium.launch(headless=True)
        try:
            yield launched
        finally:
            await launched.close()


@pytest.fixture
async def page(browser: Browser) -> AsyncIterator[Page]:
    """A real headless Chromium page in its own context.

    Browser-bound tests load markup from `tests/fixtures` with `set_content`, so the injected
    JavaScript is genuinely exercised while the test stays offline and hermetic.
    """
    context = await browser.new_context(viewport={"width": 1280, "height": 900})
    created = await context.new_page()
    try:
        yield created
    finally:
        await context.close()


@pytest.fixture
async def loaded_page(page: Page) -> AsyncIterator[Page]:
    """A page preloaded with `tests/fixtures/form.html`, the general-purpose application form."""
    await page.set_content((FIXTURES_DIR / "form.html").read_text(encoding="utf-8"))
    yield page


@pytest.fixture
def sample_profile() -> CandidateProfile:
    """The canonical candidate used by tests that need one but do not care about its shape."""
    return profile()
