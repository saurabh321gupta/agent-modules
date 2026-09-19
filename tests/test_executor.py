"""Tests for `agent_modules.executor`.

Drives real actions in a real headless Chromium against `tests/fixtures/executor.html`, so both the
browser primitive and its verification are exercised. No network is involved.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from helpers import action, profile
from playwright.async_api import Page

from agent_modules.executor import BrowserExecutor, structure_of
from agent_modules.journey import JourneyLogger
from agent_modules.reader import PageReader

FIXTURES = Path(__file__).parent / "fixtures"
APPLICANT = profile()


def id_of(snapshot, label):
    found = next((e for e in snapshot.elements if e.label == label), None)
    assert found is not None, f"no element labelled {label!r}; got {[e.label for e in snapshot.elements]}"
    return found.id


@pytest.fixture
async def executor_page(page: Page) -> Page:
    await page.set_content((FIXTURES / "executor.html").read_text(encoding="utf-8"))
    return page


@pytest.fixture
async def captured(executor_page: Page):
    reader = PageReader()
    return reader, await reader.capture(executor_page)


def make_executor(page, reader, **kwargs) -> BrowserExecutor:
    return BrowserExecutor(page, {}, reader=reader, **kwargs)


# ----------------------------------------------------------------------------------------- fill


async def test_fill_writes_and_verifies(executor_page, captured):
    reader, snapshot = captured
    executor = make_executor(executor_page, reader)
    results = await executor.execute(
        [action("fill", id_of(snapshot, "Plain"), value="Bangalore")], snapshot, APPLICANT
    )
    assert results[0].ok is True
    assert await executor_page.input_value("#plain") == "Bangalore"


async def test_fill_from_a_profile_reference(executor_page, captured):
    reader, snapshot = captured
    executor = make_executor(executor_page, reader)
    results = await executor.execute(
        [action("fill", id_of(snapshot, "Plain"), value_ref="location.city")], snapshot, APPLICANT
    )
    assert results[0].ok is True
    assert await executor_page.input_value("#plain") == "Bangalore"


async def test_a_field_that_reverts_its_own_typing_fails_verification(executor_page, captured):
    """The most expensive silent failure: a write that did not take, reported as success."""
    reader, snapshot = captured
    executor = make_executor(executor_page, reader)
    results = await executor.execute(
        [action("fill", id_of(snapshot, "Reverting"), value="x")], snapshot, APPLICANT
    )
    assert results[0].ok is False
    assert "fill verification failed" in results[0].message


async def test_a_field_rewritten_on_blur_is_logged_as_reverted(executor_page, captured, tmp_path):
    """The site accepts the value then changes it when focus leaves, which is how the email field
    silently reverted while still passing the immediate read-back."""
    reader, snapshot = captured
    logger = JourneyLogger(str(tmp_path / "j.jsonl"))
    executor = make_executor(executor_page, reader, journey=logger)
    results = await executor.execute(
        [action("fill", id_of(snapshot, "Blur revert"), value="bangalore")], snapshot, APPLICANT
    )
    logger.close()
    assert results[0].ok is True, "the write itself succeeded; only the site's rewrite is the problem"
    records = [json.loads(line) for line in (tmp_path / "j.jsonl").read_text().splitlines() if line]
    reverted = [r for r in records if r["event"] == "field_reverted"]
    assert len(reverted) == 1
    assert reverted[0]["data"]["wrote"] == "bangalore"
    assert reverted[0]["data"]["after_blur"] == "BANGALORE"
    assert "Field rewritten" in (tmp_path / "j.log").read_text()


# --------------------------------------------------------------------------------------- select


async def test_select_sets_the_option_by_value(executor_page, captured):
    reader, snapshot = captured
    executor = make_executor(executor_page, reader)
    results = await executor.execute(
        [action("select", id_of(snapshot, "Country"), option_value="IND")], snapshot, APPLICANT
    )
    assert results[0].ok is True
    assert await executor_page.input_value("#country") == "IND"


# ----------------------------------------------------------------------------------- set_checked


async def test_set_checked_ticks_the_box(executor_page, captured):
    reader, snapshot = captured
    executor = make_executor(executor_page, reader)
    results = await executor.execute(
        [action("set_checked", id_of(snapshot, "I consent"), checked=True)], snapshot, APPLICANT
    )
    assert results[0].ok is True
    assert await executor_page.is_checked("#agree") is True


# ---------------------------------------------------------------------------------------- click


async def test_click_presses_the_control(executor_page, captured):
    reader, snapshot = captured
    executor = make_executor(executor_page, reader)
    results = await executor.execute(
        [action("click", id_of(snapshot, "Go"))], snapshot, APPLICANT
    )
    assert results[0].ok is True
    assert await executor_page.get_attribute("body", "data-clicked") == "yes"


# --------------------------------------------------------------------------------------- upload


async def test_upload_targets_a_hidden_file_input(executor_page, captured, tmp_path):
    """Regression, end to end: the resume input is display:none.

    A visibility wait on an upload would always time out, and dropping the control upstream meant no
    upload was ever planned. This covers extractor -> validator -> executor for that control.
    """
    reader, snapshot = captured
    file_input_id = next(e.id for e in snapshot.elements if e.input_type == "file")
    assert next(e for e in snapshot.elements if e.id == file_input_id).visible is False

    asset = tmp_path / "resume.pdf"
    asset.write_bytes(b"%PDF-1.4 test")
    executor = BrowserExecutor(executor_page, {"resume_primary": str(asset)}, reader=reader)
    results = await executor.execute(
        [action("upload", file_input_id, asset_id="resume_primary")], snapshot, APPLICANT
    )
    assert results[0].ok is True
    assert await executor_page.evaluate("document.getElementById('resume').files.length") == 1


async def test_upload_of_a_missing_asset_fails(executor_page, captured):
    reader, snapshot = captured
    file_input_id = next(e.id for e in snapshot.elements if e.input_type == "file")
    executor = BrowserExecutor(executor_page, {"resume_primary": "/nope/missing.pdf"}, reader=reader)
    results = await executor.execute(
        [action("upload", file_input_id, asset_id="resume_primary")], snapshot, APPLICANT
    )
    assert results[0].ok is False
    assert "does not exist" in results[0].message


# --------------------------------------------------------------------------------------- scroll


async def test_scroll_with_a_target_brings_it_into_view(executor_page, captured):
    reader, snapshot = captured
    executor = make_executor(executor_page, reader)
    results = await executor.execute(
        [action("scroll", id_of(snapshot, "Far away"))], snapshot, APPLICANT
    )
    assert results[0].ok is True


async def test_scroll_without_a_target_moves_the_page(executor_page, captured):
    reader, snapshot = captured
    executor = make_executor(executor_page, reader)
    results = await executor.execute([action("scroll")], snapshot, APPLICANT)
    assert results[0].ok is True


# ------------------------------------------------------------------------------- batch behaviour


async def test_a_click_ends_the_batch(executor_page, captured):
    """Everything after a click was planned against a page that no longer exists."""
    reader, snapshot = captured
    executor = make_executor(executor_page, reader)
    results = await executor.execute(
        [
            action("click", id_of(snapshot, "Go")),
            action("fill", id_of(snapshot, "Plain"), value="never reached"),
        ],
        snapshot,
        APPLICANT,
    )
    assert len(results) == 1
    assert await executor_page.input_value("#plain") == ""


async def test_an_upload_ends_the_batch(executor_page, captured, tmp_path):
    reader, snapshot = captured
    asset = tmp_path / "resume.pdf"
    asset.write_bytes(b"%PDF-1.4")
    executor = BrowserExecutor(executor_page, {"resume_primary": str(asset)}, reader=reader)
    results = await executor.execute(
        [
            action("upload", next(e.id for e in snapshot.elements if e.input_type == "file"),
                   asset_id="resume_primary"),
            action("fill", id_of(snapshot, "Plain"), value="never reached"),
        ],
        snapshot,
        APPLICANT,
    )
    assert len(results) == 1


async def test_a_failure_ends_the_batch(executor_page, captured):
    """Continuing after a failure would act on a page state we no longer understand."""
    reader, snapshot = captured
    executor = make_executor(executor_page, reader)
    results = await executor.execute(
        [
            action("fill", id_of(snapshot, "Reverting"), value="x"),
            action("fill", id_of(snapshot, "Plain"), value="never reached"),
        ],
        snapshot,
        APPLICANT,
    )
    assert len(results) == 1
    assert results[0].ok is False
    assert await executor_page.input_value("#plain") == ""


async def test_a_batch_of_fills_runs_to_completion(executor_page, captured):
    """The batch limit exists to be used: ten independent fills should all run."""
    reader, snapshot = captured
    executor = make_executor(executor_page, reader)
    plain = id_of(snapshot, "Plain")
    results = await executor.execute(
        [action("fill", plain, value=f"v{i}") for i in range(5)], snapshot, APPLICANT
    )
    assert len(results) == 5
    assert all(r.ok for r in results)
    assert await executor_page.input_value("#plain") == "v4"


# ----------------------------------------------------------------------------------- field pause


async def test_field_pause_is_honoured(executor_page, captured):
    reader, snapshot = captured
    executor = make_executor(executor_page, reader, field_pause_s=0.6)
    results = await executor.execute(
        [action("fill", id_of(snapshot, "Plain"), value="slow")], snapshot, APPLICANT
    )
    assert results[0].ok is True
    assert await executor_page.input_value("#plain") == "slow"


async def test_zero_field_pause_is_the_default(executor_page, captured):
    reader, snapshot = captured
    executor = make_executor(executor_page, reader)
    assert executor.field_pause_s == 0.0


# ---------------------------------------------------------------------------------- locate retry


async def test_a_target_that_renders_late_is_found_by_the_retry(executor_page):
    """Lazy and virtualised regions render a control only after a scroll, so the first probe misses.

    The control is hidden for longer than one probe window, which forces the scroll-and-re-probe
    branch rather than the immediate hit.
    """
    reader = PageReader()
    snapshot = await reader.capture(executor_page)
    target = id_of(snapshot, "Far away")
    await executor_page.evaluate(
        """([sid, id]) => {
            const el = document.querySelector(
                `[data-agent-snapshot="${sid}"][data-agent-id="${id}"]`
            );
            el.style.display = 'none';
            setTimeout(() => { el.style.display = 'inline-block'; }, 1000);
        }""",
        [snapshot.snapshot_id, target],
    )
    executor = make_executor(executor_page, reader, action_timeout_ms=800)
    results = await executor.execute([action("click", target)], snapshot, APPLICANT)
    assert results[0].ok is True


async def test_structure_comparison_ignores_values(captured):
    _, snapshot = captured
    assert structure_of(snapshot) == structure_of(snapshot)
    changed = snapshot.model_copy(deep=True)
    changed.elements[0].value = "something else"
    assert structure_of(changed) == structure_of(snapshot)


async def test_structure_comparison_detects_a_real_change(captured):
    _, snapshot = captured
    changed = snapshot.model_copy(deep=True)
    changed.elements.pop()
    assert structure_of(changed) != structure_of(snapshot)


async def test_retag_declines_when_the_page_changed(executor_page, captured):
    """Re-tagging a changed page would renumber ids and write into the wrong field."""
    reader, snapshot = captured
    executor = make_executor(executor_page, reader)
    await executor_page.evaluate(
        "const d=document.createElement('input'); d.id='extra'; d.setAttribute('aria-label','Extra');"
        "document.body.appendChild(d);"
    )
    assert await executor.retag_if_unchanged(snapshot) is None


async def test_retag_accepts_an_unchanged_page(executor_page, captured):
    reader, snapshot = captured
    executor = make_executor(executor_page, reader)
    fresh = await executor.retag_if_unchanged(snapshot)
    assert fresh is not None
    assert fresh.snapshot_id != snapshot.snapshot_id


if __name__ == "__main__":  # pragma: no cover - a convenience, not a test path
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
