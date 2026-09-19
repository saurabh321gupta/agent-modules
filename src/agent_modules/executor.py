"""Performs a validated plan in the browser. Layer 2 (browser-bound).

Provider-agnostic and deliberately the only place that touches the page to change it. Both the
generative planner and the Jev planner emit the same `Action` type, so there is exactly one executor
rather than one per provider - two copies of the six primitives would drift.

Every primitive is verified after it runs, because a browser action that silently did nothing is the
most expensive failure mode: the planner is told it succeeded and plans the next step on a false
premise.
"""

from __future__ import annotations

import os

from playwright.async_api import Locator, Page

from .journey import JourneyLogger
from .types import Action, ActionResult, CandidateProfile, PageSnapshot
from .validator import resolve_profile_ref


def structure_of(snapshot: PageSnapshot) -> list[tuple[str, str, str, str | None]]:
    """The shape of a page, ignoring values. Used to decide whether re-tagging is safe."""
    return [
        (element.id, element.role, element.label, element.input_type)
        for element in snapshot.elements
    ]


def _is_readonly(snapshot: PageSnapshot, target: str) -> bool:
    element = next((item for item in snapshot.elements if item.id == target), None)
    return bool(element.readonly) if element else False


class BrowserExecutor:
    def __init__(
        self,
        page: Page,
        assets: dict[str, str],
        action_timeout_ms: int = 8_000,
        journey: JourneyLogger | None = None,
        reader: object | None = None,
        field_pause_s: float = 0.0,
    ) -> None:
        self.page = page
        self.assets = assets
        self.action_timeout_ms = action_timeout_ms
        self.journey = journey
        self.reader = reader
        self.field_pause_s = field_pause_s

    def locator_for(self, snapshot: PageSnapshot, target: str) -> Locator:
        return self.page.locator(
            f'[data-agent-snapshot="{snapshot.snapshot_id}"][data-agent-id="{target}"]'
        )

    async def locate(self, action: Action, snapshot: PageSnapshot) -> Locator | None:
        """Find the target, retrying past lazy regions and re-renders before giving up."""
        if not action.target:
            return None
        # A file input is usually hidden, so waiting for visibility on an upload would always fail.
        state = "attached" if action.type == "upload" else "visible"
        probe_ms = min(self.action_timeout_ms, 2_500)
        locator = self.locator_for(snapshot, action.target)
        try:
            await locator.wait_for(state=state, timeout=probe_ms)
            return locator
        except Exception as first_error:
            # A control inside a lazy or virtualised region may only render once it is scrolled to.
            await self.page.mouse.wheel(0, 1_200)
            await self.page.wait_for_timeout(300)
            try:
                await locator.wait_for(state=state, timeout=probe_ms)
                return locator
            except Exception:
                fresh = await self.retag_if_unchanged(snapshot)
                if fresh is None:
                    raise first_error
                locator = self.locator_for(fresh, action.target)
                await locator.wait_for(state=state, timeout=self.action_timeout_ms)
                return locator

    async def retag_if_unchanged(self, snapshot: PageSnapshot) -> PageSnapshot | None:
        """Re-tag the DOM when a re-render dropped our markers, but only when the layout is identical.

        Returning None means the page has changed underneath us; the caller re-raises rather than
        acting on a stale element id, which is how a value ends up in the wrong field.
        """
        if self.reader is None:
            return None
        fresh = await self.reader.capture(self.page)  # type: ignore[attr-defined]
        if structure_of(fresh) != structure_of(snapshot):
            return None
        return fresh

    async def execute(
        self,
        actions: list[Action],
        snapshot: PageSnapshot,
        profile: CandidateProfile,
    ) -> list[ActionResult]:
        results: list[ActionResult] = []
        for action in actions:
            if self.journey:
                self.journey.log("action_start", action=action)
            try:
                await self.execute_one(action, snapshot, profile)
                result = ActionResult(action=action, ok=True, message="Executed and verified")
                results.append(result)
                if self.journey:
                    self.journey.log("action_result", result=result)
                # A click, upload or scroll changes the page, so the rest of the batch is stale.
                if action.type in {"click", "upload", "scroll"}:
                    break
            except Exception as exc:
                result = ActionResult(action=action, ok=False, message=str(exc))
                results.append(result)
                if self.journey:
                    self.journey.log("action_result", result=result)
                break
        return results

    async def execute_one(
        self, action: Action, snapshot: PageSnapshot, profile: CandidateProfile
    ) -> None:
        locator = await self.locate(action, snapshot)

        if action.type == "fill":
            assert locator is not None
            value = resolve_profile_ref(profile, action.value_ref) if action.value_ref else action.value
            await locator.fill(str(value))
            actual = await locator.input_value()
            if actual != str(value):
                raise RuntimeError(
                    f"fill verification failed for {action.target}: "
                    f"wrote {str(value)!r} but the field now reads {actual!r}"
                    + (" (field is readonly)" if _is_readonly(snapshot, action.target) else "")
                )
            # A site may run its own validation on entry; leaving too soon can disturb it.
            if self.field_pause_s:
                await self.page.wait_for_timeout(int(self.field_pause_s * 1_000))
            # Some controlled forms commit a field only on blur before rerendering the next control.
            # Tab is a deterministic, non-submitting blur.
            await locator.press("Tab")
            await self.page.wait_for_timeout(500)
            # Read the field back after the blur: a field rewritten by the site's own validation
            # passes the check above and then silently reverts.
            after_blur = await locator.input_value()
            if after_blur != str(value) and self.journey:
                self.journey.log(
                    "field_reverted",
                    target=action.target,
                    label=next(
                        (e.label for e in snapshot.elements if e.id == action.target), None
                    ),
                    wrote=str(value),
                    after_blur=after_blur,
                    note="The page rewrote this field once focus left it.",
                )
        elif action.type == "select":
            assert locator is not None and action.option_value is not None
            await locator.select_option(action.option_value)
            if await locator.input_value() != action.option_value:
                raise RuntimeError(f"select verification failed for {action.target}")
        elif action.type == "set_checked":
            assert locator is not None and action.checked is not None
            await locator.set_checked(action.checked)
            if await locator.is_checked() != action.checked:
                raise RuntimeError(f"checkbox verification failed for {action.target}")
        elif action.type == "upload":
            assert locator is not None and action.asset_id is not None
            path = self.assets[action.asset_id]
            if not os.path.isfile(path):
                raise RuntimeError(f"Upload asset does not exist: {action.asset_id}")
            await locator.set_input_files(path)
        elif action.type == "click":
            assert locator is not None
            try:
                await locator.click()
            except Exception as exc:
                # A transient floating widget (a site chatbot, say) can intercept a click on an
                # otherwise visible, validated target. Force only this already-validated target;
                # never invent a locator.
                if "intercepts pointer events" not in str(exc):
                    raise
                await locator.click(force=True)
        elif action.type == "scroll":
            if locator:
                await locator.scroll_into_view_if_needed()
            else:
                await self.page.mouse.wheel(0, 700)
        else:
            raise RuntimeError(f"Unsupported action type: {action.type}")
        await self.page.wait_for_timeout(150)
