"""Fakes for the browser side, so the orchestrator's loop is testable without a real page.

Only the surface the executor and the orchestrator actually touch is implemented. Anything beyond
that would be a mock of Playwright rather than a stand-in for it.
"""

from __future__ import annotations

from typing import Any


class FakeLocator:
    def __init__(self, target: str, page: "FakePage") -> None:
        self.target = target
        self.page = page

    async def wait_for(self, state: str = "visible", timeout: float | None = None) -> None:
        self.page.waits.append((self.target, state))
        if self.page.fail_locate_on and self.target in self.page.fail_locate_on:
            raise TimeoutError(f"{self.target} never appeared")

    async def fill(self, value: str) -> None:
        self.page.filled[self.target] = value

    async def input_value(self) -> str:
        return self.page.filled.get(self.target, "")

    async def select_option(self, value: str) -> None:
        self.page.selected[self.target] = value

    async def set_checked(self, checked: bool) -> None:
        self.page.checked[self.target] = checked

    async def is_checked(self) -> bool:
        return self.page.checked.get(self.target, False)

    async def click(self, **kwargs: Any) -> None:
        self.page.clicked.append(self.target)

    async def press(self, key: str) -> None:
        self.page.pressed.append((self.target, key))

    async def scroll_into_view_if_needed(self) -> None:
        self.page.scrolled_to.append(self.target)

    # Surface used by the overlay cleanup, which enumerates candidate controls and asks about each.
    # Reporting zero keeps it inert on a fake page, which is what these tests want.
    async def count(self) -> int:
        return 0

    def nth(self, index: int) -> "FakeLocator":
        return self

    async def is_visible(self) -> bool:
        return False

    async def is_enabled(self) -> bool:
        return True

    async def get_attribute(self, name: str) -> str | None:
        return None

    async def text_content(self) -> str:
        return ""

    async def inner_text(self) -> str:
        return ""


class FakePage:
    """Just enough of Playwright's Page for the loop and the executor."""

    def __init__(self, url: str = "https://example.com/apply") -> None:
        self.url = url
        self.filled: dict[str, str] = {}
        self.selected: dict[str, str] = {}
        self.checked: dict[str, bool] = {}
        self.clicked: list[str] = []
        self.pressed: list[tuple[str, str]] = []
        self.scrolled_to: list[str] = []
        self.waits: list[tuple[str, str]] = []
        self.load_state_waits: list[str] = []
        self.evaluations: list[Any] = []
        self.timeouts: list[int] = []
        self.goto_error: Exception | None = None
        self.fail_locate_on: set[str] = set()
        #: What a control-count probe reports. Scripted values are consumed first, so a test can
        #: simulate a page that is still rendering and then settles.
        self.control_count = 5
        self.control_counts: list[int] = []
        # Defaults to the shape the readiness probe unpacks, so a fake page is treated as rendered
        # immediately rather than making every test wait out the settle timeout.
        self.evaluate_result: Any = [1, 1]
        self.mouse = self

    def locator(self, selector: str) -> FakeLocator:
        target = selector.rsplit('"', 2)[-2] if '"' in selector else selector
        return FakeLocator(target, self)

    async def goto(self, url: str, wait_until: str | None = None) -> None:
        if self.goto_error is not None:
            raise self.goto_error
        self.url = url

    async def wait_for_load_state(self, state: str, timeout: float | None = None) -> None:
        self.load_state_waits.append(state)

    async def wait_for_timeout(self, ms: int) -> None:
        self.timeouts.append(ms)

    async def evaluate(self, script: Any, arg: Any = None) -> Any:
        self.evaluations.append(arg if arg is not None else script)
        if not isinstance(script, str):
            return self.evaluate_result
        # The readiness probe also ends in `.length`, and it unpacks a pair. Check it first, or it
        # falls through to a count and the caller waits out its whole timeout on an unpack error.
        if "innerText" in script:
            return self.evaluate_result
        # A control-count probe: a stable number ends the settle wait at once.
        if ".length" in script:
            if self.control_counts:
                return self.control_counts.pop(0)
            return self.control_count
        return self.evaluate_result

    async def wheel(self, x: int, y: int) -> None:
        self.evaluations.append(("wheel", y))

    async def title(self) -> str:
        return "Fake page"

    async def inner_text(self) -> str:
        return ""

    def nth(self, index: int) -> "FakePage":
        return self

    async def count(self) -> int:
        return 0

    async def is_visible(self) -> bool:
        return False

    async def is_enabled(self) -> bool:
        return True

    async def set_content(self, html: str) -> None:
        return None

    async def get_attribute(self, name: str) -> str | None:
        return None


class FakeTracing:
    """Stands in for Playwright's tracing API on a browser context."""

    def __init__(self, *, fail_start: bool = False, fail_stop: bool = False) -> None:
        self.fail_start = fail_start
        self.fail_stop = fail_stop
        self.started = False
        self.stopped_path: str | None = None
        self.screenshots: bool | None = None
        self.snapshots: bool | None = None

    async def start(self, screenshots=None, snapshots=None, sources=None) -> None:
        if self.fail_start:
            raise RuntimeError("tracing is unavailable here")
        self.started = True
        self.screenshots = screenshots
        self.snapshots = snapshots

    async def stop(self, path: str | None = None) -> None:
        if self.fail_stop:
            raise RuntimeError("could not write the trace")
        self.stopped_path = path


class FakeContext:
    """Just the surface the orchestrator touches on a browser context."""

    def __init__(self, **tracing_kwargs: Any) -> None:
        self.tracing = FakeTracing(**tracing_kwargs)


class FakeReader:
    """Returns scripted snapshots in order, repeating the last one once exhausted."""

    def __init__(self, snapshots: list[Any]) -> None:
        self.snapshots = list(snapshots)
        self.calls = 0

    async def capture(self, page: Any) -> Any:
        self.calls += 1
        if len(self.snapshots) > 1:
            return self.snapshots.pop(0)
        return self.snapshots[0]


class ScriptedPlanner:
    """Returns scripted plans in order, repeating the last one once exhausted.

    Pass a callable instead of a list to build the plan from the snapshot being answered, which is
    what a real planner does and what a run that must advance past several pages needs.
    """

    def __init__(self, plans: Any, *, raises: list[Exception] | None = None) -> None:
        self.factory = plans if callable(plans) else None
        self.plans = [] if self.factory else list(plans)
        self.raises = list(raises or [])
        self.calls = 0
        self.last_step: dict[str, Any] = {}

    async def next_step(self, profile: Any, snapshot: Any) -> Any:
        self.calls += 1
        if self.raises:
            raise self.raises.pop(0)
        if self.factory is not None:
            plan = self.factory(snapshot)
        else:
            plan = self.plans.pop(0) if len(self.plans) > 1 else self.plans[0]
        self.last_step = {"class": "form", "confidence": 0.9, "actions": len(plan.actions)}
        return plan
