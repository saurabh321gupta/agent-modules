"""The run loop: browser lifecycle, budget, and the retry policies. Layer 5.

Everything above this layer is a pure decision; everything below it is a single capability. This is
where the two meet, and it is deliberately the only place that owns mutable run state.

Three policies earn their complexity, because each one was a real failure:

- **A wall-clock budget.** An application that has not been submitted in time is discarded rather
  than submitted late.
- **No-progress detection.** A page that is byte-identical three times running is not going to
  advance, and each further step costs a model call to discover that again.
- **Reduced execution.** One disabled control must not discard the rest of an otherwise valid batch.
"""

from __future__ import annotations

import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

from playwright.async_api import async_playwright

from .config import HEADLESS_VIEWPORT, MIN_AUTO_ZOOM, RunConfig
from .executor import BrowserExecutor
from .journey import JourneyLogger
from .normalizer import Normalizer
from .planner_jev import JevPlanner, LlmFieldAssist
from .planner_llm import LLMPlanner
from .planner_staged import StagedPlanner
from .policy import default_consent_actions
from .prompts_llm import FORM_SYSTEM_PROMPT
from .reader import PageReader, snapshot_fingerprint
from .models import ActionResult, ApplicationPlan, CandidateProfile, PageSnapshot, RunResult
from .validator import PlanValidationError, resolve_profile_ref, validate_plan
from .verifier import verify_completion

#: How many times a plan may be retried for a reason that looks transient.
MAX_TRANSIENT_REJECTIONS = 2
#: How many times to re-observe a page that has not rendered yet.
EMPTY_SNAPSHOT_RETRIES = 4
#: How many identical snapshots in a row mean the run is not moving.
NO_PROGRESS_LIMIT = 3
#: How many contradictory "already satisfied" plans before giving up.
STALE_PLAN_LIMIT = 3
#: How long to wait for a disabled control to come back, without spending a model call.
DISABLED_WAIT_MS = 6_000
#: A slice of the budget is kept back so teardown and the final observe always fit inside it.
DISABLED_TARGET = re.compile(r"Target (\S+) is disabled")


def build_planner(
    config: RunConfig,
    assets: dict[str, str],
    journey: JourneyLogger,
    llm_client: Any,
) -> tuple[Any, list[Any], list[Any]]:
    """Choose the planner for this run.

    Returns the planner plus the clients the orchestrator must keep an eye on: `budgeted` clients
    have their per-call timeout clamped to the remaining run budget, `closeable` ones are closed at
    the end.
    """
    budgeted: list[Any] = []
    closeable: list[Any] = []
    if llm_client is not None:
        budgeted.append(llm_client)
        closeable.append(llm_client)

    if config.planner == "staged":
        classifier = _jev_client(config, journey)
        budgeted.append(classifier)
        planner = StagedPlanner(
            classifier,
            Normalizer(llm_client, config.model, journey=journey),
            LLMPlanner(
                llm_client,
                config.model,
                assets,
                config.defaults,
                config.answers,
                journey=journey,
                system_prompt=FORM_SYSTEM_PROMPT,
            ),
            normalise=config.normalise,
            journey=journey,
        )
    elif config.planner == "jev":
        planner = JevPlanner(
            _jev_client(config, journey),
            assets,
            config.defaults,
            config.answers,
            confidence_floor=config.jev_confidence_floor,
            field_assist=LlmFieldAssist(llm_client, config.model, journey=journey)
            if llm_client is not None
            else None,
            journey=journey,
        )
    else:
        planner = LLMPlanner(
            llm_client, config.model, assets, config.defaults, config.answers, journey=journey
        )
    return planner, budgeted, closeable


def _jev_client(config: RunConfig, journey: JourneyLogger) -> Any:
    from .jev_client import JevClient

    return JevClient(config.jev_api_key or "", model=config.jev_model, journey=journey)


async def run_application(
    url: str,
    profile: CandidateProfile,
    assets: dict[str, str],
    config: RunConfig | None = None,
) -> RunResult:
    config = config or RunConfig()
    journey = JourneyLogger(
        config.journey_log_path,
        secret_values=[value for value in (config.api_key, config.jev_api_key) if value],
    )
    journey.log(
        "run_started",
        url=url,
        model=config.model,
        planner=config.planner,
        jev_model=config.jev_model if config.planner in ("staged", "jev") else None,
        base_url=config.base_url,
        headless=config.headless,
        max_steps=config.max_steps,
        allow_submission=config.allow_submission,
        zoom=config.zoom,
        answers=len(config.answers),
        answers_path=config.answers_path,
        field_pause_s=config.field_pause_s,
        normalise=config.normalise,
        reasoning_effort=config.reasoning_effort,
        request_timeout_s=config.request_timeout_s,
        hedge_after_s=config.hedge_after_s,
        max_run_seconds=config.max_run_seconds,
        trace_path=config.trace_path,
        defaults=config.defaults,
        asset_ids=sorted(assets),
    )
    run = _ApplicationRun(url, profile, assets, config, journey)
    try:
        result = await run.execute()
    except Exception as exc:
        journey.log("run_exception", error=str(exc))
        result = RunResult(
            status="failed",
            message=f"Unhandled runner error: {exc}",
            url=url,
            steps=0,
            evidence=None,
            action_results=[],
        )
    finally:
        result.journey_log = journey.human_path
        result.journey_jsonl = journey.path
        result.journey_timing = journey.timing_path
        journey.log("run_finished", result=result)
        journey.close()
    return result


class _ApplicationRun:
    """One application attempt. Holds the state the loop needs; nothing else does."""

    def __init__(
        self,
        url: str,
        profile: CandidateProfile,
        assets: dict[str, str],
        config: RunConfig,
        journey: JourneyLogger,
    ) -> None:
        self.url = url
        self.profile = profile
        self.assets = assets
        self.config = config
        self.journey = journey
        self.reader = PageReader()
        self.history: list[dict[str, Any]] = []
        self.fingerprints: Counter[str] = Counter()
        self.stale_plans: Counter[str] = Counter()
        self.results: list[ActionResult] = []
        self.transient_rejections = 0
        self.deadline = 0.0

    # ------------------------------------------------------------------------------------ setup

    async def execute(self) -> RunResult:
        if not self.url.startswith(("http://", "https://")):
            return self.fail("URL must use http:// or https://")

        llm_client = self.build_llm_client()
        if isinstance(llm_client, RunResult):
            return llm_client
        planner, budgeted, closeable = build_planner(
            self.config, self.assets, self.journey, llm_client
        )
        self.budgeted = budgeted
        self.deadline = time.monotonic() + self.config.max_run_seconds

        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(
                headless=self.config.headless,
                args=[] if self.config.headless else ["--start-maximized"],
            )
            context = (
                await browser.new_context(viewport=HEADLESS_VIEWPORT)
                if self.config.headless
                else await browser.new_context(no_viewport=True)
            )
            if self.config.zoom != 1.0:
                await context.add_init_script(_zoom_script(self.config.zoom))
            tracing = await self.start_tracing(context)
            page = await context.new_page()
            try:
                return await self.loop(page, planner)
            finally:
                # Teardown must never replace the real outcome with its own error, and the trace has
                # to be written before the context that is recording it is closed.
                if tracing:
                    await self.stop_tracing(context)
                try:
                    await context.close()
                except Exception as exc:
                    self.journey.log("teardown_error", target="context", error=str(exc))
                try:
                    await browser.close()
                except Exception as exc:
                    self.journey.log("teardown_error", target="browser", error=str(exc))

    async def start_tracing(self, context: Any) -> bool:
        """Begin recording a Playwright trace.

        A trace is the only artefact that shows what the page actually looked like when a decision
        was made, so it answers questions a snapshot dict and a stack frame cannot. It is opt-in
        because it costs disk and a little speed on every action.
        """
        if not self.config.trace_path:
            return False
        try:
            await context.tracing.start(screenshots=True, snapshots=True, sources=True)
        except Exception as exc:
            self.journey.log("trace_failed", phase="start", error=str(exc), path=self.config.trace_path)
            return False
        self.journey.log(
            "trace_started",
            path=self.config.trace_path,
            note=(
                "Recording screenshots, DOM snapshots and every action. Replay it with "
                "'playwright show-trace <path>'."
            ),
        )
        return True

    async def stop_tracing(self, context: Any) -> None:
        path = self.config.trace_path
        try:
            await context.tracing.stop(path=path)
        except Exception as exc:
            # A run that died mid-flight is exactly when a trace is worth having, so report whether
            # anything usable survived rather than only that the stop failed.
            written = Path(path).stat().st_size if path and Path(path).exists() else 0
            self.journey.log(
                "trace_failed",
                phase="stop",
                error=str(exc),
                path=path,
                bytes_written=written,
                note=(
                    "A partial trace was still written and may be readable."
                    if written
                    else "No trace file was produced."
                ),
            )
            return
        self.journey.log("trace_written", path=path)

    def build_llm_client(self) -> Any:
        """Create the model client, or return the `RunResult` explaining why the run cannot start."""
        if self.config.planner == "staged" and not self.config.jev_api_key:
            return self.fail(
                "The staged planner needs a Jev API key for page classification "
                "(--jev-api-key-file)."
            )
        if self.config.planner == "jev" and not self.config.jev_api_key:
            return self.fail("The jev planner needs a Jev API key (--jev-api-key-file).")
        if not self.config.api_key:
            if self.config.planner == "jev":
                return None
            return self.fail("No LLM API key was provided (pass --api-key-file).")
        from .llm_client import DeepSeekClient

        return DeepSeekClient(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            model=self.config.model,
            request_timeout_s=self.config.request_timeout_s,
            hedge_after_s=self.config.hedge_after_s,
            reasoning_effort=self.config.reasoning_effort,
            journey=self.journey,
        )

    def fail(self, message: str, status: str = "failed") -> RunResult:
        return RunResult(
            status=status,
            message=message,
            url=self.url,
            steps=0,
            evidence=None,
            action_results=[],
        )

    # ------------------------------------------------------------------------------------- loop

    async def loop(self, page: Any, planner: Any) -> RunResult:
        try:
            await page.goto(self.url, wait_until="domcontentloaded")
        except Exception as exc:
            return self.fail(f"Could not open application URL: {exc}")

        await _wait_for_interactive_form(page, self.config.settle_timeout_ms)
        executor = BrowserExecutor(
            page,
            self.assets,
            journey=self.journey,
            reader=self.reader,
            field_pause_s=self.config.field_pause_s,
        )

        for step in range(1, self.config.max_steps + 1):
            remaining = self.deadline - time.monotonic()
            if remaining <= 0:
                self.journey.log(
                    "run_timeout",
                    max_run_seconds=self.config.max_run_seconds,
                    elapsed_s=round(self.config.max_run_seconds - remaining, 1),
                )
                return RunResult(
                    status="timeout",
                    message=(
                        f"Discarded without submitting: the {self.config.max_run_seconds:.0f}s "
                        "budget was exhausted."
                    ),
                    url=page.url,
                    steps=step - 1,
                    evidence=None,
                    action_results=self.results,
                )
            self.clamp_budget(remaining)
            outcome = await self.step(page, planner, executor, step)
            if outcome is not None:
                return outcome

        return RunResult(
            status="step_limit",
            message=f"Stopped after {self.config.max_steps} steps",
            url=page.url,
            steps=self.config.max_steps,
            evidence=None,
            action_results=self.results,
        )

    def clamp_budget(self, remaining: float) -> None:
        """Give every in-flight client a per-call bound that cannot outlive the run."""
        for client in self.budgeted:
            client.request_timeout_s = max(5.0, min(self.config.request_timeout_s, remaining))

    async def step(
        self, page: Any, planner: Any, executor: BrowserExecutor, step: int
    ) -> RunResult | None:
        """Run one iteration. Returns a result to stop, or None to continue."""
        started = time.perf_counter()
        timings: dict[str, float] = {}
        try:
            # Time the settle too: waiting for a page to go idle is one of the largest single
            # consumers of wall clock, and leaving it untimed makes it look like unattributed time.
            settle_started = time.perf_counter()
            await self.settle(page)
            await self.fit_zoom(page)
            timings["settle_ms"] = (time.perf_counter() - settle_started) * 1_000

            observe_started = time.perf_counter()
            snapshot = await self.reader.capture(page)
            timings["observe_ms"] = (time.perf_counter() - observe_started) * 1_000

            if not snapshot.elements:
                recovered = await self.recover_empty_snapshot(page, snapshot)
                if recovered is None:
                    return RunResult(
                        status="blocked",
                        message="The page never rendered any controls",
                        url=page.url,
                        steps=step,
                        evidence=None,
                        action_results=self.results,
                    )
                snapshot = recovered
                timings["observe_ms"] = (time.perf_counter() - observe_started) * 1_000

            self.journey.log("page_snapshot", step=step, snapshot=snapshot)

            if evidence := verify_completion(snapshot):
                return RunResult(
                    status="success",
                    message="Application completion verified",
                    url=page.url,
                    steps=step,
                    evidence=evidence,
                    action_results=self.results,
                )
            if snapshot.blockers:
                return RunResult(
                    status="blocked",
                    message="; ".join(snapshot.blockers),
                    url=page.url,
                    steps=step,
                    evidence=None,
                    action_results=self.results,
                )

            fingerprint = snapshot_fingerprint(snapshot)
            self.fingerprints[fingerprint] += 1
            if self.fingerprints[fingerprint] >= NO_PROGRESS_LIMIT:
                return RunResult(
                    status="blocked",
                    message="No page progress after repeated identical snapshots",
                    url=page.url,
                    steps=step,
                    evidence=None,
                    action_results=self.results,
                )

            consent_actions = default_consent_actions(snapshot, self.config.defaults)
            if consent_actions:
                execute_started = time.perf_counter()
                results = await executor.execute(consent_actions, snapshot, self.profile)
                timings["execute_ms"] = (time.perf_counter() - execute_started) * 1_000
                self.record(snapshot, consent_actions, results, reason="applied_user_consent_default")
                return None

            return await self.plan_and_act(page, planner, executor, snapshot, step, timings)
        finally:
            timings["total_ms"] = (time.perf_counter() - started) * 1_000
            decision = getattr(planner, "last_step", None)
            self.journey.log(
                "step_timing",
                step=step,
                **{name: round(value, 1) for name, value in timings.items()},
                **(
                    {
                        "decision": f"{decision['class']} "
                        f"({float(decision.get('confidence') or 0):.2f}), "
                        f"{decision.get('actions', 0)} action(s)"
                    }
                    if decision
                    else {}
                ),
            )

    async def settle(self, page: Any) -> None:
        try:
            await page.wait_for_load_state("networkidle", timeout=self.config.step_settle_timeout_ms)
        except Exception:
            pass
        from .overlays import dismiss_nonessential_overlays

        overlay_actions = await dismiss_nonessential_overlays(page)
        if overlay_actions:
            self.journey.log("overlay_cleanup", actions=overlay_actions)
            try:
                await page.wait_for_load_state("networkidle", timeout=500)
            except Exception:
                pass

    async def fit_zoom(self, page: Any) -> None:
        if self.config.zoom != 1.0:
            await _fit_page_zoom(page)

    async def recover_empty_snapshot(self, page: Any, snapshot: PageSnapshot) -> PageSnapshot | None:
        """A page that has not painted yet looks exactly like a page with no form.

        Observing an unrendered shell and concluding there is nothing to do ends the run before the
        site has drawn anything, so give it a bounded chance to appear.
        """
        for attempt in range(1, EMPTY_SNAPSHOT_RETRIES + 1):
            self.journey.log(
                "empty_snapshot",
                attempt=attempt,
                reason="The page rendered no controls; waiting for it to paint before giving up.",
            )
            await page.wait_for_timeout(750)
            fresh = await self.reader.capture(page)
            if fresh.elements:
                return fresh
        return None

    # ----------------------------------------------------------------------------- plan and act

    async def plan_and_act(
        self,
        page: Any,
        planner: Any,
        executor: BrowserExecutor,
        snapshot: PageSnapshot,
        step: int,
        timings: dict[str, float],
    ) -> RunResult | None:
        plan: ApplicationPlan | None = None
        try:
            plan_started = time.perf_counter()
            plan = await planner.next_step(self.profile, snapshot, self.history)
            timings["plan_ms"] = (time.perf_counter() - plan_started) * 1_000
            self.journey.log("plan_received", step=step, plan=plan)
            validate_started = time.perf_counter()
            validate_plan(
                plan, snapshot, self.profile, self.assets, self.config.allow_submission
            )
            timings["validate_ms"] = (time.perf_counter() - validate_started) * 1_000
            self.journey.log("plan_validated", step=step, plan=plan)
        except PlanValidationError as exc:
            return await self.on_rejected(page, planner, executor, snapshot, plan, step, str(exc), timings)
        except Exception as exc:
            return RunResult(
                status="failed",
                message=f"Planner error: {exc}",
                url=page.url,
                steps=step,
                evidence=None,
                action_results=self.results,
            )

        if plan.status == "complete":
            # Only visible confirmation proves success, and the verifier already looked.
            self.history.append(
                {
                    "snapshot_id": snapshot.snapshot_id,
                    "event": "completion_not_verified",
                    "reason": plan.reason,
                }
            )
            return None

        if plan.status == "needs_input" and self.actions_already_satisfied(plan, snapshot):
            return await self.on_stale_plan(page, snapshot, plan, step)

        if plan.status in {"needs_input", "blocked", "failed"}:
            return RunResult(
                status="needs_input" if plan.status == "needs_input" else plan.status,
                message=plan.reason,
                url=page.url,
                steps=step,
                evidence=plan.completion_evidence,
                action_results=self.results,
            )

        if not plan.actions:
            if self.transient_rejections < MAX_TRANSIENT_REJECTIONS:
                self.transient_rejections += 1
                self.journey.log(
                    "plan_retry",
                    step=step,
                    error="Planner returned continue with no actions",
                    attempt=self.transient_rejections,
                    reason=(
                        "The page may still be mid-transition (a control can be disabled while it "
                        "saves); re-observing instead of stopping."
                    ),
                )
                await page.wait_for_timeout(2_000)
                return None
            return RunResult(
                status="blocked",
                message="Planner returned continue with no actions",
                url=page.url,
                steps=step,
                evidence=None,
                action_results=self.results,
            )

        execute_started = time.perf_counter()
        results = await executor.execute(plan.actions, snapshot, self.profile)
        timings["execute_ms"] = (time.perf_counter() - execute_started) * 1_000
        self.transient_rejections = 0
        self.record(snapshot, plan.actions, results)
        return None

    def record(
        self,
        snapshot: PageSnapshot,
        actions: list[Any],
        results: list[ActionResult],
        *,
        reason: str | None = None,
    ) -> None:
        self.results.extend(results)
        entry: dict[str, Any] = {
            "snapshot_id": snapshot.snapshot_id,
            "actions": [action.model_dump(mode="json") for action in actions],
            "results": [result.model_dump(mode="json") for result in results],
        }
        if reason:
            entry["event"] = reason
        self.history.append(entry)

    # -------------------------------------------------------------------------------- recovery

    async def on_rejected(
        self,
        page: Any,
        planner: Any,
        executor: BrowserExecutor,
        snapshot: PageSnapshot,
        plan: ApplicationPlan | None,
        step: int,
        message: str,
        timings: dict[str, float],
    ) -> RunResult | None:
        self.journey.log("plan_rejected", step=step, error=message)

        reduced = _without_disabled_target(plan, message) if plan is not None else None
        if reduced is not None:
            try:
                validate_plan(
                    reduced, snapshot, self.profile, self.assets, self.config.allow_submission
                )
            except PlanValidationError as reduced_error:
                self.journey.log("plan_reduced_rejected", step=step, error=str(reduced_error))
            else:
                self.journey.log(
                    "plan_reduced",
                    step=step,
                    reason=f"Dropped the disabled control and kept the rest: {message}",
                    remaining=len(reduced.actions),
                )
                execute_started = time.perf_counter()
                results = await executor.execute(reduced.actions, snapshot, self.profile)
                timings["execute_ms"] = (time.perf_counter() - execute_started) * 1_000
                self.record(snapshot, reduced.actions, results)
                self.transient_rejections = 0
                return None

        if self.transient_rejections < MAX_TRANSIENT_REJECTIONS and _is_recoverable(message):
            self.transient_rejections += 1
            disabled = _disabled_target(message)
            waited = bool(disabled) and await _wait_for_enabled_target(
                page, snapshot.snapshot_id, disabled
            )
            self.journey.log(
                "plan_retry",
                step=step,
                error=message,
                attempt=self.transient_rejections,
                reason=(
                    "The disabled control became actionable again; re-planning against the live page."
                    if waited
                    else "Re-observing and re-planning with the rejection in history: the control "
                    "may be briefly disabled, or the chosen option may not exist in that dropdown."
                ),
            )
            self.history.append(
                {"snapshot_id": snapshot.snapshot_id, "event": "plan_rejected", "reason": message}
            )
            return None

        return RunResult(
            status="blocked",
            message=message,
            url=page.url,
            steps=step,
            evidence=None,
            action_results=self.results,
        )

    async def on_stale_plan(
        self, page: Any, snapshot: PageSnapshot, plan: ApplicationPlan, step: int
    ) -> RunResult | None:
        """A plan whose every action already matches the page is a contradiction, not a request."""
        fingerprint = snapshot_fingerprint(snapshot)
        self.stale_plans[fingerprint] += 1
        message = (
            "Ignored a stale planner response: every requested action already matches the live page. "
            "Re-observing and asking the planner for the next required field."
        )
        self.journey.log(
            "plan_stale",
            step=step,
            plan=plan,
            reason=message,
            stale_count=self.stale_plans[fingerprint],
        )
        self.history.append(
            {
                "snapshot_id": snapshot.snapshot_id,
                "event": "stale_plan_ignored",
                "reason": message,
                "actions": [action.model_dump(mode="json") for action in plan.actions],
            }
        )
        # This was a recoverable planner contradiction, not genuine page stagnation.
        self.fingerprints[fingerprint] = 0
        if self.stale_plans[fingerprint] >= STALE_PLAN_LIMIT:
            return RunResult(
                status="blocked",
                message="Planner repeatedly returned actions that were already satisfied",
                url=page.url,
                steps=step,
                evidence=None,
                action_results=self.results,
            )
        return None

    def actions_already_satisfied(self, plan: ApplicationPlan, snapshot: PageSnapshot) -> bool:
        if not plan.actions:
            return False
        return all(_action_already_satisfied(a, snapshot, self.profile) for a in plan.actions)


# ------------------------------------------------------------------------------------ pure helpers


def _is_recoverable(message: str) -> bool:
    """A disabled control is usually mid-save; a missing dropdown option just needs another choice."""
    return "is disabled" in message or "is unavailable" in message


def _disabled_target(message: str) -> str | None:
    match = DISABLED_TARGET.search(message)
    return match.group(1) if match else None


def _without_disabled_target(plan: ApplicationPlan, message: str) -> ApplicationPlan | None:
    """A single disabled control must not discard the rest of an otherwise valid batch."""
    target = _disabled_target(message)
    if target is None:
        return None
    remaining = [action for action in plan.actions if action.target != target]
    if not remaining or len(remaining) == len(plan.actions):
        return None
    return plan.model_copy(update={"actions": remaining})


async def _wait_for_enabled_target(page: Any, snapshot_id: str, target: str) -> bool:
    """Wait for a disabled control to come back, without spending a model call to discover it."""
    script = """([sid, id]) => {
        const el = document.querySelector('[data-agent-snapshot="' + sid + '"][data-agent-id="' + id + '"]');
        return !!el && !el.disabled && el.getAttribute('aria-disabled') !== 'true';
    }"""
    deadline = time.monotonic() + DISABLED_WAIT_MS / 1_000
    while time.monotonic() < deadline:
        try:
            if await page.evaluate(script, [snapshot_id, target]):
                return True
        except Exception:
            return False
        await page.wait_for_timeout(500)
    return False


def _action_already_satisfied(
    action: Any, snapshot: PageSnapshot, profile: CandidateProfile
) -> bool:
    if not action.target:
        return False
    element = next((item for item in snapshot.elements if item.id == action.target), None)
    if element is None:
        return False
    if action.type == "set_checked":
        return action.checked is not None and element.checked == action.checked
    if action.type == "select":
        return action.option_value is not None and element.value == action.option_value
    if action.type == "fill":
        try:
            expected = (
                resolve_profile_ref(profile, action.value_ref) if action.value_ref else action.value
            )
        except PlanValidationError:
            return False
        return expected is not None and element.value == str(expected)
    if action.type == "upload":
        return element.file_attached
    return False


def _zoom_script(zoom: float) -> str:
    """Scale the layout inside the viewport on every navigation, the way browser zoom does."""
    return (
        "(() => {"
        f"const factor = '{zoom}';"
        "const apply = () => {"
        "if (document.documentElement) { document.documentElement.style.zoom = factor; }"
        "};"
        "apply();"
        "document.addEventListener('DOMContentLoaded', apply);"
        "})();"
    )


async def _fit_page_zoom(page: Any) -> None:
    """Zoom out further when the page is taller than the window, so the form needs no scrolling."""
    try:
        height, viewport, current = await page.evaluate(
            "[document.documentElement.scrollHeight, window.innerHeight,"
            " parseFloat(document.documentElement.style.zoom) || 1]"
        )
    except Exception:
        return
    if not height or not viewport:
        return
    fitted = current * viewport / height
    effective = max(MIN_AUTO_ZOOM, min(current, fitted))
    if abs(effective - current) > 0.01:
        await page.evaluate("(z) => { document.documentElement.style.zoom = String(z); }", effective)


async def _wait_for_interactive_form(page: Any, timeout_ms: int) -> None:
    """Wait for a page that has actually rendered: visible form controls, or at least real content.

    A hidden field (a search box in the site header, for instance) used to satisfy this immediately,
    so the agent observed an unrendered shell and concluded the page had no form.
    """
    deadline = time.monotonic() + max(timeout_ms, 1_000) / 1_000
    probe = """() => {
        const visible = (el) => {
            const style = window.getComputedStyle(el);
            const rect = el.getBoundingClientRect();
            return style.display !== 'none' && style.visibility !== 'hidden'
                && rect.width > 0 && rect.height > 0;
        };
        const controls = [...document.querySelectorAll(
            'input:not([type="file"]), textarea, select, [role="combobox"], [role="textbox"]'
        )].filter(visible).length;
        const text = document.body ? document.body.innerText.trim().length : 0;
        return [controls, text];
    }"""
    while time.monotonic() < deadline:
        try:
            controls, text = await page.evaluate(probe)
        except Exception:
            controls, text = 0, 0
        if controls > 0 or text > 200:
            return
        await page.wait_for_timeout(250)


__all__ = ["run_application", "build_planner"]
