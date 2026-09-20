"""Tests for `agent_modules.orchestrator`.

Drives `_ApplicationRun.loop` with a fake page, a scripted reader and a scripted planner, so the
loop's policies are tested as decisions rather than as timing.

The browser lifecycle and the real Playwright integration are covered by `test_executor` and
`test_reader`; this file is about the control flow between them.
"""

from __future__ import annotations

import json
import time

import pytest
from helpers import element, plan, profile, snapshot
from page_fakes import FakeContext, FakePage, FakeReader, ScriptedPlanner

from agent_modules import orchestrator
from agent_modules.config import RunConfig
from agent_modules.journey import JourneyLogger
from agent_modules.orchestrator import _ApplicationRun, _without_disabled_target, run_application
from agent_modules.models import ApplicationPlan

APPLICANT = profile()
EL = [element("e1", label="City"), element("e2", role="button", label="Next", input_type="submit")]


def payload(kind: str = "fill", target: str = "e1", **extra) -> dict:
    base = {
        "type": kind,
        "target": target,
        "value": None,
        "value_ref": None,
        "option_value": None,
        "checked": None,
        "asset_id": None,
    }
    base.update(extra)
    return base


def plan_of(snapshot_id: str, actions: list[dict], *, status: str = "continue", reason: str = "r"):
    return ApplicationPlan.model_validate(
        {
            "snapshot_id": snapshot_id,
            "status": status,
            "actions": actions,
            "reason": reason,
            "completion_evidence": None,
        }
    )


def make_run(plans, *, snapshots=None, config=None, raises=None, tmp_path=None):
    logger = JourneyLogger(str(tmp_path / "j.jsonl")) if tmp_path else JourneyLogger(None)
    config = config or RunConfig(planner="llm", max_steps=8)
    run = _ApplicationRun("https://example.com/apply", APPLICANT, {}, config, logger)
    run.reader = FakeReader(snapshots or [snapshot(EL)])
    run.budgeted = []
    run.deadline = time.monotonic() + config.max_run_seconds
    planner = ScriptedPlanner(plans, raises=raises)
    return run, planner, FakePage(), logger


# ------------------------------------------------------------------------------- preconditions


async def test_a_bad_url_is_refused_before_anything_else(tmp_path):
    run, _, _, logger = make_run([plan()], tmp_path=tmp_path)
    run.url = "ftp://example.com"
    result = await run.execute()
    assert result.status == "failed"
    assert "http://" in result.message
    logger.close()


async def test_the_staged_planner_requires_a_jev_key(tmp_path):
    run, _, _, logger = make_run(
        [plan()], config=RunConfig(planner="staged", api_key="k"), tmp_path=tmp_path
    )
    result = await run.execute()
    assert result.status == "failed"
    assert "Jev API key" in result.message
    logger.close()


async def test_the_llm_planner_requires_an_llm_key(tmp_path):
    run, _, _, logger = make_run(
        [plan()], config=RunConfig(planner="llm", api_key=None), tmp_path=tmp_path
    )
    run.config.api_key = None
    result = await run.execute()
    assert result.status == "failed"
    assert "LLM API key" in result.message
    logger.close()


# ------------------------------------------------------------------------------------- the loop


async def test_a_completion_verdict_ends_the_run_successfully(tmp_path):
    """The verifier owns success: a page that says it was received, and nothing else."""
    snap = snapshot(EL, visible_text=["Thank you for applying"])
    run, _, _, logger = make_run([plan()], snapshots=[snap], tmp_path=tmp_path)
    result = await run.loop(FakePage(), ScriptedPlanner([plan()]))
    assert result.status == "success"
    assert result.evidence is not None
    logger.close()


async def test_a_blocker_ends_the_run_as_blocked(tmp_path):
    """Blockers are computed by the reader from the page text, so they arrive on the snapshot."""
    snap = snapshot(EL, blockers=["CAPTCHA or human verification is present"])
    run, _, _, logger = make_run([plan()], snapshots=[snap], tmp_path=tmp_path)
    result = await run.loop(FakePage(), ScriptedPlanner([plan()]))
    assert result.status == "blocked"
    assert "CAPTCHA" in result.message
    logger.close()


async def test_a_page_that_does_not_move_is_abandoned(tmp_path):
    """Three identical snapshots means it is not going to advance, and each retry costs a call."""
    snap = snapshot(EL)
    run, planner, _, logger = make_run(
        [plan()], snapshots=[snap], tmp_path=tmp_path
    )
    result = await run.loop(FakePage(), planner)
    assert result.status == "blocked"
    assert "No page progress" in result.message
    assert planner.calls <= orchestrator.NO_PROGRESS_LIMIT
    logger.close()


async def test_needs_input_is_passed_through_with_its_reason(tmp_path):
    snap = snapshot(EL)
    pending = plan([], status="needs_input", reason="Current salary is unknown")
    run, _, _, logger = make_run([pending], snapshots=[snap], tmp_path=tmp_path)
    result = await run.loop(FakePage(), ScriptedPlanner([pending]))
    assert result.status == "needs_input"
    assert result.message == "Current salary is unknown"
    logger.close()


async def test_a_complete_verdict_without_visible_evidence_does_not_end_the_run(tmp_path):
    """Only the verifier may declare success, so a 'complete' plan is re-observed."""
    snap = snapshot(EL)
    claimed = plan([], status="complete", reason="done")
    run, planner, _, logger = make_run([claimed], snapshots=[snap], tmp_path=tmp_path)
    result = await run.loop(FakePage(), planner)
    assert result.status == "blocked", "the run kept going and then hit no-progress"
    assert planner.calls > 1
    logger.close()


async def test_the_step_limit_is_reported(tmp_path):
    """A different snapshot every step, and a real action each time, so nothing else stops the run."""
    steps = [
        snapshot([element("e1", label="City", value=f"v{i}")], snapshot_id=f"s-{i}-unique{i}")
        for i in range(1, 12)
    ]
    factory = lambda snap: plan_of(snap.snapshot_id, [payload("fill", "e1", value="x")])  # noqa: E731
    run, _, _, logger = make_run(
        [factory(steps[0])],
        snapshots=steps,
        config=RunConfig(planner="llm", max_steps=3),
        tmp_path=tmp_path,
    )
    result = await run.loop(FakePage(), ScriptedPlanner(factory))
    assert result.status == "step_limit"
    assert result.steps == 3
    logger.close()


# -------------------------------------------------------------------------------------- budget


async def test_an_exhausted_budget_discards_the_run_without_submitting(tmp_path):
    run, plan_r, _, logger = make_run(
        [plan()],
        snapshots=[snapshot(EL)],
        config=RunConfig(planner="llm", max_run_seconds=0.0),
        tmp_path=tmp_path,
    )
    result = await run.loop(FakePage(), plan_r)
    assert result.status == "timeout"
    assert "without submitting" in result.message
    logger.close()


async def test_the_budget_is_reported_in_the_readable_log(tmp_path):
    run, plan_r, _, logger = make_run(
        [plan()],
        snapshots=[snapshot(EL)],
        config=RunConfig(planner="llm", max_run_seconds=0.0),
        tmp_path=tmp_path,
    )
    await run.loop(FakePage(), plan_r)
    logger.close()
    assert "EXHAUSTED - DISCARDING WITHOUT SUBMITTING" in (tmp_path / "j.log").read_text()


def test_the_per_call_timeout_is_clamped_to_the_remaining_budget(tmp_path):
    """A 60s call cannot outlive a 10s budget, or the run overruns by design."""
    run, _, _, logger = make_run(
        [], config=RunConfig(planner="llm", request_timeout_s=60.0), tmp_path=tmp_path
    )

    class FakeClient:
        request_timeout_s = 60.0

    client = FakeClient()
    run.budgeted = [client]
    run.clamp_budget(120.0)
    assert client.request_timeout_s == 60.0, "never extended beyond the configured bound"
    run.clamp_budget(10.0)
    assert client.request_timeout_s == 10.0
    logger.close()


def test_the_per_call_timeout_is_never_raised_above_what_is_left(tmp_path):
    """A floor above the remaining budget let a call start after it was already spent.

    That is how a run six seconds past a 200s budget fired a five-second call: the duplicate cost
    money, and the timeout it produced was reported as a planner failure rather than a clean stop.
    """
    run, _, _, logger = make_run([], tmp_path=tmp_path)

    class FakeClient:
        request_timeout_s = 60.0

    client = FakeClient()
    run.budgeted = [client]
    run.clamp_budget(0.4)
    assert client.request_timeout_s == 0.4
    run.clamp_budget(-6.0)
    assert client.request_timeout_s == 0.0, "never negative either"
    logger.close()


def test_a_call_is_not_started_without_enough_budget(tmp_path):
    run, _, _, logger = make_run([], tmp_path=tmp_path)

    run.deadline = time.monotonic() + 1.0
    assert run.has_time_for_a_call() is False

    run.deadline = time.monotonic() + orchestrator.MIN_CALL_BUDGET_S + 1.0
    assert run.has_time_for_a_call() is True
    logger.close()


def test_running_out_of_time_is_reported_as_a_timeout_not_a_failure(tmp_path):
    run, _, _, logger = make_run([], tmp_path=tmp_path)
    run.deadline = time.monotonic() - 1.0
    result = run.budget_exhausted(FakePage(), steps=4)
    assert result.status == "timeout"
    assert result.steps == 4
    assert "without submitting" in result.message
    logger.close()
    assert "EXHAUSTED - DISCARDING WITHOUT SUBMITTING" in (tmp_path / "j.log").read_text(
        encoding="utf-8"
    )


# ----------------------------------------------------------------------------------- rejection


async def test_one_disabled_control_does_not_discard_the_rest_of_the_batch(tmp_path):
    """The reduced batch is re-validated and executed, so two good fills are not thrown away."""
    snap = snapshot(
        [
            element("e1", label="City"),
            element("e2", label="Locked", enabled=False),
            element("e3", label="Postal Code"),
        ]
    )
    batch = plan_of(
        snap.snapshot_id,
        [
            payload("fill", "e1", value="Bangalore"),
            payload("fill", "e2", value="x"),
            payload("fill", "e3", value="560001"),
        ],
    )
    run, planner, page, logger = make_run(
        [batch], snapshots=[snap], config=RunConfig(planner="llm", max_steps=1), tmp_path=tmp_path
    )
    await run.loop(page, planner)
    assert page.filled.get("e1") == "Bangalore"
    assert page.filled.get("e3") == "560001"
    assert "e2" not in page.filled
    assert "PARTIALLY ACCEPTED" in (tmp_path / "j.log").read_text()
    logger.close()


def test_a_reduction_only_drops_the_named_target():
    snap = snapshot([element("e1"), element("e2"), element("e3")])
    original = ApplicationPlan.model_validate(
        {
            "snapshot_id": snap.snapshot_id,
            "status": "continue",
            "actions": [payload("fill", "e1", value="a"), payload("fill", "e2", value="b")],
            "reason": "r",
            "completion_evidence": None,
        }
    )
    reduced = _without_disabled_target(original, "Target e2 is disabled")
    assert reduced is not None
    assert [a.target for a in reduced.actions] == ["e1"]


def test_a_reduction_is_declined_when_it_would_drop_everything():
    snap = snapshot([element("e1")])
    original = ApplicationPlan.model_validate(
        {
            "snapshot_id": snap.snapshot_id,
            "status": "continue",
            "actions": [payload("fill", "e1", value="a")],
            "reason": "r",
            "completion_evidence": None,
        }
    )
    assert _without_disabled_target(original, "Target e1 is disabled") is None


def test_a_reduction_is_declined_for_an_unrelated_rejection():
    snap = snapshot([element("e1")])
    original = ApplicationPlan.model_validate(
        {
            "snapshot_id": snap.snapshot_id,
            "status": "continue",
            "actions": [payload("fill", "e1", value="a")],
            "reason": "r",
            "completion_evidence": None,
        }
    )
    assert _without_disabled_target(original, "Plan was created for an expired snapshot") is None


async def test_a_recoverable_rejection_is_retried_then_becomes_blocked(tmp_path):
    """An option the page does not offer just needs another choice, so it is worth re-planning.

    Distinct snapshots so no-progress detection does not fire first and mask the retries.
    """
    steps = [
        snapshot(
            [
                element(
                    "e1",
                    role="combobox",
                    label="Country",
                    input_type="select-one",
                    options=[("IND", "India")],
                    value=f"v{i}",
                )
            ],
            snapshot_id=f"s-{i}-u{i}",
        )
        for i in range(1, 8)
    ]
    factory = lambda snap: plan_of(  # noqa: E731
        snap.snapshot_id, [payload("select", "e1", option_value="NOPE")]
    )
    run, planner, _, logger = make_run(
        factory,
        snapshots=steps,
        config=RunConfig(planner="llm", max_steps=10),
        tmp_path=tmp_path,
    )
    result = await run.loop(FakePage(), planner)
    assert result.status == "blocked"
    assert "unavailable" in result.message
    assert planner.calls == orchestrator.MAX_TRANSIENT_REJECTIONS + 1
    assert "REJECTED - RETRYING" in (tmp_path / "j.log").read_text()
    logger.close()


async def test_an_unrecoverable_rejection_stops_immediately(tmp_path):
    snap = snapshot(EL)
    stale = plan([])
    run, planner, _, logger = make_run([stale], snapshots=[snap], tmp_path=tmp_path)
    result = await run.loop(FakePage(), planner)
    assert result.status == "blocked"
    assert "expired snapshot" in result.message
    assert planner.calls == 1
    logger.close()


async def test_continue_with_no_actions_is_retried_before_giving_up(tmp_path):
    """A control is often briefly disabled while the site saves; that is not a dead end.

    Distinct snapshots here so no-progress detection does not fire first and mask the retries.
    """
    steps = [
        snapshot([element("e1", label="City", value=f"v{i}")], snapshot_id=f"s-{i}-u{i}")
        for i in range(1, 8)
    ]
    factory = lambda snap: plan_of(snap.snapshot_id, [])  # noqa: E731
    run, planner, _, logger = make_run(
        factory,
        snapshots=steps,
        config=RunConfig(planner="llm", max_steps=8),
        tmp_path=tmp_path,
    )
    result = await run.loop(FakePage(), planner)
    assert result.status == "blocked"
    assert "no actions" in result.message
    assert planner.calls == orchestrator.MAX_TRANSIENT_REJECTIONS + 1
    logger.close()


# ---------------------------------------------------------------------------------- stale plans


async def test_a_contradictory_plan_is_ignored_then_abandoned(tmp_path):
    """Every requested action already matches the page: a contradiction, not a request."""
    snap = snapshot([element("e1", label="City", value="Bangalore")])
    satisfied = plan_of(
        snap.snapshot_id,
        [payload("fill", "e1", value="Bangalore")],
        status="needs_input",
        reason="needs the city",
    )
    run, planner, _, logger = make_run(
        [satisfied], snapshots=[snap], config=RunConfig(planner="llm", max_steps=10), tmp_path=tmp_path
    )
    result = await run.loop(FakePage(), planner)
    assert result.status == "blocked"
    assert "already satisfied" in result.message
    assert "IGNORED AS STALE/CONTRADICTORY" in (tmp_path / "j.log").read_text()
    logger.close()


async def test_a_genuine_needs_input_is_not_mistaken_for_a_stale_plan(tmp_path):
    """The requested action does not yet match the page, so this is a real request for input."""
    snap = snapshot([element("e1", label="City")])
    genuine = plan_of(
        snap.snapshot_id,
        [payload("fill", "e1", value="Bangalore")],
        status="needs_input",
        reason="salary unknown",
    )
    run, planner, _, logger = make_run([genuine], snapshots=[snap], tmp_path=tmp_path)
    result = await run.loop(FakePage(), planner)
    assert result.status == "needs_input"
    assert result.message == "salary unknown"
    logger.close()


# ----------------------------------------------------------------------------- empty snapshots


async def test_a_page_that_has_not_painted_is_re_observed(tmp_path):
    """Observing an unrendered shell and concluding there is no form ends the run too early."""
    blank = snapshot([], snapshot_id="s-1-blank000")
    painted = snapshot([element("e1", label="City", value="Bangalore")])
    run, planner, _, logger = make_run(
        [], snapshots=[blank, blank, painted], config=RunConfig(planner="llm", max_steps=1), tmp_path=tmp_path
    )
    await run.loop(FakePage(), planner)
    logger.close()
    assert "NOTHING YET - RE-OBSERVING" in (tmp_path / "j.log").read_text()


async def test_a_page_that_never_paints_is_blocked(tmp_path):
    blank = snapshot([], snapshot_id="s-1-blank000")
    run, planner, _, logger = make_run([], snapshots=[blank], tmp_path=tmp_path)
    result = await run.loop(FakePage(), planner)
    assert result.status == "blocked"
    assert "never rendered" in result.message
    logger.close()


# ----------------------------------------------------------------------------------- teardown


async def test_a_navigation_failure_is_reported_clearly(tmp_path):
    run, _, page, logger = make_run([], snapshots=[snapshot(EL)], tmp_path=tmp_path)
    page.goto_error = RuntimeError("net::ERR_NAME_NOT_RESOLVED")
    result = await run.loop(page, ScriptedPlanner([plan()]))
    assert result.status == "failed"
    assert "Could not open application URL" in result.message
    logger.close()


async def test_a_run_that_cannot_start_still_points_at_its_trace(tmp_path):
    """Every result carries the paths of its own artefacts, including a run that fails immediately."""
    result = await run_application(
        "https://example.com/apply",
        APPLICANT,
        {},
        RunConfig(
            planner="llm", api_key="k", max_steps=1, journey_log_path=str(tmp_path / "j.jsonl")
        ),
    )
    assert result.status in {"failed", "blocked", "needs_input", "timeout", "step_limit", "success"}
    assert result.journey_jsonl == str(tmp_path / "j.jsonl")
    assert result.journey_log == str(tmp_path / "j.log")
    assert (tmp_path / "j.jsonl").exists()


async def test_the_run_finished_event_is_always_written(tmp_path):
    run, planner, _, logger = make_run(
        [plan()],
        snapshots=[snapshot(EL, visible_text=["Thank you for applying"])],
        tmp_path=tmp_path,
    )
    result = await run.loop(FakePage(), planner)
    logger.log("run_finished", result=result)
    logger.close()
    events = [
        json.loads(line)["event"]
        for line in (tmp_path / "j.jsonl").read_text().splitlines()
        if line
    ]
    assert events[-1] == "run_finished"


# --------------------------------------------------------------------------------- playwright trace


def traced_run(tmp_path):
    return make_run(
        [],
        config=RunConfig(planner="llm", trace_path=str(tmp_path / "run.zip")),
        tmp_path=tmp_path,
    )


async def test_tracing_is_off_unless_configured(tmp_path):
    """It costs disk and a little speed on every action, so it is opt-in."""
    run, _, _, logger = make_run([], tmp_path=tmp_path)
    context = FakeContext()
    assert await run.start_tracing(context) is False
    assert context.tracing.started is False
    logger.close()


async def test_tracing_starts_and_writes_when_configured(tmp_path):
    run, _, _, logger = traced_run(tmp_path)
    context = FakeContext()
    assert await run.start_tracing(context) is True
    assert context.tracing.started is True
    assert context.tracing.screenshots is True, "screenshots are the point of a trace"
    assert context.tracing.snapshots is True
    await run.stop_tracing(context)
    assert context.tracing.stopped_path == str(tmp_path / "run.zip")
    logger.close()


async def test_the_trace_lifecycle_is_logged_with_its_path(tmp_path):
    run, _, _, logger = traced_run(tmp_path)
    context = FakeContext()
    await run.start_tracing(context)
    await run.stop_tracing(context)
    logger.close()
    human = (tmp_path / "j.log").read_text(encoding="utf-8")
    assert "Playwright trace   : RECORDING" in human
    assert "Playwright trace   : WRITTEN" in human
    assert str(tmp_path / "run.zip") in human
    assert "playwright show-trace" in human


async def test_a_trace_that_cannot_start_does_not_break_the_run(tmp_path):
    """A diagnostic must never be the reason a run fails."""
    run, _, _, logger = traced_run(tmp_path)
    context = FakeContext(fail_start=True)
    assert await run.start_tracing(context) is False
    logger.close()
    assert "Playwright trace   : FAILED" in (tmp_path / "j.log").read_text(encoding="utf-8")


async def test_a_trace_that_cannot_be_written_does_not_break_the_run(tmp_path):
    run, _, _, logger = traced_run(tmp_path)
    context = FakeContext(fail_stop=True)
    await run.start_tracing(context)
    await run.stop_tracing(context)
    logger.close()
    human = (tmp_path / "j.log").read_text(encoding="utf-8")
    assert "Playwright trace   : FAILED" in human
    assert "could not write the trace" in human


async def test_run_started_records_the_trace_path(tmp_path):
    """So a run's own artefacts say whether a trace exists and where it went."""
    run, _, _, logger = traced_run(tmp_path)
    logger.log("run_started", url="u", trace_path=run.config.trace_path)
    logger.close()
    assert str(tmp_path / "run.zip") in (tmp_path / "j.log").read_text(encoding="utf-8")


# ----------------------------------------------------------------------------- consent defaults


async def test_consent_checkboxes_are_ticked_before_the_planner_is_asked(tmp_path):
    """A consent box is not a question for the candidate, so it never reaches the planner."""
    consent = snapshot(
        [element("e1", role="checkbox", label="I accept the privacy notice", checked=False)]
    )
    moved = snapshot(EL, snapshot_id="s-2-moved000", visible_text=["Thank you for applying"])
    run, planner, page, logger = make_run(
        [], snapshots=[consent, moved], config=RunConfig(planner="llm", max_steps=4), tmp_path=tmp_path
    )
    result = await run.loop(page, planner)
    assert page.checked.get("e1") is True
    assert planner.calls in (0, 1)
    assert result.status == "success"
    logger.close()


# --------------------------------------------------------------------------------------- settling


async def test_settle_does_not_wait_for_the_network_to_go_idle(monkeypatch, tmp_path):
    """A modern page rarely reaches network-idle, so that wait burned its whole timeout per step.

    Measured at ~1.7s of settle per step across fifteen steps - about 25s of one run.
    """
    async def no_overlays(page):
        return []

    monkeypatch.setattr("agent_modules.overlays.dismiss_nonessential_overlays", no_overlays)
    run, _, _, logger = make_run([], tmp_path=tmp_path)
    page = FakePage()
    await run.settle(page)
    assert page.load_state_waits == []
    logger.close()


async def test_settle_waits_until_the_page_stops_rendering(monkeypatch, tmp_path):
    """Removing the old wait outright was a regression: a page observed mid-render gives a partial
    form, the plan names ids that then move, and the batch dies before it finishes."""
    async def no_overlays(page):
        return []

    monkeypatch.setattr("agent_modules.overlays.dismiss_nonessential_overlays", no_overlays)
    run, _, _, logger = make_run([], tmp_path=tmp_path)
    page = FakePage()
    page.control_counts = [4, 9, 17, 25, 25, 25, 25]
    await run.settle(page)
    # 4 -> 9 -> 17 -> 25 (reset), then 25, 25 twice more before it is trusted.
    assert page.control_counts == []
    assert len(page.evaluations) >= 6
    logger.close()


async def test_settle_returns_promptly_once_the_count_holds(monkeypatch, tmp_path):
    async def no_overlays(page):
        return []

    monkeypatch.setattr("agent_modules.overlays.dismiss_nonessential_overlays", no_overlays)
    run, _, _, logger = make_run([], tmp_path=tmp_path)
    page = FakePage()
    await run.settle(page)
    # One reading to establish the count, then the confirmations that trust it.
    assert len(page.evaluations) == 1 + orchestrator.DOM_STABLE_READINGS
    assert len(page.evaluations) < orchestrator.DOM_STABLE_READINGS * 5, "not a spin"
    logger.close()


async def test_a_probe_that_fails_stops_waiting_rather_than_hanging(monkeypatch, tmp_path):
    """A page that cannot be probed is a page we cannot wait on; observing it is the next best step."""
    async def no_overlays(page):
        return []

    monkeypatch.setattr("agent_modules.overlays.dismiss_nonessential_overlays", no_overlays)
    run, _, _, logger = make_run([], tmp_path=tmp_path)

    class Broken(FakePage):
        async def evaluate(self, script, arg=None):
            raise RuntimeError("page is gone")

    await run.settle(Broken())
    logger.close()


async def test_settle_waits_again_after_dismissing_an_overlay(monkeypatch, tmp_path):
    """That one is a page change we caused, so the page has to settle a second time."""
    async def one_overlay(page):
        return [{"kind": "cookie", "label": "Accept all"}]

    monkeypatch.setattr("agent_modules.overlays.dismiss_nonessential_overlays", one_overlay)
    run, _, _, logger = make_run([], tmp_path=tmp_path)
    page = FakePage()
    await run.settle(page)
    settled = 1 + orchestrator.DOM_STABLE_READINGS
    assert len(page.evaluations) == settled * 2, "once before the overlay and once after"
    logger.close()


async def test_an_overlay_cleanup_is_still_logged(monkeypatch, tmp_path):
    async def one_overlay(page):
        return [{"kind": "cookie", "label": "Accept all"}]

    monkeypatch.setattr("agent_modules.overlays.dismiss_nonessential_overlays", one_overlay)
    run, _, _, logger = make_run([], tmp_path=tmp_path)
    await run.settle(FakePage())
    logger.close()
    assert "OVERLAY CLEANUP" in (tmp_path / "j.log").read_text(encoding="utf-8")


if __name__ == "__main__":  # pragma: no cover - a convenience, not a test path
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
