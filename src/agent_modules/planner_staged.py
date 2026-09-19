"""Classify the page, branch on the verdict, and only then spend an answering call. Layer 4.

The point of the staging is cost: most pages in an application are not form steps. An advert, a
review screen and a confirmation page can all be handled by a single cheap classification call and a
deterministic click, and only a real form step justifies normalising and then answering.

It exposes the same `next_step` signature as every other planner, so the orchestrator, validator,
executor and verifier are unchanged: the branching lives here, not in the engine.
"""

from __future__ import annotations

import time
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .journey import JourneyLogger
from .normalizer import Normalizer
from .planner_llm import LLMPlanner
from .prompts_jev import APPLY_MARKERS, classify_controls, classify_questions, classify_state
from .models import Action, ApplicationPlan, CandidateProfile, CompletionEvidence, PageSnapshot

PageClass = Literal[
    "job_listing",
    "job_review",
    "application_submitted",
    "application_form",
    "captcha",
    "login",
    "other",
]

PAGE_CLASSES = frozenset(PageClass.__args__)  # type: ignore[attr-defined]

#: Below this, a verdict is treated as "cannot tell" and the run falls through to the form path.
CLASS_FLOOR = 0.4
#: A false "submitted" ends a run early, so that verdict has to be more certain than the rest.
SUBMITTED_FLOOR = 0.7


class PageDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    page_class: PageClass
    confidence: float = 0.0
    probabilities: dict[str, float] = Field(default_factory=dict)
    control_id: str | None = None
    control_label: str | None = None
    control_confidence: float = 0.0
    submitted_evidence: float = 0.0
    #: The file input that should receive the resume, when the page offers one and still needs it.
    resume_control_id: str | None = None
    resume_confidence: float = 0.0


class StagedPlanner:
    def __init__(
        self,
        classifier: Any,
        normalizer: Normalizer,
        planner: LLMPlanner,
        assets: dict[str, str] | None = None,
        normalise: bool = True,
        journey: JourneyLogger | None = None,
    ) -> None:
        self.classifier = classifier
        self.normalizer = normalizer
        self.planner = planner
        #: The assets registered for this run, so the resume can be attached without a path.
        self.assets = assets or {}
        #: Off sends the raw snapshot to the planner instead of normalised questions.
        self.normalise = normalise
        self.journey = journey
        self.last_step: dict[str, Any] = {}

    async def next_step(
        self,
        profile: CandidateProfile,
        snapshot: PageSnapshot,
        history: list[dict[str, Any]],
    ) -> ApplicationPlan:
        decision = await self.classify(snapshot)

        # Checked before the page class, deliberately. An upload is a precondition rather than a kind
        # of page: a wizard step cannot be answered until the resume it is waiting for has arrived, and
        # the site usually fills fields in from it. Uploading first also means the page is re-read
        # afterwards, so whatever the site populates is answered rather than guessed at.
        upload = self.resume_branch(profile, snapshot, decision)
        if upload is not None:
            return upload

        branch = decision.page_class
        if branch in ("captcha", "login"):
            return self.plan(
                snapshot,
                "blocked",
                f"Jev classified this page as {branch.replace('_', ' ')}.",
                [],
                decision,
            )
        if branch == "application_submitted":
            return self.submitted(snapshot, decision)
        if branch in ("job_listing", "job_review"):
            return self.click_branch(snapshot, decision)
        if branch == "other" and decision.confidence >= CLASS_FLOOR:
            return self.plan(
                snapshot,
                "needs_input",
                "Jev could not identify this page as part of a job application.",
                [],
                decision,
            )
        return await self.form_branch(profile, snapshot, history, decision)

    # --------------------------------------------------------------------------------- classify

    async def classify(self, snapshot: PageSnapshot) -> PageDecision:
        controls = classify_controls(snapshot)
        buttons = [c for c in controls if c["role"] == "button" and c["enabled"]]
        file_inputs = [c for c in controls if c["input_type"] == "file" and c["enabled"]]
        questions = classify_questions(buttons, file_inputs)
        state = classify_state(snapshot, controls)
        started = time.perf_counter()
        if self.journey:
            self.journey.log(
                "classify_request",
                payload={"state": state, "questions": questions},
                button_count=len(buttons),
                file_input_count=len(file_inputs),
            )
        try:
            response = await self.classifier.ask(state, questions)
        except Exception as exc:
            if self.journey:
                self.journey.log(
                    "classify_failed",
                    error=str(exc),
                    note="Falling back to the form path, which is the behaviour without classification.",
                )
            return PageDecision(page_class="application_form")
        latency = time.perf_counter() - started
        answers = response.get("answers", {})
        decision = self.decide(answers, buttons, file_inputs)
        if self.journey:
            self.journey.log(
                "page_classified",
                latency_s=round(latency, 2),
                usage=response.get("usage"),
                decision=decision.model_dump(mode="json"),
                answers=answers,
            )
        return decision

    def decide(
        self,
        answers: dict[str, Any],
        buttons: list[dict[str, Any]],
        file_inputs: list[dict[str, Any]] | None = None,
    ) -> PageDecision:
        classification = answers.get("page_class") or {}
        page_class = classification.get("choice")
        if not isinstance(page_class, str) or page_class not in PAGE_CLASSES:
            page_class = "application_form"
        confidence = float(classification.get("confidence") or 0)
        if confidence < CLASS_FLOOR:
            page_class = "application_form"
        control = answers.get("next_control") or {}
        control_id = control.get("choice")
        known = {c["id"]: c for c in buttons}
        if not isinstance(control_id, str) or control_id not in known:
            control_id = None

        # The resume verdict is only honoured when Jev both says the page wants a resume and names a
        # control that really is one of this page's file inputs. An id it invented is discarded, same
        # discipline as the plan validator.
        resume = answers.get("resume_upload_control") or {}
        resume_choice = resume.get("choice")
        file_ids = {c["id"] for c in (file_inputs or [])}
        offered = (answers.get("has_resume_upload") or {}).get("choice") == "yes"
        resume_control_id = (
            resume_choice if offered and isinstance(resume_choice, str) and resume_choice in file_ids else None
        )

        return PageDecision(
            page_class=page_class,  # type: ignore[arg-type]
            confidence=confidence,
            probabilities=classification.get("probabilities") or {},
            control_id=control_id,
            control_label=known[control_id]["label"] if control_id else None,
            control_confidence=float(control.get("confidence") or 0),
            submitted_evidence=float((answers.get("submitted_evidence") or {}).get("noul") or 0),
            resume_control_id=resume_control_id,
            resume_confidence=float(resume.get("confidence") or 0),
        )

    # ------------------------------------------------------------------ resume upload

    def resume_branch(
        self,
        profile: CandidateProfile,
        snapshot: PageSnapshot,
        decision: PageDecision,
    ) -> ApplicationPlan | None:
        """Attach the resume when the page is waiting for one, or None to carry on as before.

        Returning a plan ends the step here. The executor performs the upload and stops the batch, and
        the loop re-observes, so the page that follows is treated as a fresh page rather than answered
        against the state that preceded the upload - which is what a site that fills fields in from
        the resume needs.
        """
        if not decision.resume_control_id:
            return None
        element = next((e for e in snapshot.elements if e.id == decision.resume_control_id), None)
        if element is None or element.input_type != "file":
            return None
        if element.file_attached:
            # Nothing to do: an upload here changes nothing, and the batch would end on a no-op.
            return None
        asset = self.registered_asset(profile)
        if asset is None:
            if self.journey:
                self.journey.log(
                    "resume_upload_skipped",
                    field=element.label or element.id,
                    reason=(
                        "Jev found a resume upload, but the profile declares no asset that this run "
                        "registered, so there is no file to attach."
                    ),
                )
            return None
        if self.journey:
            self.journey.log(
                "resume_upload_chosen",
                field=element.label or element.id,
                target=element.id,
                asset_id=asset,
                confidence=decision.resume_confidence,
                page_class=decision.page_class,
                note=(
                    "Uploading before answering: the page is checked for a resume first, and whatever "
                    "this brings back is treated as a new page."
                ),
            )
        return self.plan(
            snapshot,
            "continue",
            f"Attach the resume to {element.label or element.id} before answering this page "
            f"(Jev confidence {decision.resume_confidence:.2f}).",
            [self.action("upload", element.id, asset_id=asset)],
            decision,
        )

    def registered_asset(self, profile: CandidateProfile) -> str | None:
        """The asset the profile declares, but only when this run registered it."""
        for document in (profile.documents or {}).values():
            if not isinstance(document, dict):
                continue
            asset_id = document.get("asset_id")
            if isinstance(asset_id, str) and asset_id in self.assets:
                return asset_id
        return None

    # --------------------------------------------------------------------------------- branches

    def submitted(self, snapshot: PageSnapshot, decision: PageDecision) -> ApplicationPlan:
        certain = (
            decision.confidence >= SUBMITTED_FLOOR
            and decision.submitted_evidence >= SUBMITTED_FLOOR
        )
        if not certain:
            if self.journey:
                self.journey.log(
                    "submitted_verdict_rejected",
                    confidence=decision.confidence,
                    submitted_evidence=decision.submitted_evidence,
                    note="Not certain enough to declare the application submitted; treating it as a form.",
                )
            return self.plan(
                snapshot,
                "needs_input",
                "Page looks like a confirmation but the evidence is not strong enough to finish.",
                [],
                decision,
            )
        evidence_text = next((line for line in snapshot.visible_text if line.strip()), "")
        return self.plan(
            snapshot,
            "complete",
            "Jev reports a confirmation page.",
            [],
            decision,
            evidence=CompletionEvidence(text=evidence_text or None, url=snapshot.url, reference=None),
        )

    def click_branch(self, snapshot: PageSnapshot, decision: PageDecision) -> ApplicationPlan:
        element = next((e for e in snapshot.elements if e.id == decision.control_id), None)
        if element is None or not element.enabled:
            return self.plan(
                snapshot,
                "needs_input",
                f"Jev reported a {decision.page_class} page but named no usable control.",
                [],
                decision,
            )
        label = (element.label or "").lower()
        if decision.page_class == "job_listing" and not any(
            marker in label for marker in APPLY_MARKERS
        ):
            return self.plan(
                snapshot,
                "needs_input",
                f"Jev reported a job listing but chose '{element.label}', which does not start an application.",
                [],
                decision,
            )
        return self.plan(
            snapshot,
            "continue",
            f"{decision.page_class}: press {element.label!r} "
            f"(Jev confidence {decision.control_confidence:.2f}). "
            "No planner call was needed for this page.",
            [self.action("click", element.id)],
            decision,
        )

    async def form_branch(
        self,
        profile: CandidateProfile,
        snapshot: PageSnapshot,
        history: list[dict[str, Any]],
        decision: PageDecision,
    ) -> ApplicationPlan:
        """The only expensive branch.

        Normalised, the model answers canonical questions; raw, it answers the page itself, which
        carries the site's validation errors inline but costs a much larger payload.
        """
        form = await self.normalizer.normalise(snapshot) if self.normalise else None
        plan = await self.planner.next_step(profile, snapshot, history, form=form)
        self.last_step = {
            "class": decision.page_class,
            "confidence": decision.confidence,
            "actions": len(plan.actions),
        }
        return plan

    # ---------------------------------------------------------------------------------- helpers

    def plan(
        self,
        snapshot: PageSnapshot,
        status: str,
        reason: str,
        actions: list[Action],
        decision: PageDecision,
        evidence: CompletionEvidence | None = None,
    ) -> ApplicationPlan:
        self.last_step = {
            "class": decision.page_class,
            "confidence": decision.confidence,
            "actions": len(actions),
        }
        return ApplicationPlan(
            snapshot_id=snapshot.snapshot_id,
            status=status,  # type: ignore[arg-type]
            actions=actions,
            reason=reason,
            completion_evidence=evidence,
        )

    @staticmethod
    def action(kind: str, target: str, **extra: Any) -> Action:
        fields: dict[str, Any] = {
            "value": None,
            "value_ref": None,
            "option_value": None,
            "checked": None,
            "asset_id": None,
        }
        fields.update(extra)
        return Action(type=kind, target=target, **fields)
