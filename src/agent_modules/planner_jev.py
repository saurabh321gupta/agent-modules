"""The Jev per-field planner. Layer 4.

Plans a step by asking many independent, constrained questions and composing the actions in code.

The model never writes a value: it picks a *source key* (a dotted profile path, or an `answers.*`
entry) or an *option value*, and every value that reaches the form is resolved here from the
candidate profile or the approved answer bank. That is what makes this path fast and hard to
hallucinate through - and it is also why the criteria maps in `prompts_jev` matter so much, since
they are the entire space of answers the model may choose from.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from .config import MAX_ACTIONS, MAX_CHOICES, AutomationDefaults
from .journey import JourneyLogger
from .llm_client import ModelClient
from .prompts_jev import (
    ADVANCE_MARKERS,
    APPLY_MARKERS,
    LARGE_OPTION_SYSTEM,
    REPAIR_SYSTEM,
    SUBMIT_MARKERS,
    build_field_questions,
    profile_facts,
)
from .prompts_llm import profile_for_llm
from .models import Action, ApplicationPlan, CandidateProfile, PageElement, PageSnapshot


class LlmFieldAssist:
    """Narrow model help for a single field where Jev cannot act.

    Two cases, both deliberately small: choosing one option from a list too long for a Jev Choice,
    and repairing a value the site has rejected because of its format. Neither call plans a page;
    each returns one value, which is validated here before it is used.
    """

    def __init__(self, client: ModelClient, model: str, journey: JourneyLogger | None = None) -> None:
        self.client = client
        self.model = model
        self.journey = journey

    async def _ask(self, system: str, payload: dict[str, Any], event: str) -> dict[str, Any]:
        if self.journey:
            self.journey.log(event, model=self.model, field=payload.get("field"))
        completion = await self.client.complete(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            response_format={"type": "json_object"},
            model=self.model,
        )
        try:
            data = json.loads(completion.text)
        except json.JSONDecodeError:
            data = {}
        if not isinstance(data, dict):
            data = {}
        data["_latency_s"] = round(completion.latency_s, 2)
        return data

    async def choose(
        self, element: PageElement, facts: dict[str, str], answer_bank: dict[str, str]
    ) -> str | None:
        options = [
            {"value": option.value, "label": option.label}
            for option in element.options
            if option.value != ""
        ]
        if not options:
            return None
        data = await self._ask(
            LARGE_OPTION_SYSTEM,
            {
                "field": {
                    "label": element.label,
                    "help_text": element.help_text,
                    "role": element.role,
                },
                "candidate_facts": facts,
                "approved_answers": answer_bank,
                "options": options,
            },
            "large_option_request",
        )
        value = data.get("option_value")
        valid = {option["value"] for option in options}
        if not isinstance(value, str) or value not in valid:
            if self.journey:
                self.journey.log(
                    "large_option_rejected",
                    field=element.label,
                    latency_s=data.get("_latency_s"),
                    model_returned=value,
                    reason=data.get("reason"),
                    note="The model's answer was not one of the page's option values, so it was discarded.",
                )
            return None
        if self.journey:
            self.journey.log(
                "large_option_resolved",
                field=element.label,
                option_value=value,
                latency_s=data.get("_latency_s"),
                reason=data.get("reason"),
            )
        return value

    async def repair_value(
        self,
        element: PageElement,
        error: str,
        facts: dict[str, str],
        answer_bank: dict[str, str],
        page_context: dict[str, Any],
    ) -> str | None:
        data = await self._ask(
            REPAIR_SYSTEM,
            {
                "field": {
                    "label": element.label,
                    "input_type": element.input_type,
                    "current_value": element.value,
                    "help_text": element.help_text,
                },
                "site_error": error,
                "candidate_facts": facts,
                "approved_answers": answer_bank,
                "page": page_context,
            },
            "field_repair_request",
        )
        value = data.get("value")
        if not isinstance(value, str) or not value.strip():
            if self.journey:
                self.journey.log(
                    "field_repair_failed",
                    field=element.label,
                    error=error,
                    model_returned=value,
                    reason=data.get("reason"),
                )
            return None
        if self.journey:
            self.journey.log(
                "field_repaired",
                field=element.label,
                error=error,
                replacement=value,
                was=element.value,
                latency_s=data.get("_latency_s"),
                reason=data.get("reason"),
            )
        return value


class JevPlanner:
    """Asks Jev many independent questions and composes the actions in code."""

    def __init__(
        self,
        client: Any,
        assets: dict[str, str] | None = None,
        defaults: AutomationDefaults | None = None,
        answer_bank: dict[str, str] | None = None,
        confidence_floor: float = 0.35,
        field_assist: LlmFieldAssist | None = None,
        journey: JourneyLogger | None = None,
    ) -> None:
        self.client = client
        self.assets = assets or {}
        self.defaults = defaults or AutomationDefaults()
        self.answer_bank = answer_bank or {}
        self.confidence_floor = confidence_floor
        self.field_assist = field_assist
        self.journey = journey
        # Values the site has already accepted after a repair, keyed by field label. Without this the
        # next step's ordinary question would overwrite the repair with the value the site rejected.
        self.repaired_values: dict[str, str] = {}
        self.last_step: dict[str, Any] = {}

    # ---------------------------------------------------------------------------------- request

    def state(
        self,
        profile: CandidateProfile,
        snapshot: PageSnapshot,
    ) -> dict[str, Any]:
        """Build the Jev state leanly: Jev has a far smaller input budget than a language model.

        Element options are omitted here because the same option list is already sent as the criteria
        of that element's question - sending them twice doubled the request.
        """
        elements = [
            {
                "id": element.id,
                "role": element.role,
                "label": element.label,
                "input_type": element.input_type,
                "required": element.required,
                "value": element.value,
                "enabled": element.enabled,
                "checked": element.checked,
                "help_text": element.help_text,
            }
            for element in snapshot.elements
            if element.visible and element.input_type not in ("file", "password")
        ]
        return {
            "candidate_profile": profile_for_llm(profile, self.assets, self.defaults),
            "user_provided_answers": self.answer_bank,
            "current_page": {
                "url": snapshot.url,
                "title": snapshot.title,
                "elements": elements,
                "visible_text": snapshot.visible_text[:20],
                "validation_errors": snapshot.validation_errors,
                "blockers": snapshot.blockers,
            },
        }

    async def next_step(
        self,
        profile: CandidateProfile,
        snapshot: PageSnapshot,
    ) -> ApplicationPlan:
        started = time.perf_counter()
        state = self.state(profile, snapshot)
        questions, element_by_question, large_dropdowns, invalid_elements = build_field_questions(
            profile, snapshot, self.answer_bank
        )
        response = await self.client.ask(state, questions)
        answers = response.get("answers", {})
        resolved = await self.resolve_large_dropdowns(large_dropdowns, profile)
        repairs = await self.repair_invalid_fields(invalid_elements, profile, snapshot)
        elapsed = time.perf_counter() - started

        plan = self.compose(
            profile, snapshot, answers, element_by_question, resolved, repairs, elapsed
        )
        self.last_step = {
            "class": self.kind(answers) or "form",
            "confidence": 0.0,
            "actions": len(plan.actions),
        }
        if self.journey:
            self.journey.log("plan_received", plan=plan)
        return plan

    # ------------------------------------------------------------------------------ llm fallbacks

    async def repair_invalid_fields(
        self,
        elements: list[PageElement],
        profile: CandidateProfile,
        snapshot: PageSnapshot,
    ) -> dict[str, str]:
        """Ask the model for a corrected value for each field the site has rejected."""
        if not elements:
            return {}
        if self.field_assist is None:
            if self.journey:
                self.journey.log(
                    "field_repair_skipped",
                    fields=[element.label or element.id for element in elements],
                    reason="No LLM assist is configured, so the rejected values were left as they are.",
                )
            return {}
        facts = profile_facts(profile)
        page_context = {
            "url": snapshot.url,
            "title": snapshot.title,
            "validation_errors": snapshot.validation_errors,
            "fields": [
                {"id": e.id, "label": e.label, "role": e.role, "value": e.value}
                for e in snapshot.elements
                if e.visible
            ][:40],
        }
        outcomes = await asyncio.gather(
            *[
                self.field_assist.repair_value(
                    element,
                    self.error_for(element, snapshot),
                    facts,
                    self.answer_bank,
                    page_context,
                )
                for element in elements
            ],
            return_exceptions=True,
        )
        repairs: dict[str, str] = {}
        for element, outcome in zip(elements, outcomes):
            if isinstance(outcome, str):
                repairs[element.id] = outcome
                if element.label:
                    self.repaired_values[element.label] = outcome
            elif self.journey:
                self.journey.log(
                    "field_repair_failed", field=element.label or element.id, error=str(outcome)
                )
        return repairs

    @staticmethod
    def error_for(element: PageElement, snapshot: PageSnapshot) -> str:
        label = (element.label or "").lower()
        matching = [
            error for error in snapshot.validation_errors if label and label in error.lower()
        ]
        if matching:
            return "; ".join(matching)
        return "; ".join(snapshot.validation_errors[:3]) or "the site rejected this value"

    async def resolve_large_dropdowns(
        self,
        elements: list[PageElement],
        profile: CandidateProfile,
    ) -> dict[str, str]:
        """Ask the model for the dropdowns whose option list is too long for a Jev Choice."""
        if not elements:
            return {}
        if self.field_assist is None:
            if self.journey:
                self.journey.log(
                    "large_option_skipped",
                    fields=[element.label or element.id for element in elements],
                    reason=(
                        "No LLM assist is configured, so these dropdowns were left alone. "
                        "Pass an LLM API key to enable the fallback."
                    ),
                )
            return {}
        facts = profile_facts(profile)
        outcomes = await asyncio.gather(
            *[self.field_assist.choose(element, facts, self.answer_bank) for element in elements],
            return_exceptions=True,
        )
        resolved: dict[str, str] = {}
        for element, outcome in zip(elements, outcomes):
            if isinstance(outcome, str):
                resolved[element.id] = outcome
            elif self.journey:
                self.journey.log(
                    "large_option_failed", field=element.label or element.id, error=str(outcome)
                )
        return resolved

    # ---------------------------------------------------------------------------------- compose

    @staticmethod
    def kind(answers: dict[str, Any]) -> str | None:
        answer = answers.get("page_kind")
        if isinstance(answer, dict) and answer.get("choice"):
            return str(answer["choice"])
        return None

    @staticmethod
    def choice(answers: dict[str, Any], key: str) -> tuple[str, float] | None:
        answer = answers.get(key)
        if not isinstance(answer, dict) or not answer.get("choice"):
            return None
        return str(answer["choice"]), float(answer.get("confidence") or 0)

    def compose(
        self,
        profile: CandidateProfile,
        snapshot: PageSnapshot,
        answers: dict[str, Any],
        element_by_question: dict[str, str],
        resolved_options: dict[str, str],
        repairs: dict[str, str],
        elapsed: float,
    ) -> ApplicationPlan:
        elements = {element.id: element for element in snapshot.elements}
        facts = profile_facts(profile)
        actions: list[Action] = []
        notes: list[str] = []
        skipped_required: list[str] = []

        kind = self.kind(answers)
        if kind == "captcha":
            return self.status(snapshot, "blocked", "Jev reports a CAPTCHA or human verification challenge.")
        if kind == "login":
            return self.status(snapshot, "blocked", "Jev reports a sign-in or verification wall.")

        # Uploads are deterministic: attach the asset the profile declares, if it was registered.
        # Note this scans every element, including ones the reader marked invisible - the real resume
        # input is hidden behind a styled button, and it is the element that must receive the file.
        asset = self.registered_asset(profile)
        for element in snapshot.elements:
            if element.input_type == "file" and element.enabled and not element.file_attached and asset:
                actions.append(self.action("upload", element.id, asset_id=asset))
                notes.append(f"upload {asset} -> {element.label or element.id}")

        # Dropdowns too long for a Jev Choice, already resolved by the model fallback.
        for element_id, option_value in resolved_options.items():
            element = elements.get(element_id)
            if element is None or element.value == option_value:
                continue
            actions.append(self.action("select", element.id, option_value=option_value))
            notes.append(f"select {element.label or element.id}={option_value} (llm fallback)")

        # Values the site rejected, rewritten into the format it asked for.
        for element_id, replacement in repairs.items():
            element = elements.get(element_id)
            if element is None or element.value == replacement:
                continue
            actions.append(self.action("fill", element.id, value=replacement))
            notes.append(
                f"repair {element.label or element.id}={replacement} (was {element.value!r})"
            )

        for question, element_id in element_by_question.items():
            if question == "advance":
                continue
            answer = answers.get(question) or {}
            element = elements.get(element_id)
            if element is None:
                continue
            if answer.get("type") == "noul":
                if float(answer.get("noul") or 0) >= 0.5:
                    actions.append(self.action("set_checked", element.id, checked=True))
                    notes.append(f"tick {element.label or element.id} ({answer.get('noul'):.2f})")
                continue
            choice = answer.get("choice")
            confidence = float(answer.get("confidence") or 0)
            if not choice or choice == "none" or confidence < self.confidence_floor:
                # "none" on a field that already holds a value means "leave it", not "cannot answer".
                if element.required and not (element.value or "").strip():
                    skipped_required.append(element.label or element.id)
                continue
            if element.role == "combobox":
                if element.value == choice:
                    notes.append(f"{element.label or element.id} already {choice}")
                    continue
                actions.append(self.action("select", element.id, option_value=choice))
                notes.append(f"select {element.label or element.id}={choice} ({confidence:.2f})")
                continue
            value = self.resolve(choice, facts)
            if value is None:
                if element.required:
                    skipped_required.append(element.label or element.id)
                continue
            remembered = self.repaired_values.get(element.label or "")
            if remembered is not None and remembered != value:
                notes.append(
                    f"{element.label or element.id}: keeping the site-accepted {remembered!r} "
                    f"instead of {value!r}"
                )
                value = remembered
            if (element.value or "").strip() == str(value).strip():
                notes.append(f"{element.label or element.id} already correct")
                continue
            actions.append(self.action("fill", element.id, value=value))
            notes.append(f"fill {element.label or element.id}<-{choice} ({confidence:.2f})")

        # A job advert is not a form: its fields are search, newsletter and alert furniture, never
        # candidate answers, so drop them and leave only the control that starts the application.
        on_job_listing = kind == "job_listing"
        if on_job_listing and actions:
            notes.append(f"job listing: ignored {len(actions)} page-furniture control(s)")
            actions = []

        # The advance click must be last: the executor stops a batch once it clicks.
        actions = actions[: MAX_ACTIONS - 1]

        # A page that still flags fields as invalid will not advance, whatever the model thinks.
        pending_invalid = [element for element in snapshot.elements if element.invalid]
        submit_ready = float((answers.get("submit_ready") or {}).get("noul") or 0)
        advance = self.choice(answers, "advance")
        target = elements.get(advance[0]) if advance else None
        if pending_invalid:
            notes.append(
                "not advancing: the site still flags "
                + ", ".join(element.label or element.id for element in pending_invalid[:3])
                + (f" (+{len(pending_invalid) - 3} more)" if len(pending_invalid) > 3 else "")
            )
            target = None
        if target is not None:
            label = (target.label or "").lower()
            looks_submit = any(marker in label for marker in SUBMIT_MARKERS)
            looks_apply = any(marker in label for marker in APPLY_MARKERS)
            if on_job_listing:
                if looks_apply:
                    actions.append(self.action("click", target.id))
                    notes.append(f"start application: {target.label} ({advance[1]:.2f})")
                else:
                    notes.append(
                        f"job listing: ignored '{target.label}' - it does not start an application"
                    )
            elif looks_submit and (submit_ready < 0.5 or skipped_required):
                notes.append(
                    f"held back submit on {target.label} (readiness {submit_ready:.2f}, "
                    f"ungrounded {len(skipped_required)})"
                )
            elif looks_submit or any(marker in label for marker in ADVANCE_MARKERS):
                actions.append(self.action("click", target.id))
                notes.append(f"click {target.label or target.id} ({advance[1]:.2f})")

        if kind == "confirmation":
            return self.status(snapshot, "complete", "Jev reports a confirmation page.", notes)

        if skipped_required and not actions:
            return self.status(
                snapshot,
                "needs_input",
                "Jev could not ground these required fields from the profile or the answer bank: "
                + ", ".join(skipped_required),
                notes,
            )

        reason = (
            f"Jev step in {elapsed:.2f}s. "
            + ("; ".join(notes) if notes else "Nothing to do on this page.")
            + (
                f" | required still ungrounded: {', '.join(skipped_required)}"
                if skipped_required
                else ""
            )
        )
        return ApplicationPlan(
            status="continue",
            actions=actions,
            reason=reason,
            completion_evidence=None,
        )

    # ---------------------------------------------------------------------------------- helpers

    def registered_asset(self, profile: CandidateProfile) -> str | None:
        """The asset the profile declares, but only when it was registered on the command line."""
        for document in (profile.documents or {}).values():
            if not isinstance(document, dict):
                continue
            asset_id = document.get("asset_id")
            if isinstance(asset_id, str) and asset_id in self.assets:
                return asset_id
        return None

    def resolve(self, source: str, facts: dict[str, str]) -> str | None:
        if source.startswith("answers."):
            return self.answer_bank.get(source[len("answers."):])
        return facts.get(source)

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

    @staticmethod
    def status(
        snapshot: PageSnapshot,
        status: str,
        reason: str,
        notes: list[str] | None = None,
    ) -> ApplicationPlan:
        if notes:
            reason = reason + " | " + "; ".join(notes)
        return ApplicationPlan(
            status=status,  # type: ignore[arg-type]
            actions=[],
            reason=reason,
            completion_evidence=None,
        )


__all__ = ["JevPlanner", "LlmFieldAssist", "MAX_CHOICES"]
