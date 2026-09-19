"""The single generative planner. Layer 4.

One call per page plan: profile, approved answers, the current page, and the recent history go in;
one constrained `ApplicationPlan` comes out. The staged planner uses the same answering prompt for
form pages, so the prompt lives in `prompts_llm` rather than here.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from .config import AutomationDefaults
from .journey import JourneyLogger
from .llm_client import ModelClient
from .prompts_llm import MAIN_SYSTEM_PROMPT, profile_for_llm, snapshot_for_llm, with_schema
from .models import ApplicationPlan, CandidateProfile, NormalisedForm, PageSnapshot


class LLMPlanner:
    def __init__(
        self,
        client: ModelClient,
        model: str,
        assets: dict[str, str] | None = None,
        defaults: AutomationDefaults | None = None,
        answer_bank: dict[str, str] | None = None,
        journey: JourneyLogger | None = None,
        system_prompt: str = MAIN_SYSTEM_PROMPT,
    ) -> None:
        self.client = client
        self.model = model
        self.assets = assets or {}
        self.defaults = defaults or AutomationDefaults()
        self.answer_bank = answer_bank or {}
        self.journey = journey
        self.system_prompt = system_prompt
        #: Escape hatch: set after a plan is rejected for choosing an option we did not send.
        self.include_all_options = False
        #: Reported by the orchestrator in STEP TIMING.
        self.last_step: dict[str, Any] = {}

    def payload(
        self,
        profile: CandidateProfile,
        snapshot: PageSnapshot,
        history: list[dict[str, Any]],
        form: NormalisedForm | None = None,
    ) -> dict[str, Any]:
        """The user message.

        With a normalised form the page is sent as questions rather than as raw controls, which is
        both smaller and easier to answer. Without one, the snapshot goes as it is.
        """
        base: dict[str, Any] = {
            "candidate_profile": profile_for_llm(profile, self.assets, self.defaults),
            "user_provided_answers": self.answer_bank,
            "recent_history": history[-6:],
        }
        if form is None:
            base["current_page"] = snapshot_for_llm(snapshot, self.include_all_options)
            return base
        # The normalised form is cached by *shape*, so its questions are reusable but its state is
        # not. Anything that changes between steps - the value in a field, whether the site has
        # rejected it, the page's own validation messages - has to be read from the live snapshot.
        # Taking them from the cached form instead is how a rejected phone number was re-sent
        # unchanged, step after step, while the page refused to advance.
        live = {element.id: element for element in snapshot.elements}
        base["snapshot_id"] = snapshot.snapshot_id
        base["validation_errors"] = snapshot.validation_errors
        base["form"] = {
            "page_kind": form.page_kind,
            "questions": [
                {
                    "id": field.id,
                    "question": field.question,
                    "field_type": field.field_type,
                    "required": field.required,
                    "group": field.group,
                    # Complete, from the page: a dropdown's option_value is often a code that only
                    # the page knows.
                    "options": [{"value": o.value, "label": o.label} for o in field.options],
                    "current_value": live[field.id].value if field.id in live else None,
                    # True when the site has flagged this control as holding a bad value.
                    "rejected": live[field.id].invalid if field.id in live else False,
                }
                for field in form.fields
            ],
            "submit_controls": form.submit_controls,
        }
        return base

    async def next_step(
        self,
        profile: CandidateProfile,
        snapshot: PageSnapshot,
        history: list[dict[str, Any]],
        form: NormalisedForm | None = None,
    ) -> ApplicationPlan:
        payload = self.payload(profile, snapshot, history, form)
        self.last_step = {"class": "page", "confidence": 0.0, "actions": 0}

        attempts_left = 2
        while attempts_left > 0:
            attempts_left -= 1

            def build_messages(mode: str) -> list[dict[str, Any]]:
                return [
                    {"role": "system", "content": with_schema(self.system_prompt, mode)},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ]

            if self.journey:
                self.journey.log(
                    "llm_request",
                    request={
                        "model": self.model,
                        "messages": build_messages(self.client.schema_mode),
                    },
                )

            completion = await self.client.complete_json(
                build_messages=build_messages,
                schema=ApplicationPlan.model_json_schema(),
                schema_name="application_plan",
                model=self.model,
            )
            if self.journey:
                self.journey.log(
                    "llm_response",
                    model=self.model,
                    response_text=completion.text,
                    finish_reason=completion.finish_reason,
                    latency_s=round(completion.latency_s, 2),
                    prompt_tokens=completion.prompt_tokens,
                    completion_tokens=completion.completion_tokens,
                    cached_tokens=completion.cached_tokens,
                    reasoning_tokens=completion.reasoning_tokens,
                )

            try:
                plan = ApplicationPlan.model_validate_json(completion.text)
            except ValidationError as exc:
                if attempts_left <= 0:
                    raise
                if self.journey:
                    self.journey.log("plan_malformed", error=str(exc))
                continue

            self.last_step = {"class": "page", "confidence": 0.0, "actions": len(plan.actions)}
            return plan

        raise RuntimeError("planner returned no usable plan")
