"""The single generative planner. Layer 4.

One call per page: the profile, the approved answers, the page, and the prompt. That is the whole
payload. Everything that used to ride along - a normalised form, a recent history, an id the model
had to echo back - is gone, because each of those turned into a way for a run to fail rather than a
way for it to succeed.

The planner is deliberately dumb. It sees what a person would see on the page and answers from facts
it is allowed to use; it is not asked to remember anything between steps.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from .config import AutomationDefaults
from .journey import JourneyLogger
from .llm_client import ModelClient
from .models import ApplicationPlan, CandidateProfile, PageSnapshot
from .prompts_llm import MAIN_SYSTEM_PROMPT, profile_for_llm, snapshot_for_llm, with_schema


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
        thinking: bool | None = None,
    ) -> None:
        self.client = client
        self.model = model
        self.assets = assets or {}
        self.defaults = defaults or AutomationDefaults()
        self.answer_bank = answer_bank or {}
        self.journey = journey
        self.system_prompt = system_prompt
        #: None leaves the provider's own default in place. Answering a form involves real judgement
        #: - matching a fact to a question, choosing between similar options - so this is left alone
        #: unless a caller has measured that turning it off holds up.
        self.thinking = thinking
        #: Reported by the orchestrator in STEP TIMING.
        self.last_step: dict[str, Any] = {}

    def payload(self, profile: CandidateProfile, snapshot: PageSnapshot) -> dict[str, Any]:
        """The user message: who the candidate is, what they have approved, and the page itself."""
        return {
            "candidate_profile": profile_for_llm(profile, self.assets, self.defaults),
            "user_provided_answers": self.answer_bank,
            "current_page": snapshot_for_llm(snapshot),
        }

    async def next_step(
        self,
        profile: CandidateProfile,
        snapshot: PageSnapshot,
    ) -> ApplicationPlan:
        payload = self.payload(profile, snapshot)
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
                thinking=self.thinking,
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
