"""Transport for the main generative model. Layer 3.

Every hardening measure from the speed work lives here rather than in each planner, so a planner
cannot accidentally be written without them:

- **A hard deadline.** The provider SDK's read timeout is per-chunk, not wall-clock, so a call can
  and did run for 250s. A wall-clock bound is the only way to guarantee a step finishes.
- **Hedging.** Past a threshold, an identical request is raced; whichever answers first is used and
  the other is abandoned. A slow tail is a latency problem, not a correctness one, so a duplicate is
  cheaper than waiting.
- **`max_retries=0`.** The SDK retries twice by default, silently tripling the worst case.
- **A schema fallback.** A provider that rejects strict `json_schema` gets JSON mode with the schema
  stated in the prompt instead. The caller supplies `build_messages` so it can restate the schema.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Protocol, runtime_checkable


@dataclass(frozen=True)
class Completion:
    """One model reply, with the usage a run needs to account for its cost."""

    text: str
    finish_reason: str | None = None
    model: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cached_tokens: int | None = None
    reasoning_tokens: int | None = None
    latency_s: float = 0.0
    hedged: bool = False
    schema_mode: str = "json_schema"


@runtime_checkable
class ModelClient(Protocol):
    """What a planner may depend on. Tests implement this with a fake; nothing else."""

    request_timeout_s: float
    hedge_after_s: float
    schema_mode: str

    async def complete(
        self,
        *,
        messages: list[dict[str, Any]],
        response_format: dict[str, Any] | None = None,
        model: str | None = None,
        timeout_s: float | None = None,
        thinking: bool | None = None,
    ) -> Completion: ...


def extract_completion(
    response: Any,
    *,
    latency_s: float = 0.0,
    hedged: bool = False,
    schema_mode: str = "json_schema",
) -> Completion:
    """Read a provider reply defensively: a provider may omit any usage field."""
    choice = response.choices[0]
    usage = getattr(response, "usage", None)
    prompt_details = getattr(usage, "prompt_tokens_details", None)
    completion_details = getattr(usage, "completion_tokens_details", None)
    return Completion(
        text=(choice.message.content or ""),
        finish_reason=getattr(choice, "finish_reason", None),
        model=getattr(response, "model", None),
        prompt_tokens=getattr(usage, "prompt_tokens", None),
        completion_tokens=getattr(usage, "completion_tokens", None),
        cached_tokens=getattr(prompt_details, "cached_tokens", None),
        reasoning_tokens=getattr(completion_details, "reasoning_tokens", None),
        latency_s=latency_s,
        hedged=hedged,
        schema_mode=schema_mode,
    )


class ModelTimeoutError(TimeoutError):
    """Raised when a call exceeds its wall-clock bound."""


class DeepSeekClient:
    """An OpenAI-compatible client with a wall-clock bound and a raced duplicate."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str,
        request_timeout_s: float = 60.0,
        hedge_after_s: float = 30.0,
        reasoning_effort: str | None = None,
        journey: Any | None = None,
        client: Any | None = None,
        sdk_timeout_s: float | None = None,
    ) -> None:
        self.model = model
        #: Mutable: the orchestrator clamps these down to whatever is left of the run budget.
        self.request_timeout_s = request_timeout_s
        self.hedge_after_s = hedge_after_s
        self.reasoning_effort = reasoning_effort
        self.schema_mode = "json_schema"
        self.journey = journey
        if client is not None:
            self._client = client
        else:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(
                base_url=base_url,
                api_key=api_key,
                timeout=sdk_timeout_s if sdk_timeout_s is not None else request_timeout_s,
                # The SDK default of 2 silently triples the worst case, and the hedge already
                # covers a slow provider.
                max_retries=0,
            )

    # ------------------------------------------------------------------------------ transport

    async def _create(self, body: dict[str, Any]) -> Any:
        return await self._client.chat.completions.create(**body)

    def _body(
        self,
        messages: list[dict[str, Any]],
        response_format: dict[str, Any] | None,
        model: str | None,
        thinking: bool | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": model or self.model,
            "messages": messages,
        }
        if response_format is not None:
            body["response_format"] = response_format
        if self.reasoning_effort:
            body["reasoning_effort"] = self.reasoning_effort
        if thinking is not None:
            # `thinking` is provider-specific, so it has to travel in extra_body. Sent as a top-level
            # kwarg the SDK raises "create() got an unexpected keyword argument 'thinking'", and the
            # fakes cannot catch that because they accept anything - only a real call does.
            body["extra_body"] = {"thinking": {"type": "enabled" if thinking else "disabled"}}
        return body

    async def complete(
        self,
        *,
        messages: list[dict[str, Any]],
        response_format: dict[str, Any] | None = None,
        model: str | None = None,
        timeout_s: float | None = None,
        thinking: bool | None = None,
    ) -> Completion:
        return await self._race(
            self._body(messages, response_format, model, thinking), timeout_s
        )

    async def complete_json(
        self,
        *,
        build_messages: Callable[[str], list[dict[str, Any]]],
        schema: dict[str, Any],
        schema_name: str = "response",
        model: str | None = None,
        timeout_s: float | None = None,
        thinking: bool | None = None,
    ) -> Completion:
        """Ask for a strict JSON object, degrading to JSON mode if the provider refuses the schema.

        `build_messages` is called with the mode actually in use, so the caller can restate the
        schema in the prompt when the provider cannot be handed it structurally.
        """
        attempts = 0
        while True:
            attempts += 1
            mode = self.schema_mode
            if mode == "json_schema":
                response_format: dict[str, Any] = {
                    "type": "json_schema",
                    "json_schema": {"name": schema_name, "strict": True, "schema": schema},
                }
            else:
                response_format = {"type": "json_object"}
            body = self._body(build_messages(mode), response_format, model, thinking)
            try:
                return await self._race(body, timeout_s)
            except Exception as exc:
                rejected = "response_format" in str(exc)
                if mode == "json_schema" and rejected and attempts <= 2:
                    self.schema_mode = "json_object"
                    if self.journey:
                        self.journey.log(
                            "schema_fallback",
                            model=self.model,
                            error=str(exc),
                            note=(
                                "This provider rejects strict json_schema; falling back to JSON mode "
                                "with the schema stated in the prompt."
                            ),
                        )
                    continue
                raise

    # ---------------------------------------------------------------------------------- racing

    async def _race(self, body: dict[str, Any], timeout_s: float | None = None) -> Completion:
        """Race an identical request when the first is slow, and keep whichever answers first."""
        limit = self.request_timeout_s if timeout_s is None else timeout_s
        deadline = time.monotonic() + limit
        started = time.monotonic()
        running: set[asyncio.Future[Any]] = {asyncio.ensure_future(self._create(body))}
        hedge_at: float | None = (time.monotonic() + self.hedge_after_s) if self.hedge_after_s else None
        hedged = False
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise ModelTimeoutError(
                        f"model call exceeded its {limit:.0f}s wall-clock bound"
                    )
                pause = (
                    remaining
                    if hedge_at is None
                    else max(0.0, min(remaining, hedge_at - time.monotonic()))
                )
                done, running = await asyncio.wait(
                    running, timeout=pause, return_when=asyncio.FIRST_COMPLETED
                )
                if done:
                    for task in running:
                        self._abandon(task)
                    response = next(iter(done)).result()
                    return extract_completion(
                        response,
                        latency_s=time.monotonic() - started,
                        hedged=hedged,
                        schema_mode=self.schema_mode,
                    )
                if hedge_at is not None:
                    hedge_at = None
                    hedged = True
                    running.add(asyncio.ensure_future(self._create(body)))
                    if self.journey:
                        self.journey.log(
                            "llm_hedge",
                            model=self.model,
                            after_s=self.hedge_after_s,
                            note=(
                                "The first attempt was slow, so an identical request was raced; "
                                "whichever answers first is used and the other is abandoned."
                            ),
                        )
        finally:
            for task in running:
                self._abandon(task)

    @staticmethod
    def _abandon(task: asyncio.Future[Any]) -> None:
        if not task.done():
            task.cancel()
        # Retrieve the outcome so a cancelled or failed duplicate cannot surface as a warning.
        task.add_done_callback(lambda finished: finished.cancelled() or finished.exception())


async def close_client(client: Any) -> None:
    """Close the underlying SDK client if there is one. Safe to call on a fake."""
    closer = getattr(client, "close", None)
    if closer is None:
        return
    result = closer()
    if isinstance(result, Awaitable):
        await result
