"""Fakes implementing the ports, so planner tests need no network and no model.

These are deliberately thin: a queue of canned replies plus a record of what was asked. Anything
cleverer would start testing the fake.
"""

from __future__ import annotations

from typing import Any

from agent_modules.llm_client import Completion


def thinking_of(call: dict[str, Any]) -> Any:
    """The thinking setting a `ModelClient` call carries.

    This is the port's own parameter. What the SDK receives is a different shape - a provider field
    has to travel in `extra_body` - and `test_llm_client` asserts on that directly.
    """
    return call.get("thinking")


class FakeModelClient:
    """A `ModelClient` that replies from a queue. Implements both call styles."""

    def __init__(self, replies: Any = None, *, latency_s: float = 0.0, raises: Any = None) -> None:
        # `None` means "no reply queued"; an empty string is a real reply (an empty model message)
        # and must not be confused with an absent one.
        if replies is None:
            self.replies: list[Any] = []
        elif isinstance(replies, (list, tuple)):
            self.replies = list(replies)
        else:
            self.replies = [replies]
        self.latency_s = latency_s
        self.raises = list(raises) if isinstance(raises, (list, tuple)) else ([raises] if raises else [])
        self.calls: list[dict[str, Any]] = []
        self.request_timeout_s = 60.0
        self.hedge_after_s = 30.0
        self.schema_mode = "json_schema"

    def _next_reply(self) -> str:
        if self.raises:
            raise self.raises.pop(0)
        if not self.replies:
            return "{}"
        reply = self.replies.pop(0)
        # Accepting an exception here as well as via `raises` avoids the trap of an exception
        # silently becoming the response text.
        if isinstance(reply, BaseException):
            raise reply
        return reply() if callable(reply) else reply

    async def complete(
        self,
        *,
        messages: list[dict[str, Any]],
        response_format: dict[str, Any] | None = None,
        model: str | None = None,
        timeout_s: float | None = None,
        thinking: bool | None = None,
    ) -> Completion:
        self.calls.append(
            {
                "messages": messages,
                "response_format": response_format,
                "model": model,
                "thinking": thinking,
                "kind": "complete",
            }
        )
        return Completion(
            text=self._next_reply(),
            finish_reason="stop",
            model=model or "fake",
            prompt_tokens=11,
            completion_tokens=22,
            latency_s=self.latency_s,
            schema_mode=self.schema_mode,
        )

    async def complete_json(
        self,
        *,
        build_messages: Any,
        schema: dict[str, Any],
        schema_name: str = "response",
        model: str | None = None,
        timeout_s: float | None = None,
        thinking: bool | None = None,
    ) -> Completion:
        messages = build_messages(self.schema_mode)
        self.calls.append(
            {
                "messages": messages,
                "schema_name": schema_name,
                "model": model,
                "thinking": thinking,
                "kind": "complete_json",
            }
        )
        return Completion(
            text=self._next_reply(),
            finish_reason="stop",
            model=model or "fake",
            prompt_tokens=11,
            completion_tokens=22,
            latency_s=self.latency_s,
            schema_mode=self.schema_mode,
        )


class FakeClassifier:
    """A stand-in for the Jev client's `ask`."""

    def __init__(self, answers: Any = None, *, raises: Exception | None = None) -> None:
        self.answers = list(answers) if isinstance(answers, (list, tuple)) else ([answers] if answers else [])
        self.raises = raises
        self.calls: list[dict[str, Any]] = []

    async def ask(self, state: Any, questions: dict[str, Any]) -> dict[str, Any]:
        self.calls.append({"state": state, "questions": questions})
        if self.raises is not None:
            raise self.raises
        answer = self.answers.pop(0) if self.answers else {}
        return {"answers": answer, "model": "jev-latest", "usage": {"n": 1}}


def classification(page_class: str, confidence: float = 0.9, **extra: Any) -> dict[str, Any]:
    """The shape a Jev classification reply takes."""
    answers: dict[str, Any] = {
        "page_class": {"type": "choice", "choice": page_class, "confidence": confidence}
    }
    answers.update(extra)
    return answers
