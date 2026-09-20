"""Tests for `agent_modules.llm_client`.

The transport is faked, so hedging, deadlines and the schema fallback are tested as behaviour rather
than waited for. Nothing here touches a network.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from agent_modules.journey import JourneyLogger
from agent_modules.llm_client import (
    Completion,
    DeepSeekClient,
    ModelTimeoutError,
    close_client,
    extract_completion,
)


def fake_response(text: str = "{}", **usage: object):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text), finish_reason="stop")],
        usage=SimpleNamespace(**usage),
        model="deepseek-v4-flash",
    )


class FakeTransport:
    """Records every body it is sent and replies via a supplied handler."""

    def __init__(self, handler):
        self.handler = handler
        self.bodies: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **body):
        self.bodies.append(body)
        return await self.handler(body, len(self.bodies))


def client_with(handler, **kwargs) -> tuple[DeepSeekClient, FakeTransport]:
    transport = FakeTransport(handler)
    client = DeepSeekClient(
        model="deepseek-v4-flash",
        client=transport,
        journey=kwargs.pop("journey", None),
        **kwargs,
    )
    return client, transport


# ------------------------------------------------------------------------------------- basics


async def test_a_normal_call_returns_text_and_usage():
    async def handler(body, n):
        return fake_response("hi", prompt_tokens=10, completion_tokens=2)

    client, _ = client_with(handler)
    result = await client.complete(messages=[{"role": "user", "content": "x"}])
    assert isinstance(result, Completion)
    assert result.text == "hi"
    assert result.prompt_tokens == 10
    assert result.completion_tokens == 2
    assert result.hedged is False


async def test_usage_details_are_read_when_present():
    async def handler(body, n):
        return fake_response(
            "{}",
            prompt_tokens=100,
            completion_tokens=5,
            prompt_tokens_details=SimpleNamespace(cached_tokens=80),
            completion_tokens_details=SimpleNamespace(reasoning_tokens=3),
        )

    client, _ = client_with(handler)
    result = await client.complete(messages=[])
    assert result.cached_tokens == 80
    assert result.reasoning_tokens == 3


async def test_missing_usage_fields_do_not_break_extraction():
    """A provider may omit any of them; the run must still get its answer."""
    result = extract_completion(SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="x"))]))
    assert result.text == "x"
    assert result.prompt_tokens is None
    assert result.cached_tokens is None


async def test_reasoning_effort_is_sent_when_configured():
    async def handler(body, n):
        return fake_response()

    client, transport = client_with(handler, reasoning_effort="low")
    await client.complete(messages=[])
    assert transport.bodies[0]["reasoning_effort"] == "low"


async def test_reasoning_effort_is_omitted_when_unset():
    async def handler(body, n):
        return fake_response()

    client, transport = client_with(handler)
    await client.complete(messages=[])
    assert "reasoning_effort" not in transport.bodies[0]


# ------------------------------------------------------------------------------------- hedging


async def test_a_slow_call_is_hedged_and_the_faster_answer_wins():
    async def handler(body, n):
        if n == 1:
            await asyncio.sleep(5)
            return fake_response("slow")
        return fake_response("fast")

    client, transport = client_with(handler, hedge_after_s=0.05, request_timeout_s=5)
    result = await client.complete(messages=[])
    assert result.text == "fast"
    assert result.hedged is True
    assert len(transport.bodies) == 2


async def test_the_hedge_is_logged(tmp_path):
    async def handler(body, n):
        if n == 1:
            await asyncio.sleep(5)
            return fake_response("slow")
        return fake_response("fast")

    logger = JourneyLogger(str(tmp_path / "j.jsonl"))
    client, _ = client_with(handler, hedge_after_s=0.05, request_timeout_s=5, journey=logger)
    await client.complete(messages=[])
    logger.close()
    events = [json.loads(line)["event"] for line in (tmp_path / "j.jsonl").read_text().splitlines() if line]
    assert "llm_hedge" in events
    assert "SECOND REQUEST RACED" in (tmp_path / "j.log").read_text()


async def test_hedging_is_disabled_by_a_zero_threshold():
    async def handler(body, n):
        return fake_response("only")

    client, transport = client_with(handler, hedge_after_s=0)
    result = await client.complete(messages=[])
    assert result.text == "only"
    assert result.hedged is False
    assert len(transport.bodies) == 1


async def test_a_fast_call_is_never_hedged():
    async def handler(body, n):
        return fake_response("quick")

    client, transport = client_with(handler, hedge_after_s=30)
    await client.complete(messages=[])
    assert len(transport.bodies) == 1


# ------------------------------------------------------------------------------------ deadlines


async def test_a_call_that_never_returns_is_abandoned():
    """The provider SDK's read timeout is per-chunk, so only a wall-clock bound guarantees a step ends."""
    async def handler(body, n):
        await asyncio.sleep(30)
        return fake_response()

    client, _ = client_with(handler, hedge_after_s=0, request_timeout_s=0.15)
    with pytest.raises(ModelTimeoutError, match="wall-clock bound"):
        await client.complete(messages=[])


async def test_the_timeout_can_be_clamped_per_call():
    """The orchestrator pulls this down to whatever is left of the run budget."""

    async def handler(body, n):
        await asyncio.sleep(30)
        return fake_response()

    client, _ = client_with(handler, hedge_after_s=0, request_timeout_s=60)
    client.request_timeout_s = 0.15
    with pytest.raises(ModelTimeoutError):
        await client.complete(messages=[])


async def test_an_explicit_timeout_overrides_the_instance_setting():
    async def handler(body, n):
        await asyncio.sleep(30)
        return fake_response()

    client, _ = client_with(handler, hedge_after_s=0, request_timeout_s=60)
    with pytest.raises(ModelTimeoutError, match="0s wall-clock"):
        await client.complete(messages=[], timeout_s=0.1)


async def test_a_timed_out_call_leaves_nothing_running():
    """A leaked request would keep spending after the run has already given up."""
    started = asyncio.Event()

    async def handler(body, n):
        started.set()
        await asyncio.sleep(30)
        return fake_response()

    client, _ = client_with(handler, hedge_after_s=0, request_timeout_s=0.15)
    with pytest.raises(ModelTimeoutError):
        await client.complete(messages=[])
    await asyncio.sleep(0.05)
    assert started.is_set()


# ------------------------------------------------------------------------------- schema fallback


async def test_a_rejected_schema_falls_back_to_json_mode():
    """DeepSeek accepts `json_object` but refuses `json_schema` outright."""

    async def handler(body, n):
        if body["response_format"]["type"] == "json_schema":
            raise RuntimeError("400: This response_format type is unavailable now")
        return fake_response('{"ok": true}')

    client, transport = client_with(handler)
    modes: list[str] = []
    result = await client.complete_json(
        build_messages=lambda mode: modes.append(mode) or [{"role": "system", "content": mode}],
        schema={"type": "object"},
    )
    assert result.text == '{"ok": true}'
    assert modes == ["json_schema", "json_object"]
    assert transport.bodies[-1]["response_format"] == {"type": "json_object"}
    assert client.schema_mode == "json_object"


async def test_the_fallback_is_remembered_for_later_calls():
    async def handler(body, n):
        if body["response_format"]["type"] == "json_schema":
            raise RuntimeError("response_format unsupported")
        return fake_response("{}")

    client, transport = client_with(handler)
    await client.complete_json(build_messages=lambda m: [], schema={})
    await client.complete_json(build_messages=lambda m: [], schema={})
    assert all(body["response_format"]["type"] == "json_object" for body in transport.bodies[1:])


async def test_the_fallback_is_logged(tmp_path):
    async def handler(body, n):
        if n == 1:
            raise RuntimeError("response_format is unavailable")
        return fake_response("{}")

    logger = JourneyLogger(str(tmp_path / "j.jsonl"))
    client, _ = client_with(handler, journey=logger)
    await client.complete_json(build_messages=lambda m: [], schema={})
    logger.close()
    assert "STRICT JSON SCHEMA UNAVAILABLE" in (tmp_path / "j.log").read_text()


async def test_an_unrelated_error_is_not_treated_as_a_schema_rejection():
    async def handler(body, n):
        raise RuntimeError("connection reset by peer")

    client, transport = client_with(handler)
    with pytest.raises(RuntimeError, match="connection reset"):
        await client.complete_json(build_messages=lambda m: [], schema={})
    assert len(transport.bodies) == 1
    assert client.schema_mode == "json_schema"


async def test_the_strict_schema_is_sent_when_the_provider_accepts_it():
    async def handler(body, n):
        return fake_response("{}")

    client, transport = client_with(handler)
    await client.complete_json(
        build_messages=lambda m: [], schema={"type": "object"}, schema_name="application_plan"
    )
    sent = transport.bodies[0]["response_format"]
    assert sent["type"] == "json_schema"
    assert sent["json_schema"]["strict"] is True
    assert sent["json_schema"]["name"] == "application_plan"


# -------------------------------------------------------------------------------- sdk settings


def test_the_real_sdk_client_disables_its_own_retries():
    """The SDK default of 2 silently triples the worst case, on top of the hedge."""
    client = DeepSeekClient(model="m", api_key="k", base_url="https://example.invalid/v1")
    assert client._client.max_retries == 0


# ------------------------------------------------------------------------------------- thinking


async def test_thinking_can_be_switched_off_for_a_call():
    """The only control that actually stops this provider reasoning.

    Measured against a real 30 KB form payload: 24.5s with thinking, 2.4s without, same task.
    `reasoning_effort` alone barely restrained it.
    """

    async def handler(body, n):
        return fake_response()

    client, transport = client_with(handler)
    await client.complete(messages=[], thinking=False)
    assert transport.bodies[0]["extra_body"]["thinking"] == {"type": "disabled"}


async def test_thinking_can_be_switched_on_explicitly():
    async def handler(body, n):
        return fake_response()

    client, transport = client_with(handler)
    await client.complete(messages=[], thinking=True)
    assert transport.bodies[0]["extra_body"]["thinking"] == {"type": "enabled"}


async def test_thinking_is_absent_when_not_requested():
    """Absent leaves the provider's own default, which is not the same as switching it off."""

    async def handler(body, n):
        return fake_response()

    client, transport = client_with(handler)
    await client.complete(messages=[])
    assert "extra_body" not in transport.bodies[0]


async def test_thinking_reaches_the_structured_call_too():
    """The answering call goes through complete_json, where the schema fallback lives."""

    async def handler(body, n):
        return fake_response("{}")

    client, transport = client_with(handler)
    await client.complete_json(build_messages=lambda m: [], schema={}, thinking=False)
    assert transport.bodies[0]["extra_body"]["thinking"] == {"type": "disabled"}


async def test_thinking_survives_the_schema_fallback():
    """The retry must keep the setting, or a fallback would silently re-enable reasoning."""

    async def handler(body, n):
        if body["response_format"]["type"] == "json_schema":
            raise RuntimeError("response_format is unavailable")
        return fake_response("{}")

    client, transport = client_with(handler)
    await client.complete_json(build_messages=lambda m: [], schema={}, thinking=False)
    assert len(transport.bodies) == 2
    assert all(body["extra_body"]["thinking"] == {"type": "disabled"} for body in transport.bodies)


def test_every_key_the_client_sends_is_one_the_sdk_accepts():
    """A provider-specific field must travel in extra_body, or the SDK rejects the whole call.

    This was a real failure. `thinking` sent as a top-level keyword raised
    "AsyncCompletions.create() got an unexpected keyword argument 'thinking'" - and only on a live
    call, because the fakes accept anything. Checking the signature catches it offline.
    """
    import inspect

    from openai.resources.chat.completions import AsyncCompletions

    accepted = set(inspect.signature(AsyncCompletions.create).parameters) | {"self"}
    client = DeepSeekClient(model="m", api_key="k", base_url="https://example.invalid/v1")
    bodies = [
        client._body([{"role": "user", "content": "x"}], None, None),
        client._body([], {"type": "json_object"}, "m", False),
        client._body([], {"type": "json_object"}, "m", True),
    ]
    for body in bodies:
        unknown = sorted(set(body) - accepted)
        assert not unknown, f"the SDK would reject these keys: {unknown}"


async def test_closing_a_client_without_a_close_method_is_safe():
    class Bare:
        pass

    await close_client(Bare())


if __name__ == "__main__":  # pragma: no cover - a convenience, not a test path
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
