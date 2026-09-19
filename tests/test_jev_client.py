"""Tests for `agent_modules.jev_client`. No network: the transport or urllib is faked."""

from __future__ import annotations

import json
import urllib.error

import pytest

from agent_modules import jev_client
from agent_modules.jev_client import DEFAULT_JEV_URL, JevClient, JevError
from agent_modules.journey import JourneyLogger


class FakeHttpResponse:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return self._payload

    def __enter__(self) -> "FakeHttpResponse":
        return self

    def __exit__(self, *exc: object) -> bool:
        return False


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    """The retry backoff is real in production and pointless in a test."""
    monkeypatch.setattr(jev_client.time, "sleep", lambda _seconds: None)


# ----------------------------------------------------------------------------------- request shape


async def test_ask_sends_state_model_and_questions():
    seen: list[dict] = []

    async def transport(body):
        seen.append(body)
        return {"answers": {"a": {"type": "choice", "choice": "x"}}, "model": "jev-latest", "usage": {}}

    client = JevClient("key", model="jev-latest", transport=transport)
    response = await client.ask({"url": "https://x"}, {"q": {"type": "noul"}})
    assert seen[0]["model"] == "jev-latest"
    assert seen[0]["state"] == {"url": "https://x"}
    assert seen[0]["questions"] == {"q": {"type": "noul"}}
    assert response["answers"]["a"]["choice"] == "x"


async def test_the_response_is_passed_back_untouched():
    payload = {"answers": {}, "usage": {"total_tokens": 5}, "model": "m"}

    async def transport(body):
        return payload

    assert await JevClient("k", transport=transport).ask({}, {}) is payload


async def test_the_default_endpoint_is_the_system_one_api():
    assert DEFAULT_JEV_URL == "https://api.typesafe.ai/v1/systemone"


async def test_the_timeout_is_mutable_for_budget_clamping():
    """The orchestrator pulls this down to whatever is left of the run budget."""
    client = JevClient("k", timeout_s=60.0)
    client.timeout_s = 8.0
    assert client.timeout_s == 8.0


# ---------------------------------------------------------------------------------------- logging


async def test_request_and_response_are_logged(tmp_path):
    async def transport(body):
        return {"answers": {"a": {"type": "noul", "noul": 0.9}}, "model": "jev-latest", "usage": {"n": 1}}

    logger = JourneyLogger(str(tmp_path / "j.jsonl"))
    client = JevClient("k", transport=transport, journey=logger)
    await client.ask({"url": "https://x"}, {"a": {"type": "noul"}})
    logger.close()

    events = [json.loads(line)["event"] for line in (tmp_path / "j.jsonl").read_text().splitlines() if line]
    assert events == ["jev_request", "jev_response"]
    human = (tmp_path / "j.log").read_text(encoding="utf-8")
    assert "Questions asked:" in human
    assert "a: 0.9" in human


async def test_the_logged_state_size_is_the_payload_size(tmp_path):
    async def transport(body):
        return {"answers": {}}

    logger = JourneyLogger(str(tmp_path / "j.jsonl"))
    client = JevClient("k", transport=transport, journey=logger)
    state = {"url": "https://x", "n": 1}
    await client.ask(state, {})
    logger.close()
    record = json.loads((tmp_path / "j.jsonl").read_text().splitlines()[0])
    assert record["data"]["state_bytes"] == len(json.dumps(state, ensure_ascii=False))


async def test_the_api_key_is_never_logged(tmp_path):
    async def transport(body):
        return {"answers": {}}

    logger = JourneyLogger(str(tmp_path / "j.jsonl"), secret_values=["super-secret-key"])
    client = JevClient("super-secret-key", transport=transport, journey=logger)
    await client.ask({}, {})
    logger.close()
    assert "super-secret-key" not in (tmp_path / "j.jsonl").read_text(encoding="utf-8")


# ----------------------------------------------------------------------------------------- retries


def test_a_transient_status_is_retried_then_succeeds(monkeypatch):
    attempts: list[int] = []

    def fake_urlopen(request, timeout=None):
        attempts.append(1)
        if len(attempts) == 1:
            raise urllib.error.HTTPError("u", 429, "too many requests", {}, None)
        return FakeHttpResponse(b'{"answers": {}}')

    monkeypatch.setattr(jev_client.urllib.request, "urlopen", fake_urlopen)
    assert JevClient("k")._post({"state": {}, "model": "m", "questions": {}}) == {"answers": {}}
    assert len(attempts) == 2


def test_the_overloaded_status_is_retried(monkeypatch):
    attempts: list[int] = []

    def fake_urlopen(request, timeout=None):
        attempts.append(1)
        if len(attempts) == 1:
            raise urllib.error.HTTPError("u", 529, "overloaded", {}, None)
        return FakeHttpResponse(b"{}")

    monkeypatch.setattr(jev_client.urllib.request, "urlopen", fake_urlopen)
    JevClient("k")._post({})
    assert len(attempts) == 2


def test_a_real_error_is_surfaced_immediately(monkeypatch):
    """A 400 means the question was malformed; retrying it wastes the run budget."""
    attempts: list[int] = []

    def fake_urlopen(request, timeout=None):
        attempts.append(1)
        raise urllib.error.HTTPError("u", 400, "too many choices", {}, None)

    monkeypatch.setattr(jev_client.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(JevError, match="Jev request failed"):
        JevClient("k")._post({})
    assert len(attempts) == 1


def test_a_persistent_transient_status_gives_up_after_the_attempt_limit(monkeypatch):
    attempts: list[int] = []

    def fake_urlopen(request, timeout=None):
        attempts.append(1)
        raise urllib.error.HTTPError("u", 429, "slow down", {}, None)

    monkeypatch.setattr(jev_client.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(JevError, match="after 3 attempts"):
        JevClient("k")._post({})
    assert len(attempts) == 3


def test_a_connection_error_is_retried_and_then_reported(monkeypatch):
    attempts: list[int] = []

    def fake_urlopen(request, timeout=None):
        attempts.append(1)
        raise OSError("connection reset")

    monkeypatch.setattr(jev_client.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(JevError):
        JevClient("k")._post({})
    assert len(attempts) == 3


def test_the_authorization_header_carries_the_bearer_token(monkeypatch):
    captured: list[object] = []

    def fake_urlopen(request, timeout=None):
        captured.append(request)
        return FakeHttpResponse(b"{}")

    monkeypatch.setattr(jev_client.urllib.request, "urlopen", fake_urlopen)
    JevClient("my-key")._post({"a": 1})
    request = captured[0]
    assert request.get_header("Authorization") == "Bearer my-key"
    assert request.get_header("Content-type") == "application/json"


def test_a_successful_post_parses_json(monkeypatch):
    def fake_urlopen(request, timeout=None):
        return FakeHttpResponse(b'{"answers": {"q": {"type": "noul", "noul": 1}}}')

    monkeypatch.setattr(jev_client.urllib.request, "urlopen", fake_urlopen)
    assert JevClient("k")._post({})["answers"]["q"]["noul"] == 1


if __name__ == "__main__":  # pragma: no cover - a convenience, not a test path
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
