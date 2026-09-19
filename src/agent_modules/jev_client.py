"""Transport for Jev / System One. Layer 3.

A thin HTTP client and nothing more. Jev answers a Fixed set of constrained questions - a Choice over
a supplied criteria map, or a 0..1 score - so unlike the main model there is no prompt to build here;
that lives in `prompts_jev`. This module only knows how to send a question set and read the answer
back, including the one failure worth retrying.
"""

from __future__ import annotations

import asyncio
import json
import time
import urllib.error
import urllib.request
from typing import Any

DEFAULT_JEV_URL = "https://api.typesafe.ai/v1/systemone"

#: Transient statuses. Anything else is a real error and is surfaced immediately rather than retried.
RETRYABLE_STATUS = frozenset({429, 529})
MAX_ATTEMPTS = 3


class JevError(RuntimeError):
    """A Jev request that did not succeed."""


class JevClient:
    """Minimal client for the TypeSafe System One endpoint (POST /v1/systemone)."""

    def __init__(
        self,
        api_key: str,
        url: str = DEFAULT_JEV_URL,
        model: str = "jev-latest",
        timeout_s: float = 60.0,
        journey: Any | None = None,
        transport: Any | None = None,
    ) -> None:
        self.api_key = api_key
        self.url = url
        self.model = model
        #: Mutable: the orchestrator may clamp this down to the remaining run budget.
        self.timeout_s = timeout_s
        self.journey = journey
        self._transport = transport

    async def ask(self, state: Any, questions: dict[str, Any]) -> dict[str, Any]:
        body = {"state": state, "model": self.model, "questions": questions}
        if self.journey:
            self.journey.log(
                "jev_request",
                model=self.model,
                question_count=len(questions),
                state_bytes=len(json.dumps(state, ensure_ascii=False)),
                questions=sorted(questions),
            )
        started = time.monotonic()
        if self._transport is not None:
            response = await self._transport(body)
        else:
            response = await asyncio.to_thread(self._post, body)
        latency = time.monotonic() - started
        if self.journey:
            self.journey.log(
                "jev_response",
                model=response.get("model", self.model),
                latency_s=round(latency, 2),
                usage=response.get("usage"),
                answers=response.get("answers", {}),
            )
        return response

    def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
        last: Exception | None = None
        for attempt in range(MAX_ATTEMPTS):
            request = urllib.request.Request(
                self.url,
                data=payload,
                method="POST",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                last = exc
                if exc.code not in RETRYABLE_STATUS:
                    raise JevError(f"Jev request failed: {exc}") from exc
                time.sleep(0.5 * (attempt + 1))
            except Exception as exc:  # noqa: BLE001 - surfaced verbatim below
                last = exc
        raise JevError(f"Jev request failed after {MAX_ATTEMPTS} attempts: {last}")
