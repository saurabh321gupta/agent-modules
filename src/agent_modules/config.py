"""Run configuration and tunable constants. Layer 0: imports nothing internal.

Constants live here when more than one layer needs them — the batch limit is enforced by the
validator and respected by every planner, so it cannot live in either.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

DEFAULT_MODEL = "deepseek-v4-flash"
DEFAULT_BASE_URL = "https://api.experientiallabs.ai/v1"
DEFAULT_ZOOM = 0.6
MIN_AUTO_ZOOM = 0.3
DEFAULT_JEV_MODEL = "jev-latest"

#: One plan may not carry more actions than this. Kept small because a batch is invalidated by any
#: click, upload or scroll, so a longer batch would be built on assumptions that expire mid-run.
MAX_ACTIONS = 12

#: A dropdown offering more options than this cannot be expressed as a Jev Choice.
MAX_CHOICES = 255

HEADLESS_VIEWPORT = {"width": 1920, "height": 1080}


@dataclass(frozen=True)
class AutomationDefaults:
    """Explicit user-provided defaults for routine application questions."""

    accept_matching_consents: bool = True
    prior_employment_with_applying_company: bool = False
    citizenship_country: str = "India"
    requires_sponsorship_now: bool = False
    requires_sponsorship_future: bool = False


@dataclass
class RunConfig:
    """Everything a run needs, resolved from flags and environment.

    `request_timeout_s` and `hedge_after_s` are copied down onto the client by the orchestrator on
    every step, clamped to the remaining run budget, which is why they are mutable rather than frozen.
    """

    model: str = DEFAULT_MODEL
    base_url: str = ""
    api_key: str | None = None
    headless: bool = False
    max_steps: int = 40
    settle_timeout_ms: int = 5_000
    step_settle_timeout_ms: int = 1_000
    llm_timeout_ms: int = 90_000
    allow_submission: bool = True
    zoom: float = DEFAULT_ZOOM
    answers: dict[str, str] = field(default_factory=dict)
    answers_path: str | None = None
    planner: str = "staged"
    #: When off, form pages go to the planner as a raw snapshot instead of normalised questions.
    #: Kept as a switch because it is the difference between two very different payloads, and
    #: comparing them on a live page is the only way to judge the normaliser.
    normalise: bool = True
    jev_api_key: str | None = None
    jev_model: str = DEFAULT_JEV_MODEL
    jev_confidence_floor: float = 0.35
    defaults: AutomationDefaults = field(default_factory=AutomationDefaults)
    journey_log_path: str | None = None
    field_pause_s: float = 0.0
    #: When set, a Playwright trace is recorded here for replay with `playwright show-trace`.
    trace_path: str | None = None

    # Hardening, ported from the speed/robustness line of work.
    reasoning_effort: str | None = "low"
    request_timeout_s: float = 60.0
    hedge_after_s: float = 30.0
    max_run_seconds: float = 300.0

    def __post_init__(self) -> None:
        self.base_url = self.base_url or os.getenv("LLM_BASE_URL", DEFAULT_BASE_URL)
        self.api_key = self.api_key or os.getenv("EXPLABS_API_KEY") or os.getenv("OPENAI_API_KEY")
