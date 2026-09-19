"""Tests for `agent_modules.config`."""

from __future__ import annotations

import pytest

from agent_modules.config import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    MAX_ACTIONS,
    MAX_CHOICES,
    AutomationDefaults,
    RunConfig,
)


def test_defaults_are_the_documented_ones(monkeypatch):
    """Clears LLM_BASE_URL first: the environment fallback is real behaviour, so a bare default
    assertion would only pass on a machine that happens not to configure one."""
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    config = RunConfig()
    assert config.model == DEFAULT_MODEL
    assert config.base_url == DEFAULT_BASE_URL
    assert config.max_steps == 40
    assert config.allow_submission is True
    assert config.planner == "staged"


def test_hardening_defaults_are_on():
    """These came from the speed/robustness line of work; a default of 0 would disable the guard."""
    config = RunConfig()
    assert config.request_timeout_s == 60.0
    assert config.hedge_after_s == 30.0
    assert config.max_run_seconds == 300.0
    assert config.reasoning_effort == "low"


def test_base_url_and_key_fall_back_to_the_environment(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("EXPLABS_API_KEY", "env-key")
    config = RunConfig()
    assert config.base_url == "https://example.invalid/v1"
    assert config.api_key == "env-key"


def test_explicit_values_beat_the_environment(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("EXPLABS_API_KEY", "env-key")
    config = RunConfig(base_url="https://explicit.invalid/v1", api_key="explicit")
    assert config.base_url == "https://explicit.invalid/v1"
    assert config.api_key == "explicit"


def test_openai_key_is_the_last_resort(monkeypatch):
    monkeypatch.delenv("EXPLABS_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    assert RunConfig().api_key == "openai-key"


def test_field_pause_is_kept():
    """A run's artefacts must be able to show whether the pause was applied."""
    assert RunConfig(field_pause_s=1.0).field_pause_s == 1.0
    assert RunConfig().field_pause_s == 0.0


def test_tracing_is_off_by_default():
    assert RunConfig().trace_path is None
    assert RunConfig(trace_path="/tmp/t.zip").trace_path == "/tmp/t.zip"


def test_normalisation_is_on_by_default():
    assert RunConfig().normalise is True
    assert RunConfig(normalise=False).normalise is False


def test_answers_default_is_not_shared():
    first = RunConfig()
    second = RunConfig()
    first.answers["q"] = "a"
    assert second.answers == {}


def test_answers_runtime_state_is_mutable():
    """The orchestrator clamps the per-call timeout down to the remaining budget each step."""
    config = RunConfig()
    config.request_timeout_s = 12.5
    assert config.request_timeout_s == 12.5


def test_automation_defaults_are_frozen():
    defaults = AutomationDefaults()
    with pytest.raises(Exception):
        defaults.citizenship_country = "Elsewhere"  # type: ignore[misc]


def test_automation_defaults_values():
    defaults = AutomationDefaults()
    assert defaults.accept_matching_consents is True
    assert defaults.prior_employment_with_applying_company is False
    assert defaults.requires_sponsorship_now is False
    assert defaults.requires_sponsorship_future is False


def test_batch_and_choice_limits_are_sane():
    assert MAX_ACTIONS == 12
    assert MAX_CHOICES == 255


def test_thinking_defaults():
    """Normalisation reasons for nothing; answering is left to the provider until it is measured."""
    config = RunConfig()
    assert config.normaliser_thinking is False
    assert config.planner_thinking is None


if __name__ == "__main__":  # pragma: no cover - a convenience, not a test path
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
