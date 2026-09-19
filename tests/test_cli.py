"""Tests for `agent_modules.cli` — flags into a `RunConfig`.

The file formats the flags point at are covered by `test_files`; this file is only about the mapping
and about the parser rejecting what it should.
"""

from __future__ import annotations

import json

import pytest

from agent_modules import cli
from agent_modules.config import DEFAULT_MODEL

BASE = ["--url", "https://example.com/job", "--profile", "profile.json"]


def parse(*argv):
    return cli.build_parser().parse_args(list(argv))


def test_the_defaults_map_through(monkeypatch):
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    config = cli.config_from_args(parse(*BASE))
    assert config.model == DEFAULT_MODEL
    assert config.planner == "staged"
    assert config.max_steps == 40
    assert config.allow_submission is True
    assert config.answers == {}
    assert config.field_pause_s == 0.0


def test_no_submit_disables_submission():
    assert cli.config_from_args(parse(*BASE, "--no-submit")).allow_submission is False


def test_no_default_consents_opts_out():
    config = cli.config_from_args(parse(*BASE, "--no-default-consents"))
    assert config.defaults.accept_matching_consents is False


def test_consent_defaults_are_accepted_unless_refused():
    config = cli.config_from_args(parse(*BASE))
    assert config.defaults.accept_matching_consents is True


def test_max_seconds_is_the_run_budget():
    assert cli.config_from_args(parse(*BASE, "--max-seconds", "120")).max_run_seconds == 120.0


def test_the_budget_defaults_to_five_minutes():
    assert cli.config_from_args(parse(*BASE)).max_run_seconds == 300.0


def test_effort_low_is_passed_through():
    assert cli.config_from_args(parse(*BASE, "--effort", "low")).reasoning_effort == "low"


def test_effort_default_becomes_none():
    """'default' means leave it to the provider, which is not the same as the string 'default'."""
    assert cli.config_from_args(parse(*BASE, "--effort", "default")).reasoning_effort is None


@pytest.mark.parametrize("effort", ["low", "medium", "high"])
def test_the_named_efforts_are_accepted(effort):
    assert cli.config_from_args(parse(*BASE, "--effort", effort)).reasoning_effort == effort


def test_an_invalid_effort_is_rejected():
    with pytest.raises(SystemExit):
        parse(*BASE, "--effort", "turbo")


def test_field_pause_is_carried_into_the_config():
    """Carried so the run's own trace can show whether the pause was applied."""
    assert cli.config_from_args(parse(*BASE, "--field-pause", "1.0")).field_pause_s == 1.0


def test_the_jev_settings_map_through():
    config = cli.config_from_args(
        parse(*BASE, "--jev-model", "jev-latest", "--jev-confidence", "0.5")
    )
    assert config.jev_model == "jev-latest"
    assert config.jev_confidence_floor == 0.5


def test_an_invalid_planner_is_rejected():
    with pytest.raises(SystemExit):
        parse(*BASE, "--planner", "magic")


def test_planner_choices_are_the_supported_ones():
    for choice in ("llm", "jev", "staged"):
        assert cli.config_from_args(parse(*BASE, "--planner", choice)).planner == choice


def test_files_are_read_into_the_config(tmp_path):
    key = tmp_path / "k"
    key.write_text("LLM_KEY=secret\n", encoding="utf-8")
    jev = tmp_path / "j"
    jev.write_text("JEV_KEY=jevsecret\n", encoding="utf-8")
    bank = tmp_path / "default.json"
    bank.write_text(json.dumps({"answers": {"Q": "A"}}), encoding="utf-8")

    config = cli.config_from_args(
        parse(
            *BASE,
            "--api-key-file", str(key),
            "--jev-api-key-file", str(jev),
            "--defaults", str(bank),
            "--asset", "resume_primary=/tmp/r.pdf",
        )
    )
    assert config.api_key == "secret"
    assert config.jev_api_key == "jevsecret"
    assert config.answers == {"Q": "A"}
    assert config.answers_path == str(bank)


def test_assets_are_collected_into_a_mapping():
    parsed = parse(*BASE, "--asset", "resume_primary=/tmp/r.pdf", "--asset", "cover=/tmp/c.pdf")
    assert dict(parsed.asset) == {"resume_primary": "/tmp/r.pdf", "cover": "/tmp/c.pdf"}


def test_url_and_profile_are_required():
    with pytest.raises(SystemExit):
        parse()


def test_headless_and_zoom_map_through():
    config = cli.config_from_args(parse(*BASE, "--headless", "--zoom", "0.5"))
    assert config.headless is True
    assert config.zoom == 0.5


def test_the_journey_log_path_is_carried():
    config = cli.config_from_args(parse(*BASE, "--journey-log", "/tmp/j.jsonl"))
    assert config.journey_log_path == "/tmp/j.jsonl"


# ---------------------------------------------------------------- debugging and diagnostics knobs


def test_request_timeout_maps_through():
    """The per-call bound is wall-clock, so a debugger holding a coroutine open needs this raised."""
    assert cli.config_from_args(parse(*BASE, "--request-timeout", "600")).request_timeout_s == 600.0


def test_request_timeout_defaults_to_sixty_seconds():
    assert cli.config_from_args(parse(*BASE)).request_timeout_s == 60.0


def test_hedge_after_maps_through():
    assert cli.config_from_args(parse(*BASE, "--hedge-after", "5")).hedge_after_s == 5.0


def test_hedge_can_be_disabled_from_the_command_line():
    """0 means never race a duplicate, which is what you want under a debugger."""
    assert cli.config_from_args(parse(*BASE, "--hedge-after", "0")).hedge_after_s == 0.0


def test_hedge_after_defaults_to_thirty_seconds():
    assert cli.config_from_args(parse(*BASE)).hedge_after_s == 30.0


def test_trace_maps_through():
    config = cli.config_from_args(parse(*BASE, "--trace", "/tmp/run.zip"))
    assert config.trace_path == "/tmp/run.zip"


def test_tracing_is_off_unless_asked_for():
    assert cli.config_from_args(parse(*BASE)).trace_path is None


def test_normalisation_is_on_by_default():
    assert cli.config_from_args(parse(*BASE)).normalise is True


def test_normalisation_can_be_switched_off_from_the_command_line():
    """Used to compare the two payloads on a live page."""
    assert cli.config_from_args(parse(*BASE, "--no-normalise")).normalise is False
    assert cli.config_from_args(parse(*BASE, "--normalise")).normalise is True


def test_the_debugging_flags_are_documented_in_help():
    help_text = cli.build_parser().format_help()
    assert "--request-timeout" in help_text
    assert "--hedge-after" in help_text
    assert "--trace" in help_text
    assert "debugger" in help_text


def test_help_text_documents_the_staged_default():
    """The planner help is where a reader learns which path is the default."""
    help_text = cli.build_parser().format_help()
    assert "'staged' classifies the page with Jev" in help_text
    assert "--jev-api-key-file" in help_text
    assert "--max-seconds" in help_text


if __name__ == "__main__":  # pragma: no cover - a convenience, not a test path
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
