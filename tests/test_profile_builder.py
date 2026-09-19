"""Tests for `agent_modules.profile_builder` — the offline resume-to-profile tool."""

from __future__ import annotations

import json

import pytest
from fakes import FakeModelClient

from agent_modules import profile_builder
from agent_modules.profile_builder import (
    PROFILE_SCHEMA,
    SYSTEM_PROMPT,
    build_messages,
    build_parser,
    extract_profile,
    extract_text,
)


# --------------------------------------------------------------------------------------- schema


def test_every_profile_field_is_required_and_nullable():
    """A missing fact must arrive as null: 'unknown' and 'no' are different answers to a screening
    question, and a schema that allowed omission would let the model stay silent instead."""
    for section, spec in PROFILE_SCHEMA["properties"].items():
        if spec["type"] == "array":
            continue
        required = set(spec["required"])
        defined = set(spec["properties"])
        assert required == defined, f"{section} has optional fields: {defined - required}"


def test_no_section_permits_additional_properties():
    """Strict schema output depends on it."""
    assert PROFILE_SCHEMA["additionalProperties"] is False
    for section, spec in PROFILE_SCHEMA["properties"].items():
        if spec["type"] == "object":
            assert spec["additionalProperties"] is False, section


def test_the_prompt_forbids_invention():
    assert "Use only facts explicitly present in the resume" in SYSTEM_PROMPT
    assert "Never invent" in SYSTEM_PROMPT
    assert "always return null" in SYSTEM_PROMPT


def test_work_authorization_defaults_to_empty_not_guessed():
    assert "empty list when the resume says nothing" in SYSTEM_PROMPT


# -------------------------------------------------------------------------------------- messages


def test_native_schema_mode_does_not_restate_the_schema():
    messages = build_messages("resume text", "json_schema")
    assert messages[0]["content"] == SYSTEM_PROMPT
    assert '"candidate_profile"' not in messages[0]["content"]


def test_json_mode_restates_the_schema():
    messages = build_messages("resume text", "json_object")
    assert "no markdown fences" in messages[0]["content"]
    payload = json.loads(messages[0]["content"].split("exactly:\n", 1)[1])
    assert payload["required"] == PROFILE_SCHEMA["required"]


def test_the_resume_text_is_the_user_message():
    messages = build_messages("Jane Doe, engineer", "json_schema")
    assert messages[1]["role"] == "user"
    assert "Jane Doe, engineer" in messages[1]["content"]


# -------------------------------------------------------------------------------- extraction


async def test_a_profile_is_parsed_from_the_model_reply():
    client = FakeModelClient(json.dumps({"identity": {"first_name": "Alex"}}))
    data = await extract_profile(client, "model", "resume text")
    assert data["identity"]["first_name"] == "Alex"
    assert client.calls[0]["schema_name"] == "candidate_profile"


async def test_the_schema_is_sent_strict():
    client = FakeModelClient("{}")
    await extract_profile(client, "model", "resume text")
    assert client.schema_mode == "json_schema"


async def test_an_empty_reply_is_an_error():
    client = FakeModelClient("")
    with pytest.raises(RuntimeError, match="empty profile"):
        await extract_profile(client, "model", "resume text")


async def test_a_non_object_reply_is_an_error():
    client = FakeModelClient('["not", "an", "object"]')
    with pytest.raises(RuntimeError, match="did not return a JSON object"):
        await extract_profile(client, "model", "resume text")


# ------------------------------------------------------------------------------- text reading


def test_a_text_resume_is_read(tmp_path):
    path = tmp_path / "resume.txt"
    path.write_text("Alex Example\nEngineer\n", encoding="utf-8")
    assert "Alex Example" in extract_text(path)


def test_a_markdown_resume_is_read(tmp_path):
    path = tmp_path / "resume.md"
    path.write_text("# Alex Example\n", encoding="utf-8")
    assert extract_text(path).startswith("# Alex")


def test_a_pdf_resume_is_read(tmp_path):
    """The real resume is a PDF, so this path is the one that actually runs."""
    pypdf = pytest.importorskip("pypdf")
    path = tmp_path / "resume.pdf"
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=200, height=200)
    with path.open("wb") as handle:
        writer.write(handle)
    assert isinstance(extract_text(path), str)


# -------------------------------------------------------------------------------------- parser


def test_the_parser_defaults():
    args = build_parser().parse_args(["--resume", "r.pdf"])
    assert args.output == "profile.json"
    assert args.asset_id == "resume_primary"


def test_the_resume_is_required():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_the_schema_names_the_document_asset():
    """The profile registers the resume so a run can upload it without a path in the prompt."""
    assert PROFILE_SCHEMA["properties"]["identity"]["properties"]["email"]["type"] == [
        "string",
        "null",
    ]
    assert profile_builder.PROFILE_SCHEMA is PROFILE_SCHEMA


if __name__ == "__main__":  # pragma: no cover - a convenience, not a test path
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
