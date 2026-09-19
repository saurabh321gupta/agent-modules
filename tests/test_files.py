"""Tests for `agent_modules.files` — the small config files a run is built from."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_modules.files import load_answer_bank, parse_asset, read_api_key


# ------------------------------------------------------------------------------------- key files


def test_a_bare_key_file_is_read_verbatim(tmp_path):
    path = tmp_path / "k"
    path.write_text("sk-abc123\n", encoding="utf-8")
    assert read_api_key(str(path)) == "sk-abc123"


def test_a_name_value_key_file_is_unwrapped(tmp_path):
    """Both plain keys and `NAME=value` lines appear in practice, and a stray prefix produces an
    invalid bearer token that fails far from its cause."""
    path = tmp_path / "k"
    path.write_text("CMD_API_KEY=sk-abc123\n", encoding="utf-8")
    assert read_api_key(str(path)) == "sk-abc123"


def test_only_the_first_equals_sign_splits(tmp_path):
    """Keys can legitimately contain `=`, so the split must be on the first one only."""
    path = tmp_path / "k"
    path.write_text("KEY=abc=def", encoding="utf-8")
    assert read_api_key(str(path)) == "abc=def"


def test_surrounding_whitespace_is_trimmed(tmp_path):
    path = tmp_path / "k"
    path.write_text("  sk-abc123  \n\n", encoding="utf-8")
    assert read_api_key(str(path)) == "sk-abc123"


def test_an_empty_key_file_is_an_error(tmp_path):
    path = tmp_path / "k"
    path.write_text("\n", encoding="utf-8")
    with pytest.raises(Exception, match="empty"):
        read_api_key(str(path))


def test_a_prefix_with_no_value_is_an_error(tmp_path):
    path = tmp_path / "k"
    path.write_text("KEY=\n", encoding="utf-8")
    with pytest.raises(Exception, match="empty"):
        read_api_key(str(path))


def test_a_missing_key_file_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_api_key(str(tmp_path / "nope"))


# ------------------------------------------------------------------------------------ answer bank


def test_an_answer_bank_is_read_from_its_answers_key(tmp_path):
    path = tmp_path / "default.json"
    path.write_text(json.dumps({"answers": {"Notice period": "30 days"}}), encoding="utf-8")
    assert load_answer_bank(str(path)) == {"Notice period": "30 days"}


def test_a_bare_object_is_accepted_as_an_answer_bank(tmp_path):
    path = tmp_path / "default.json"
    path.write_text(json.dumps({"Notice period": "30 days"}), encoding="utf-8")
    assert load_answer_bank(str(path)) == {"Notice period": "30 days"}


def test_scalar_answers_are_stringified(tmp_path):
    """A number in the bank is still an answer; the form needs text."""
    path = tmp_path / "default.json"
    path.write_text(json.dumps({"answers": {"Salary": 8000000, "Relocate": True}}), encoding="utf-8")
    assert load_answer_bank(str(path)) == {"Salary": "8000000", "Relocate": "True"}


def test_an_empty_bank_is_an_error(tmp_path):
    path = tmp_path / "default.json"
    path.write_text(json.dumps({"answers": {}}), encoding="utf-8")
    with pytest.raises(Exception, match="non-empty"):
        load_answer_bank(str(path))


def test_a_non_object_bank_is_an_error(tmp_path):
    path = tmp_path / "default.json"
    path.write_text(json.dumps(["nope"]), encoding="utf-8")
    with pytest.raises(Exception, match="JSON object"):
        load_answer_bank(str(path))


def test_a_nested_answer_is_an_error(tmp_path):
    """Questions and answers must be plain pairs, or they cannot be sent to a model as text."""
    path = tmp_path / "default.json"
    path.write_text(json.dumps({"answers": {"Q": {"nested": "x"}}}), encoding="utf-8")
    with pytest.raises(Exception, match="plain question/answer pairs"):
        load_answer_bank(str(path))


def test_malformed_json_is_surfaced(tmp_path):
    path = tmp_path / "default.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        load_answer_bank(str(path))


# ----------------------------------------------------------------------------------------- assets


def test_an_asset_is_split_and_absolutised():
    asset_id, path = parse_asset("resume_primary=relative/resume.pdf")
    assert asset_id == "resume_primary"
    assert Path(path).is_absolute()
    assert path.endswith("relative/resume.pdf")


def test_an_absolute_asset_path_is_kept():
    assert parse_asset("resume_primary=/tmp/r.pdf") == ("resume_primary", "/tmp/r.pdf")


def test_an_asset_path_with_equals_signs_is_kept():
    """A path may legitimately contain `=`, so only the first one splits."""
    _, path = parse_asset("resume_primary=/tmp/a=b.pdf")
    assert path == "/tmp/a=b.pdf"


@pytest.mark.parametrize("value", ["no-equals", "=path", "id=", ""])
def test_a_malformed_asset_is_rejected(value):
    with pytest.raises(Exception, match="asset_id=/absolute/path"):
        parse_asset(value)


if __name__ == "__main__":  # pragma: no cover - a convenience, not a test path
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
