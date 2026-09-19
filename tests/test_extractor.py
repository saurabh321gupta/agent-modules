"""Tests for `agent_modules.extractor` — the view of a page handed to a model."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from helpers import EBAY_LIKE_ELEMENTS, element, snapshot

from agent_modules import extractor


def test_described_keys_are_stable():
    """The key names are the normaliser's input contract; renaming one silently changes a payload."""
    described = extractor.describe(snapshot([element("e1", label="City")]))
    assert set(described[0]) == {
        "id",
        "tag_role",
        "input_type",
        "current_label",
        "group",
        "help_text",
        "current_value",
        "checked",
        "required_flag",
        "file_attached",
        "visible",
        "options",
    }


def test_links_are_never_described():
    """Anchors are navigation. Sending them cost payload budget and was never once acted on."""
    described = extractor.describe(
        snapshot([element("e1", role="link", label="Apply Now", input_type=None)])
    )
    assert described == []


def test_anchor_declared_as_button_is_kept():
    """An apply control styled as a link but declaring role=button is a real control."""
    described = extractor.describe(
        snapshot([element("e1", role="button", label="Apply Now", input_type=None)])
    )
    assert [item["id"] for item in described] == ["e1"]


def test_invisible_text_field_is_dropped():
    described = extractor.describe(
        snapshot([element("e1", label="City", visible=False)])
    )
    assert described == []


def test_hidden_file_input_is_kept():
    """Regression: eBay hides the real resume input behind a styled button.

    Dropping it here is what stopped the required upload from ever happening: the field never
    reached the normaliser, so no planner could see it, so no upload was ever planned, and the form
    refused to advance with "No page progress after repeated identical snapshots".
    """
    described = extractor.describe(
        snapshot(
            [
                element("e9", role="textbox", label="", input_type="file", visible=False),
                element("e10", role="button", label="Upload Resume", input_type="button"),
            ]
        )
    )
    assert [item["id"] for item in described] == ["e9", "e10"]
    assert described[0]["input_type"] == "file"
    assert described[0]["file_attached"] is False


def test_hidden_file_input_reaches_the_real_ebay_shape():
    described = extractor.describe(snapshot(EBAY_LIKE_ELEMENTS))
    ids = [item["id"] for item in described]
    assert "e9" in ids, "the hidden resume input must survive the filter"
    assert len(described) == len(EBAY_LIKE_ELEMENTS)


def test_all_options_are_carried_whole():
    """A dropdown's option values are often codes only the page knows, so none may be trimmed."""
    options = [(f"CODE_{i}", f"Option {i}") for i in range(300)]
    described = extractor.describe(
        snapshot([element("e1", role="combobox", label="Major", input_type="select-one", options=options)])
    )
    assert len(described[0]["options"]) == 300
    assert described[0]["options"][88]["value"] == "CODE_88"


def test_form_signature_ignores_values():
    """Filling a field changes its value, not what it asks, so the cache must survive it."""
    empty = snapshot([element("e1", label="City"), element("e2", label="Email")], url="https://x.com/a?q=1")
    filled = snapshot(
        [element("e1", label="City", value="Bangalore"), element("e2", label="Email", value="s@x.com")],
        url="https://x.com/a?q=1",
    )
    assert extractor.form_signature(empty) == extractor.form_signature(filled)


def test_form_signature_ignores_query_string():
    a = snapshot([element("e1", label="City")], url="https://x.com/a?step=1")
    b = snapshot([element("e1", label="City")], url="https://x.com/a?step=2")
    assert extractor.form_signature(a) == extractor.form_signature(b)


def test_form_signature_changes_with_structure():
    a = snapshot([element("e1", label="City")])
    b = snapshot([element("e1", label="City"), element("e2", label="Postal Code")])
    assert extractor.form_signature(a) != extractor.form_signature(b)


def test_form_signature_includes_file_inputs():
    """A form that gains a resume upload is a different form and must not reuse a cached answer."""
    without = snapshot([element("e10", role="button", label="Upload Resume", input_type="button")])
    with_file = snapshot(
        [
            element("e9", role="textbox", label="", input_type="file", visible=False),
            element("e10", role="button", label="Upload Resume", input_type="button"),
        ]
    )
    assert extractor.form_signature(without) != extractor.form_signature(with_file)


def test_form_signature_is_stable_across_processes():
    """Guards the sha1 choice: `hash()` is salted per process, so a signature built on it would
    differ between runs and could not be replayed or cached meaningfully."""
    snapshot_json = snapshot([element("e1", label="City")]).model_dump_json()
    code = (
        "import sys; sys.path.insert(0, 'src');"
        "from agent_modules.types import PageSnapshot;"
        "from agent_modules.extractor import form_signature;"
        f"s = PageSnapshot.model_validate_json({snapshot_json!r});"
        "print(form_signature(s))"
    )
    root = Path(__file__).resolve().parents[1]
    outputs = set()
    for seed in ("0", "1", "12345"):
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=root,
            capture_output=True,
            text=True,
            env={"PYTHONHASHSEED": seed, "PATH": "/usr/bin:/bin"},
            check=True,
        )
        outputs.add(result.stdout.strip())
    assert len(outputs) == 1, f"signature varied with PYTHONHASHSEED: {outputs}"


@pytest.mark.parametrize(
    ("raw", "input_type", "role", "expected"),
    [
        ("email", None, "textbox", "email"),
        (None, "email", "textbox", "email"),
        (None, "tel", "textbox", "phone"),
        (None, "number", "textbox", "number"),
        (None, "date", "textbox", "date"),
        (None, "file", "textbox", "file"),
        (None, "select-one", "combobox", "select"),
        (None, "text", "textbox", "text"),
        (None, "button", "button", "button"),
        ("nonsense", "text", "textbox", "text"),
        (None, None, "generic", "unknown"),
    ],
)
def test_field_type_inference(raw, input_type, role, expected):
    assert extractor.field_type_for(raw, input_type, role) == expected


def test_page_kind_falls_back_to_other():
    assert extractor.page_kind_of("form") == "form"
    assert extractor.page_kind_of("nonsense") == "other"
    assert extractor.page_kind_of(None) == "other"


if __name__ == "__main__":  # pragma: no cover - a convenience, not a test path
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
