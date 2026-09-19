"""Tests for `agent_modules.normalizer`."""

from __future__ import annotations

import json

import pytest
from fakes import FakeModelClient
from helpers import EBAY_LIKE_ELEMENTS, element, snapshot

from agent_modules import extractor
from agent_modules.journey import JourneyLogger
from agent_modules.normalizer import Normalizer


def reply(fields, page_kind="form", submit_controls=None) -> str:
    return json.dumps(
        {
            "page_kind": page_kind,
            "fields": fields,
            "submit_controls": submit_controls or [],
        }
    )


def make(snapshot_, reply_text: str, **kwargs):
    client = FakeModelClient(reply_text)
    normalizer = Normalizer(client, "model", journey=kwargs.pop("journey", None))
    return client, normalizer


async def test_fields_are_built_from_the_model_reply():
    snap = snapshot([element("e1", label="City", required=True)])
    client, normalizer = make(snap, reply([{"id": "e1", "question": "Which city?", "field_type": "text", "required": True}]))
    form = await normalizer.normalise(snap)
    assert form.page_kind == "form"
    assert [f.id for f in form.fields] == ["e1"]
    assert form.fields[0].question == "Which city?"
    assert form.fields[0].required is True


async def test_options_always_come_from_the_page_not_the_model():
    """A model's recollection of a dropdown is not authoritative, and option values are often codes."""
    snap = snapshot(
        [
            element(
                "e1",
                role="combobox",
                label="Major",
                input_type="select-one",
                options=[("COMPUTER_AND_INFORMATION_SCIENCE", "Computer and Information Science")],
            )
        ]
    )
    client, normalizer = make(
        snap,
        reply([{"id": "e1", "question": "Major?", "field_type": "select", "required": False}]),
    )
    form = await normalizer.normalise(snap)
    assert [(o.value, o.label) for o in form.fields[0].options] == [
        ("COMPUTER_AND_INFORMATION_SCIENCE", "Computer and Information Science")
    ]


async def test_an_invented_id_is_dropped_and_reported():
    """A normaliser that invents a field id is worse than a bad label."""
    snap = snapshot([element("e1", label="City")])
    client, normalizer = make(
        snap,
        reply(
            [
                {"id": "e1", "question": "City?", "field_type": "text", "required": False},
                {"id": "e99", "question": "Hallucinated?", "field_type": "text", "required": False},
            ]
        ),
    )
    form = await normalizer.normalise(snap)
    assert [f.id for f in form.fields] == ["e1"]
    assert form.dropped_ids == ["e99"]


async def test_duplicate_ids_are_dropped():
    snap = snapshot([element("e1", label="City")])
    client, normalizer = make(
        snap,
        reply(
            [
                {"id": "e1", "question": "First", "field_type": "text", "required": False},
                {"id": "e1", "question": "Second", "field_type": "text", "required": False},
            ]
        ),
    )
    form = await normalizer.normalise(snap)
    assert [f.question for f in form.fields] == ["First"]
    assert form.dropped_ids == ["e1"]


async def test_a_control_the_model_skipped_is_still_reported():
    """Nothing may silently disappear between the page and the planner."""
    snap = snapshot(
        [
            element("e1", label="City"),
            element("e2", label="Postal Code"),
        ]
    )
    client, normalizer = make(
        snap, reply([{"id": "e1", "question": "City?", "field_type": "text", "required": False}])
    )
    form = await normalizer.normalise(snap)
    assert [f.id for f in form.fields] == ["e1", "e2"]
    assert form.fields[1].question == "Postal Code", "falls back to the raw label"


async def test_a_hidden_file_input_survives_the_safety_net():
    """Regression: this is the control whose loss stopped the resume upload from ever happening.

    It is not visible, so a visibility filter would remove it from the model's view *and* from the
    safety net that exists to catch omitted controls.
    """
    snap = snapshot(EBAY_LIKE_ELEMENTS)
    client, normalizer = make(snap, reply([]))
    form = await normalizer.normalise(snap)
    ids = [f.id for f in form.fields]
    assert "e9" in ids
    file_field = next(f for f in form.fields if f.id == "e9")
    assert file_field.field_type == "file"
    assert file_field.raw_input_type == "file"


async def test_the_model_is_shown_the_hidden_file_input():
    snap = snapshot(EBAY_LIKE_ELEMENTS)
    client, normalizer = make(snap, reply([]))
    await normalizer.normalise(snap)
    sent = json.loads(client.calls[0]["messages"][1]["content"])
    assert "e9" in [control["id"] for control in sent["controls"]]


async def test_links_are_never_sent_to_the_normaliser():
    snap = snapshot(
        [
            element("e1", role="link", label="Apply Now", input_type=None),
            element("e2", label="City"),
        ]
    )
    client, normalizer = make(snap, reply([]))
    await normalizer.normalise(snap)
    sent = json.loads(client.calls[0]["messages"][1]["content"])
    assert [c["id"] for c in sent["controls"]] == ["e2"]


async def test_submit_controls_are_filtered_to_real_ids():
    snap = snapshot([element("e1", label="City"), element("e2", role="button", label="Submit")])
    client, normalizer = make(
        snap,
        reply(
            [{"id": "e1", "question": "City?", "field_type": "text", "required": False}],
            submit_controls=["e2", "e404"],
        ),
    )
    form = await normalizer.normalise(snap)
    assert form.submit_controls == ["e2"]


async def test_the_second_call_for_the_same_form_is_served_from_cache():
    """One model call per form, not per step: filling a field does not change what it asks."""
    snap = snapshot([element("e1", label="City")])
    client, normalizer = make(
        snap, reply([{"id": "e1", "question": "City?", "field_type": "text", "required": False}])
    )
    first = await normalizer.normalise(snap)
    second = await normalizer.normalise(snap)
    assert first.cached is False
    assert second.cached is True
    assert len(client.calls) == 1


async def test_a_changed_value_does_not_invalidate_the_cache():
    base = [element("e1", label="City")]
    client, normalizer = make(
        snapshot(base),
        reply([{"id": "e1", "question": "City?", "field_type": "text", "required": False}]),
    )
    await normalizer.normalise(snapshot(base))
    await normalizer.normalise(snapshot([element("e1", label="City", value="Bangalore")]))
    assert len(client.calls) == 1


async def test_a_changed_structure_does_invalidate_the_cache():
    client, normalizer = make(
        snapshot([element("e1", label="City")]),
        reply([{"id": "e1", "question": "City?", "field_type": "text", "required": False}]),
    )
    await normalizer.normalise(snapshot([element("e1", label="City")]))
    await normalizer.normalise(snapshot([element("e1", label="City"), element("e2", label="Postal")]))
    assert len(client.calls) == 2


async def test_malformed_json_degrades_to_the_safety_net():
    snap = snapshot([element("e1", label="City")])
    client, normalizer = make(snap, "not json at all")
    form = await normalizer.normalise(snap)
    assert [f.id for f in form.fields] == ["e1"]


async def test_an_unknown_page_kind_falls_back_to_other():
    snap = snapshot([element("e1", label="City")])
    client, normalizer = make(snap, reply([], page_kind="nonsense"))
    form = await normalizer.normalise(snap)
    assert form.page_kind == "other"


async def test_an_unusable_field_type_is_inferred_from_the_element():
    snap = snapshot([element("e1", label="Email", input_type="email")])
    client, normalizer = make(
        snap, reply([{"id": "e1", "question": "Email?", "field_type": "nonsense", "required": False}])
    )
    form = await normalizer.normalise(snap)
    assert form.fields[0].field_type == "email"


async def test_a_group_is_carried_through():
    snap = snapshot([element("e1", label="From", group="Work experience #2")])
    client, normalizer = make(
        snap,
        reply([{"id": "e1", "question": "Start date?", "field_type": "date", "required": False, "group": "Work experience #2"}]),
    )
    form = await normalizer.normalise(snap)
    assert form.fields[0].group == "Work experience #2"


async def test_the_request_and_response_are_logged(tmp_path):
    snap = snapshot([element("e1", label="City")])
    logger = JourneyLogger(str(tmp_path / "j.jsonl"))
    client, normalizer = make(
        snap,
        reply([{"id": "e1", "question": "City?", "field_type": "text", "required": False}]),
        journey=logger,
    )
    await normalizer.normalise(snap)
    logger.close()
    events = [
        json.loads(line)["event"]
        for line in (tmp_path / "j.jsonl").read_text().splitlines()
        if line
    ]
    assert events == ["normalise_request", "normalise_response", "normalised_form"]
    human = (tmp_path / "j.log").read_text(encoding="utf-8")
    assert "Normaliser request (exact payload):" in human
    assert "Raw response (verbatim):" in human


async def test_the_readable_log_tabulates_the_normalised_form(tmp_path):
    snap = snapshot([element("e1", label="City", required=True)])
    logger = JourneyLogger(str(tmp_path / "j.jsonl"))
    client, normalizer = make(
        snap,
        reply([{"id": "e1", "question": "Which city?", "field_type": "text", "required": True}]),
        journey=logger,
    )
    await normalizer.normalise(snap)
    logger.close()
    human = (tmp_path / "j.log").read_text(encoding="utf-8")
    assert "Which city?" in human
    assert "Page kind         : form" in human


async def test_a_cached_form_is_logged_as_cached(tmp_path):
    snap = snapshot([element("e1", label="City")])
    logger = JourneyLogger(str(tmp_path / "j.jsonl"))
    client, normalizer = make(snap, reply([]), journey=logger)
    await normalizer.normalise(snap)
    await normalizer.normalise(snap)
    logger.close()
    human = (tmp_path / "j.log").read_text(encoding="utf-8")
    assert "reused from cache" in human


def test_the_extractor_is_the_only_place_that_decides_visibility():
    """A guard on the fix: the normalizer must not re-implement the filter."""
    import inspect

    source = inspect.getsource(Normalizer)
    assert "is_capturable" in source
    assert "element.visible" not in source


if __name__ == "__main__":  # pragma: no cover - a convenience, not a test path
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
