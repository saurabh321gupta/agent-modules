"""Fixture-driven replay over real recorded journeys.

Every other test in this suite builds its own page, which means every other test encodes my
assumptions about what a page looks like. These fixtures are actual eBay application pages captured
during real runs, so this is the only place the pipeline meets the shapes it has to survive:

- a hidden `<input type="file">` behind a styled button
- a 246-option country list and a 250-option dialling-code list
- the same form observed five times with different values in it

No browser is launched and no model is called: the snapshot goes in, the payload comes out.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fakes import FakeModelClient
from helpers import profile as applicant

from agent_modules import extractor
from agent_modules.config import MAX_CHOICES, AutomationDefaults
from agent_modules.normalizer import Normalizer
from agent_modules.prompts_jev import build_field_questions
from agent_modules.prompts_llm import profile_for_llm, snapshot_for_llm
from agent_modules.models import PageSnapshot

FIXTURES = Path(__file__).parent / "fixtures" / "journeys"

#: Observed on the real page. Generous enough not to be brittle, tight enough to catch a payload
#: regression such as sending every option of every dropdown on every step.
DESCRIBED_PAYLOAD_BUDGET = 40_000


def journeys() -> list[tuple[str, int, PageSnapshot]]:
    loaded: list[tuple[str, int, PageSnapshot]] = []
    for path in sorted(FIXTURES.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            loaded.append(
                (path.name, record["step"], PageSnapshot.model_validate(record["snapshot"]))
            )
    return loaded


ALL = journeys()
FORM_STEPS = [(name, step, snap) for name, step, snap in ALL if snap.elements]
WITH_FILE = [(name, step, snap) for name, step, snap in ALL if any(
    e.input_type == "file" for e in snap.elements
)]


def ids(name: str, step: int, snap: PageSnapshot | None = None) -> str:
    """A failure label that names the exact recorded page, so a regression is reproducible."""
    return f"{name}:{step}"


def test_the_fixtures_are_present_and_real():
    assert len(ALL) >= 8, "the replay harness needs the recorded journeys to be meaningful"
    assert all(snap.url.startswith("https://") for _, _, snap in ALL)
    assert WITH_FILE, "at least one recorded page must carry the hidden resume input"


def test_every_recorded_snapshot_still_parses():
    """The models are the contract; a real page that no longer validates is a breaking change."""
    for name, step, snap in ALL:
        assert snap.snapshot_id, ids(name, step, snap)


def test_every_visible_control_is_described():
    """The model may only act on what it is shown, so a dropped control is an impossible action."""
    for name, step, snap in ALL:
        described = extractor.describe(snap)
        visible = [e for e in snap.elements if e.visible and e.role != "link"]
        assert len(described) >= len(visible), ids(name, step, snap)


def test_the_hidden_resume_input_is_described_on_every_form_step():
    """Regression, on real data: this is the control whose loss stalled the run.

    It is `visible: false`, so a visibility filter removes it from the model's view and no upload is
    ever planned - and the page then refuses to advance without it.
    """
    for name, step, snap in WITH_FILE:
        described = extractor.describe(snap)
        file_controls = [d for d in described if d["input_type"] == "file"]
        assert file_controls, ids(name, step, snap)
        hidden = [d for d in file_controls if not d["visible"]]
        assert hidden, f"{ids(name, step, snap)} should still carry its hidden file input"


def test_no_anchors_are_ever_described():
    """A standing rule: an anchor is navigation, and one was never once acted on."""
    for name, step, snap in ALL:
        roles = {d["tag_role"] for d in extractor.describe(snap)}
        assert "link" not in roles, ids(name, step, snap)


def test_no_recorded_page_offers_an_option_list_we_trimmed():
    """A dropdown's option values are often codes, so the full list has to survive."""
    for name, step, snap in ALL:
        described = extractor.describe(snap)
        by_id = {d["id"]: d for d in described}
        for element in snap.elements:
            if element.id not in by_id:
                continue
            assert len(by_id[element.id]["options"]) == len(element.options), ids(name, step, snap)


def test_the_described_payload_stays_within_budget():
    for name, step, snap in ALL:
        payload = json.dumps(extractor.describe(snap))
        assert len(payload) < DESCRIBED_PAYLOAD_BUDGET, (
            f"{ids(name, step, snap)} described payload grew to {len(payload)} bytes"
        )


def test_the_form_signature_is_stable_across_steps_of_the_same_form():
    """Five observations of one page must cost one normaliser call, not five."""
    signatures: dict[str, set[str]] = {}
    for name, step, snap in ALL:
        key = snap.url.split("?")[0]
        signatures.setdefault(key, set()).add(extractor.form_signature(snap))
    repeated = {key: value for key, value in signatures.items() if len(value) == 1}
    assert repeated, "at least one page was observed several times with an unchanged shape"


def test_the_signature_did_not_change_as_the_form_was_filled_in():
    """The reason the cache is safe: filling a field changes its value, not what it asks.

    Scoped per journey. Two runs can legitimately see slightly different markup for the "same" page,
    and when they do the cache *should* miss - that is the signature doing its job.
    """
    per_journey: dict[str, set[str]] = {}
    for name, step, snap in ALL:
        if "personalInformation" not in snap.url:
            continue
        per_journey.setdefault(name, set()).add(extractor.form_signature(snap))
    assert len(per_journey) >= 2, "need more than one recorded visit to compare"
    for name, signatures in per_journey.items():
        assert len(signatures) == 1, f"{name} changed shape while the form was being filled: {signatures}"


async def test_the_normaliser_payload_carries_the_file_control():
    """The whole point of the fix: the model must be able to see that an upload is wanted."""
    name, step, snap = WITH_FILE[0]
    client = FakeModelClient('{"page_kind": "form", "fields": [], "submit_controls": []}')
    await Normalizer(client, "model").normalise(snap)
    sent = json.loads(client.calls[0]["messages"][1]["content"])
    assert any(c["input_type"] == "file" for c in sent["controls"]), ids(name, step, snap)


async def test_the_normaliser_returns_a_control_for_the_file_input():
    name, step, snap = WITH_FILE[0]
    client = FakeModelClient('{"page_kind": "form", "fields": [], "submit_controls": []}')
    form = await Normalizer(client, "model").normalise(snap)
    files = [f for f in form.fields if f.field_type == "file"]
    assert files, ids(name, step, snap)
    assert files[0].raw_input_type == "file"


async def test_the_normaliser_falls_back_to_the_raw_page_when_the_model_returns_nothing():
    """Even with an empty reply, no describable control may vanish."""
    name, step, snap = WITH_FILE[0]
    client = FakeModelClient("{}")
    form = await Normalizer(client, "model").normalise(snap)
    assert len(form.fields) == len(extractor.describe(snap))


def test_the_planner_payload_is_serialisable_and_path_free():
    for name, step, snap in ALL:
        payload = snapshot_for_llm(snap)
        encoded = json.dumps(payload)
        assert "/Users/" not in encoded, ids(name, step, snap)


def test_the_profile_payload_carries_no_local_path():
    encoded = json.dumps(
        profile_for_llm(applicant(), {"resume_primary": "/tmp/r.pdf"}, AutomationDefaults())
    )
    assert "/secret/resume.pdf" not in encoded


OVERSIZED = [
    (name, step, element)
    for name, step, snap in ALL
    for element in snap.elements
    if len(element.options) > MAX_CHOICES
]


def test_a_real_dropdown_over_the_choice_limit_is_in_the_fixtures():
    """Anchors the Jev fallback path to a real page rather than a synthetic one."""
    assert OVERSIZED, "the recorded pages should still contain a dropdown Jev cannot take"
    assert any(len(element.options) == 330 for _, _, element in OVERSIZED)


def test_the_oversized_dropdown_keeps_every_option_in_the_described_view():
    """Trimming this list is what hid 'Computer and Information Science' at index 88."""
    for name, step, element in OVERSIZED:
        described = next(
            d
            for _, _, snap in ALL
            for d in extractor.describe(snap)
            if d["id"] == element.id and _snapshot_has(snap, element.id)
        )
        assert len(described["options"]) == len(element.options), ids(name, step, None)
        assert described["options"][88] == {
            "value": element.options[88].value,
            "label": element.options[88].label,
        }


def test_the_oversized_dropdown_is_diverted_by_the_jev_question_builder():
    """A Choice cannot carry more than 255 criteria, so the planner must not be asked to try."""
    for name, step, element in OVERSIZED:
        snap = next(s for n, st, s in ALL if n == name and st == step)
        questions, mapping, large, _ = build_field_questions(applicant(), snap, {})
        assert element.id in [e.id for e in large], ids(name, step)
        assert f"state_{element.id}" not in mapping
        assert f"state_{element.id}" not in questions


def _snapshot_has(snap: PageSnapshot, element_id: str) -> bool:
    return any(element.id == element_id for element in snap.elements)


def test_the_country_dropdown_keeps_its_option_values_not_just_its_labels():
    """Option values are codes on this page; labels alone would be unusable."""
    codes = [
        option.value
        for _, _, snap in WITH_FILE
        for element in snap.elements
        for option in element.options
        if option.value.isupper() and len(option.value) <= 4
    ]
    assert codes, "the recorded pages should contain coded option values like 'IND'"


@pytest.mark.parametrize("name,step,snap", ALL, ids=[ids(n, s, p) for n, s, p in ALL])
def test_every_snapshot_is_describable(name, step, snap):
    """Parametrised so a failure names the exact recorded page that broke."""
    described = extractor.describe(snap)
    assert isinstance(described, list)
    assert all({"id", "tag_role", "input_type", "options"} <= set(item) for item in described)


if __name__ == "__main__":  # pragma: no cover - a convenience, not a test path
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
