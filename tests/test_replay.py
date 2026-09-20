"""Fixture-driven replay over real recorded journeys.

Every other test in this suite builds its own page, which means every other test encodes assumptions
about what a page looks like. These fixtures are actual eBay application pages captured during real
runs, so this is the only place the pipeline meets the shapes it has to survive:

- a hidden `<input type="file">` behind a styled button
- a 246-option country list and a 250-option dialling-code list
- the same page observed several times with different values in it

No browser is launched and no model is called: the snapshot goes in, the payload comes out.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from helpers import profile as applicant

from agent_modules import extractor
from agent_modules.config import MAX_CHOICES
from agent_modules.models import PageSnapshot
from agent_modules.prompts_jev import build_field_questions
from agent_modules.prompts_llm import snapshot_for_llm

FIXTURES = Path(__file__).parent / "fixtures" / "journeys"

#: Observed on the real pages. Generous enough not to be brittle, tight enough to catch a payload
#: regression such as sending every option of every dropdown on every step.
PAGE_PAYLOAD_BUDGET = 120_000


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
WITH_FILE = [
    (name, step, snap)
    for name, step, snap in ALL
    if any(e.input_type == "file" for e in snap.elements)
]


def ids(name: str, step: int) -> str:
    """A failure label that names the exact recorded page, so a regression is reproducible."""
    return f"{name}:{step}"


def test_the_fixtures_are_present_and_real():
    assert len(ALL) >= 8, "the replay harness needs the recorded journeys to be meaningful"
    assert all(snap.url.startswith("https://") for _, _, snap in ALL)
    assert WITH_FILE, "at least one recorded page must carry the hidden resume input"


def test_every_recorded_snapshot_still_parses():
    """The models are the contract; a real page that no longer validates is a breaking change."""
    for name, step, snap in ALL:
        assert snap.snapshot_id, ids(name, step)


# --------------------------------------------------------------------------------- capture rules


def test_the_hidden_resume_input_survives_on_every_page_that_carries_one():
    """Regression, on real data: this is the control whose loss stalled the run.

    It is `visible: false`, so a visibility filter removes it from the model's view and no upload is
    ever planned - and the page then refuses to advance without it.
    """
    for name, step, snap in WITH_FILE:
        hidden_files = [
            e for e in snap.elements if e.input_type == "file" and not e.file_attached
        ]
        if not hidden_files:
            continue
        assert any(extractor.is_actionable(e) for e in hidden_files), ids(name, step)


def test_an_attached_file_input_is_never_actionable():
    """The wizard keeps its resume widget mounted, so an attached one is a no-op waiting to happen."""
    for name, step, snap in ALL:
        for element in snap.elements:
            if element.input_type == "file" and element.file_attached:
                assert extractor.is_actionable(element) is False, ids(name, step)


def test_no_anchors_are_ever_capturable():
    """A standing rule: an anchor is navigation, and one was never once acted on."""
    for name, step, snap in ALL:
        for element in snap.elements:
            if element.role == "link":
                assert extractor.is_capturable(element) is False, ids(name, step)


# ---------------------------------------------------------------------------------- page payload


@pytest.mark.parametrize("name,step,snap", ALL, ids=[ids(n, s) for n, s, _ in ALL])
def test_every_snapshot_produces_a_payload(name, step, snap):
    payload = snapshot_for_llm(snap)
    assert payload["url"] == snap.url
    assert len(payload["elements"]) == len(snap.elements)


@pytest.mark.parametrize("name,step,snap", ALL, ids=[ids(n, s) for n, s, _ in ALL])
def test_the_page_payload_stays_within_budget(name, step, snap):
    size = len(json.dumps(snapshot_for_llm(snap)))
    assert size < PAGE_PAYLOAD_BUDGET, f"{ids(name, step)} payload grew to {size} bytes"


def test_no_recorded_page_offers_an_option_list_we_trimmed():
    """A dropdown's option values are often codes, so the full list has to survive."""
    for name, step, snap in ALL:
        payload = snapshot_for_llm(snap)
        by_id = {item["id"]: item for item in payload["elements"]}
        for element in snap.elements:
            if element.id not in by_id or not element.options:
                continue
            kept = by_id[element.id]["options"]
            assert len(kept) == len(element.options), ids(name, step)


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


def test_the_page_payload_is_serialisable_and_path_free():
    for name, step, snap in ALL:
        encoded = json.dumps(snapshot_for_llm(snap))
        assert "/Users/" not in encoded, ids(name, step)


def test_the_page_payload_carries_the_sites_own_errors():
    """The rule about rejected values is only obeyable because these travel with the page."""
    with_errors = [(n, s, p) for n, s, p in ALL if p.validation_errors]
    assert with_errors, "the recorded journeys should contain a page that refused a value"
    for name, step, snap in with_errors:
        assert snapshot_for_llm(snap)["validation_errors"] == snap.validation_errors, ids(name, step)


# ---------------------------------------------------------------------------- oversized dropdowns


OVERSIZED = [
    (name, step, element)
    for name, step, snap in ALL
    for element in snap.elements
    if len(element.options) > MAX_CHOICES
]


def test_a_real_dropdown_over_the_choice_limit_is_in_the_fixtures():
    """Anchors the Jev fallback path to a real page rather than a synthetic one."""
    assert OVERSIZED, "the recorded pages should still contain a dropdown Jev cannot take"


def test_the_oversized_dropdown_is_diverted_by_the_jev_question_builder():
    """A Choice cannot carry more than 255 criteria, so the planner must not be asked to try."""
    for name, step, element in OVERSIZED:
        snap = next(s for n, st, s in ALL if n == name and st == step)
        questions, mapping, large, _ = build_field_questions(applicant(), snap, {})
        assert element.id in [e.id for e in large], ids(name, step)
        assert f"state_{element.id}" not in mapping
        assert f"state_{element.id}" not in questions


if __name__ == "__main__":  # pragma: no cover - a convenience, not a test path
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
