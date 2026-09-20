"""Tests for `agent_modules.planner_jev`.

The property under test throughout: the model only ever picks a *source key* or an *option*. Every
value that reaches the form is resolved in code from the profile or the answer bank.
"""

from __future__ import annotations

import pytest
from fakes import FakeModelClient
from helpers import element, profile, snapshot

from agent_modules.config import MAX_ACTIONS, MAX_CHOICES, AutomationDefaults
from agent_modules.planner_jev import JevPlanner, LlmFieldAssist

ASSETS = {"resume_primary": "/tmp/resume.pdf"}
APPLICANT = profile()
BANK = {"Current notice period": "30 days"}


class StubJev:
    """Stands in for the Jev client's `ask`."""

    def __init__(self, answers=None, model="jev-latest"):
        self.answers = answers or {}
        self.model = model
        self.calls = []

    async def ask(self, state, questions):
        self.calls.append({"state": state, "questions": questions})
        return {"answers": self.answers, "model": self.model, "usage": {}}


def make(answers=None, **kwargs) -> tuple[StubJev, JevPlanner]:
    client = StubJev(answers)
    planner = JevPlanner(
        client,
        assets=ASSETS,
        defaults=AutomationDefaults(),
        answer_bank=BANK,
        field_assist=kwargs.pop("field_assist", None),
        journey=kwargs.pop("journey", None),
    )
    return client, planner


def form_snapshot(*elements):
    return snapshot(list(elements))


def choice(value: str, confidence: float = 0.9) -> dict:
    return {"type": "choice", "choice": value, "confidence": confidence}


def noul(score: float) -> dict:
    return {"type": "noul", "noul": score}


# ---------------------------------------------------------------------------------------- upload


async def test_a_registered_asset_is_uploaded_to_a_visible_file_input():
    _, planner = make({"page_kind": choice("form")})
    snap = form_snapshot(
        element("e9", role="textbox", label="Resume", input_type="file", visible=True)
    )
    plan = await planner.next_step(APPLICANT, snap, [])
    uploads = [a for a in plan.actions if a.type == "upload"]
    assert len(uploads) == 1
    assert uploads[0].target == "e9"
    assert uploads[0].asset_id == "resume_primary"


async def test_a_hidden_file_input_is_uploaded_too():
    """Regression: this is the real-resume-input shape, hidden behind a styled button.

    Unlike the generative path, the Jev path scans every element for a file input rather than
    filtering on visibility, so it always composed this action correctly.
    """
    _, planner = make({"page_kind": choice("form")})
    snap = form_snapshot(
        element("e9", role="textbox", label="", input_type="file", visible=False),
        element("e10", role="button", label="Upload Resume", input_type="button"),
    )
    plan = await planner.next_step(APPLICANT, snap, [])
    assert [a.target for a in plan.actions if a.type == "upload"] == ["e9"]


async def test_an_already_attached_file_is_not_uploaded_again():
    _, planner = make({"page_kind": choice("form")})
    snap = form_snapshot(
        element("e9", role="textbox", label="Resume", input_type="file", visible=False, file_attached=True)
    )
    plan = await planner.next_step(APPLICANT, snap, [])
    assert [a for a in plan.actions if a.type == "upload"] == []


async def test_nothing_is_uploaded_when_the_asset_was_not_registered():
    planner = JevPlanner(StubJev({"page_kind": choice("form")}), assets={}, answer_bank=BANK)
    snap = form_snapshot(
        element("e9", role="textbox", label="Resume", input_type="file", visible=False)
    )
    plan = await planner.next_step(APPLICANT, snap, [])
    assert [a for a in plan.actions if a.type == "upload"] == []


# ------------------------------------------------------------------------------------ answering


async def test_a_dropdown_choice_becomes_a_select_with_the_pages_own_value():
    _, planner = make(
        {
            "page_kind": choice("form"),
            "state_e1": choice("IND"),
        }
    )
    snap = form_snapshot(
        element("e1", role="combobox", label="Country", input_type="select-one",
                options=[("", "Please Select"), ("IND", "India")])
    )
    plan = await planner.next_step(APPLICANT, snap, [])
    selects = [a for a in plan.actions if a.type == "select"]
    assert selects[0].option_value == "IND"


async def test_a_profile_fact_choice_is_resolved_to_its_value():
    """The model picks the dotted path; the value never passes through the model."""
    _, planner = make({"page_kind": choice("form"), "state_e1": choice("location.city")})
    snap = form_snapshot(element("e1", label="City"))
    plan = await planner.next_step(APPLICANT, snap, [])
    fills = [a for a in plan.actions if a.type == "fill"]
    assert fills[0].value == "Bangalore"


async def test_an_approved_answer_choice_is_resolved_from_the_bank():
    _, planner = make({"page_kind": choice("form"), "state_e1": choice("answers.Current notice period")})
    snap = form_snapshot(element("e1", label="Notice period"))
    plan = await planner.next_step(APPLICANT, snap, [])
    assert [a for a in plan.actions if a.type == "fill"][0].value == "30 days"


async def test_a_choice_below_the_confidence_floor_is_ignored():
    _, planner = make({"page_kind": choice("form"), "state_e1": choice("location.city", confidence=0.1)})
    snap = form_snapshot(element("e1", label="City"))
    plan = await planner.next_step(APPLICANT, snap, [])
    assert [a for a in plan.actions if a.type == "fill"] == []


async def test_a_field_already_holding_the_right_value_is_not_rewritten():
    _, planner = make({"page_kind": choice("form"), "state_e1": choice("location.city")})
    snap = form_snapshot(element("e1", label="City", value="Bangalore"))
    plan = await planner.next_step(APPLICANT, snap, [])
    assert [a for a in plan.actions if a.type == "fill"] == []


async def test_a_yes_no_question_above_the_threshold_ticks_the_box():
    _, planner = make({"page_kind": choice("form"), "state_e1": noul(0.9)})
    snap = form_snapshot(element("e1", role="checkbox", label="I consent", checked=False))
    plan = await planner.next_step(APPLICANT, snap, [])
    assert [a.checked for a in plan.actions if a.type == "set_checked"] == [True]


async def test_a_yes_no_question_below_the_threshold_leaves_the_box_alone():
    _, planner = make({"page_kind": choice("form"), "state_e1": noul(0.2)})
    snap = form_snapshot(element("e1", role="checkbox", label="I consent", checked=False))
    plan = await planner.next_step(APPLICANT, snap, [])
    assert [a for a in plan.actions if a.type == "set_checked"] == []


# -------------------------------------------------------------------------------------- gating


async def test_an_ungrounded_required_field_stops_the_run():
    _, planner = make({"page_kind": choice("form"), "state_e1": choice("none")})
    snap = form_snapshot(element("e1", label="Current salary", required=True))
    plan = await planner.next_step(APPLICANT, snap, [])
    assert plan.status == "needs_input"
    assert "Current salary" in plan.reason


async def test_an_ungrounded_optional_field_does_not_stop_the_run():
    _, planner = make({"page_kind": choice("form"), "state_e1": choice("none")})
    snap = form_snapshot(element("e1", label="Middle name", required=False))
    plan = await planner.next_step(APPLICANT, snap, [])
    assert plan.status == "continue"


async def test_submit_is_held_back_when_the_form_is_not_ready():
    _, planner = make(
        {"page_kind": choice("form"), "submit_ready": noul(0.2), "advance": choice("e1")}
    )
    snap = form_snapshot(element("e1", role="button", label="Submit Application", input_type="submit"))
    plan = await planner.next_step(APPLICANT, snap, [])
    assert [a for a in plan.actions if a.type == "click"] == []
    assert "held back submit" in plan.reason


async def test_submit_is_clicked_when_the_form_is_ready():
    _, planner = make(
        {"page_kind": choice("form"), "submit_ready": noul(0.9), "advance": choice("e1")}
    )
    snap = form_snapshot(element("e1", role="button", label="Submit Application", input_type="submit"))
    plan = await planner.next_step(APPLICANT, snap, [])
    assert [a.target for a in plan.actions if a.type == "click"] == ["e1"]


async def test_nothing_is_clicked_while_the_site_still_flags_a_field_invalid():
    """A page with invalid fields will not advance, whatever the model thinks."""
    _, planner = make(
        {"page_kind": choice("form"), "submit_ready": noul(0.9), "advance": choice("e1")}
    )
    snap = form_snapshot(
        element("e1", role="button", label="Next", input_type="submit"),
        element("e2", label="Email", invalid=True),
    )
    plan = await planner.next_step(APPLICANT, snap, [])
    assert [a for a in plan.actions if a.type == "click"] == []
    assert "still flags" in plan.reason


async def test_next_is_clicked_on_a_form_step():
    _, planner = make(
        {"page_kind": choice("form"), "submit_ready": noul(0.5), "advance": choice("e1")}
    )
    snap = form_snapshot(element("e1", role="button", label="Next", input_type="submit"))
    plan = await planner.next_step(APPLICANT, snap, [])
    assert [a.target for a in plan.actions if a.type == "click"] == ["e1"]


async def test_the_advance_click_is_always_last():
    """The executor stops a batch at a click, so anything after it would never run."""
    _, planner = make(
        {
            "page_kind": choice("form"),
            "state_e1": choice("location.city"),
            "submit_ready": noul(0.9),
            "advance": choice("e9"),
        }
    )
    snap = form_snapshot(
        element("e1", label="City"),
        element("e9", role="button", label="Next", input_type="submit"),
    )
    plan = await planner.next_step(APPLICANT, snap, [])
    assert plan.actions[-1].type == "click"


# ---------------------------------------------------------------------------------- page kinds


@pytest.mark.parametrize(
    ("kind", "expected"),
    [("captcha", "blocked"), ("login", "blocked"), ("confirmation", "complete")],
)
async def test_special_pages_short_circuit(kind, expected):
    _, planner = make({"page_kind": choice(kind)})
    plan = await planner.next_step(APPLICANT, form_snapshot(element("e1", label="City")), [])
    assert plan.status == expected


async def test_a_job_listing_drops_page_furniture_and_keeps_only_the_apply_control():
    """An advert's fields are search and alert boxes, never candidate answers."""
    _, planner = make(
        {
            "page_kind": choice("job_listing"),
            "state_e1": choice("location.city"),
            "advance": choice("e2"),
        }
    )
    snap = form_snapshot(
        element("e1", label="Search location"),
        element("e2", role="button", label="Apply Now", input_type="button"),
    )
    plan = await planner.next_step(APPLICANT, snap, [])
    assert [a.target for a in plan.actions] == ["e2"]
    assert all(a.type == "click" for a in plan.actions)


async def test_a_job_listing_ignores_a_control_that_does_not_start_an_application():
    _, planner = make({"page_kind": choice("job_listing"), "advance": choice("e1")})
    snap = form_snapshot(element("e1", role="button", label="Job alerts", input_type="button"))
    plan = await planner.next_step(APPLICANT, snap, [])
    assert plan.actions == []
    assert "does not start an application" in plan.reason


# ------------------------------------------------------------------------------ action ceiling


async def test_the_action_list_is_capped():
    elements = [element(f"e{i}", label=f"Field {i}") for i in range(1, 30)]
    answers = {"page_kind": choice("form")}
    for i in range(1, 30):
        answers[f"state_e{i}"] = choice("location.city")
    _, planner = make(answers)
    plan = await planner.next_step(APPLICANT, form_snapshot(*elements), [])
    assert len(plan.actions) <= MAX_ACTIONS - 1


# -------------------------------------------------------------------------- llm field fallbacks


async def test_an_oversized_dropdown_is_handled_by_the_model_fallback():
    """The diverted dropdown is resolved by the model, and the chosen value must be a real one."""
    options = [(f"C{i}", f"Choice {i}") for i in range(MAX_CHOICES + 1)]
    assist = FakeModelClient(['{"option_value": "C100", "reason": "closest"}'])
    _, planner = make(
        {"page_kind": choice("form")},
        field_assist=LlmFieldAssist(assist, "model"),
    )
    snap = form_snapshot(
        element("e1", role="combobox", label="Major", input_type="select-one", options=options)
    )
    plan = await planner.next_step(APPLICANT, snap, [])
    assert [a.option_value for a in plan.actions if a.type == "select"] == ["C100"]


async def test_a_fallback_option_not_on_the_page_is_discarded():
    """A model's answer is only trusted when the page itself offers it."""
    options = [(f"C{i}", f"Choice {i}") for i in range(MAX_CHOICES + 1)]
    assist = FakeModelClient(['{"option_value": "NOT_A_REAL_CODE"}'])
    _, planner = make({"page_kind": choice("form")}, field_assist=LlmFieldAssist(assist, "model"))
    snap = form_snapshot(
        element("e1", role="combobox", label="Major", input_type="select-one", options=options)
    )
    plan = await planner.next_step(APPLICANT, snap, [])
    assert [a for a in plan.actions if a.type == "select"] == []


async def test_without_an_assist_oversized_dropdowns_are_left_alone():
    _, planner = make({"page_kind": choice("form")})
    options = [(f"C{i}", f"Choice {i}") for i in range(MAX_CHOICES + 1)]
    snap = form_snapshot(
        element("e1", role="combobox", label="Major", input_type="select-one", options=options)
    )
    plan = await planner.next_step(APPLICANT, snap, [])
    assert [a for a in plan.actions if a.type == "select"] == []


async def test_a_rejected_value_is_repaired_and_the_repair_sticks():
    """Otherwise the next step's ordinary question would overwrite the accepted value with the
    one the site already refused."""
    assist = FakeModelClient(['{"value": "0000000000", "reason": "drop the country code"}'])
    planner = JevPlanner(
        StubJev({"page_kind": choice("form")}),
        assets=ASSETS,
        answer_bank=BANK,
        field_assist=LlmFieldAssist(assist, "model"),
    )
    snap = form_snapshot(element("e1", label="Phone Number", value="+910000000000", invalid=True))
    plan = await planner.next_step(APPLICANT, snap, [])
    assert [a.value for a in plan.actions if a.type == "fill"] == ["0000000000"]
    assert planner.repaired_values["Phone Number"] == "0000000000"


# -------------------------------------------------------------------------------------- payload


async def test_the_jev_state_omits_options_and_file_inputs():
    """Options already travel as the question's criteria, and sending them twice doubled the request."""
    client, planner = make({"page_kind": choice("form")})
    snap = form_snapshot(
        element("e1", role="combobox", label="Country", input_type="select-one",
                options=[("IND", "India")]),
        element("e2", role="textbox", label="Password", input_type="password"),
        element("e3", label="Resume", input_type="file", visible=False),
    )
    await planner.next_step(APPLICANT, snap, [])
    sent = client.calls[0]["state"]
    ids = [e["id"] for e in sent["current_page"]["elements"]]
    assert ids == ["e1"]
    assert "options" not in sent["current_page"]["elements"][0]


async def test_the_state_carries_no_history():
    """The whole transcript used to ride along here; now the state is just the facts and the page."""
    client, planner = make({"page_kind": choice("form")})
    await planner.next_step(APPLICANT, form_snapshot(element("e1", label="City")))
    assert "recent_history" not in client.calls[0]["state"]


if __name__ == "__main__":  # pragma: no cover - a convenience, not a test path
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
