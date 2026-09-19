"""Builders shared by the tests. Deliberately plain functions rather than fixtures, so a test can
construct exactly the data it needs inline and a failure points at the test, not at a fixture.
"""

from __future__ import annotations

from typing import Any

from agent_modules.models import (
    Action,
    ApplicationPlan,
    CandidateProfile,
    Option,
    PageElement,
    PageSnapshot,
)


def element(
    id: str,
    *,
    role: str = "textbox",
    label: str = "",
    input_type: str | None = "text",
    required: bool = False,
    value: str | None = None,
    enabled: bool = True,
    checked: bool | None = None,
    options: list[tuple[str, str]] | None = None,
    visible: bool = True,
    file_attached: bool = False,
    group: str | None = None,
    readonly: bool = False,
    help_text: str | None = None,
    label_source: str | None = None,
    invalid: bool = False,
) -> PageElement:
    return PageElement(
        id=id,
        role=role,
        label=label,
        label_source=label_source,
        input_type=input_type,
        required=required,
        value=value,
        enabled=enabled,
        checked=checked,
        options=[Option(value=v, label=text) for v, text in (options or [])],
        help_text=help_text,
        visible=visible,
        file_attached=file_attached,
        group=group,
        readonly=readonly,
        invalid=invalid,
    )


def snapshot(
    elements: list[PageElement],
    *,
    snapshot_id: str = "s-1-abc12345",
    url: str = "https://example.com/apply",
    title: str = "Apply",
    visible_text: list[str] | None = None,
    validation_errors: list[str] | None = None,
    blockers: list[str] | None = None,
) -> PageSnapshot:
    return PageSnapshot(
        snapshot_id=snapshot_id,
        url=url,
        title=title,
        visible_text=visible_text if visible_text is not None else ["Apply for this job"],
        elements=elements,
        validation_errors=validation_errors or [],
        blockers=blockers or [],
    )


def profile(**overrides: Any) -> CandidateProfile:
    data: dict[str, Any] = {
        "identity": {"first_name": "Alex", "last_name": "Example", "email": "s@example.com"},
        "location": {"city": "Bangalore", "country": "India"},
        # Present-but-null, exactly as a real generated profile looks. The distinction matters:
        # an absent key and a null key produce different validator errors.
        "preferences": {
            "willing_to_relocate": None,
            "expected_salary": None,
            "referral_source": None,
        },
        "documents": {
            "resume": {
                "asset_id": "resume_primary",
                "filename": "resume.pdf",
                "path": "/secret/resume.pdf",
            }
        },
    }
    data.update(overrides)
    return CandidateProfile.model_validate(data)


def action(kind: str, target: str | None = None, **extra: Any) -> Action:
    fields: dict[str, Any] = {
        "value": None,
        "value_ref": None,
        "option_value": None,
        "checked": None,
        "asset_id": None,
    }
    fields.update(extra)
    return Action(type=kind, target=target, **fields)


def plan(snapshot_id: str, actions: list[Action] | None = None, **overrides: Any) -> ApplicationPlan:
    data: dict[str, Any] = {
        "snapshot_id": snapshot_id,
        "status": "continue",
        "actions": actions or [],
        "reason": "test plan",
        "completion_evidence": None,
    }
    data.update(overrides)
    return ApplicationPlan.model_validate(data)


#: The eBay personal-information page that exposed the hidden-file-input defect: the real resume
#: input is not visible, and a styled button sits in front of it.
EBAY_LIKE_ELEMENTS = [
    element("e1", role="button", label="My Information", input_type="button"),
    element("e9", role="textbox", label="", input_type="file", visible=False),
    element("e10", role="button", label="Upload Resume", input_type="button"),
    element("e11", role="combobox", label="Country", input_type="select-one", required=True,
            options=[("IND", "India"), ("USA", "United States")]),
    element("e12", role="textbox", label="Given Name(s)", required=True, value="Alex"),
    element("e20", role="textbox", label="Email Address", input_type="email", required=True,
            value="s@example.com"),
    element("e25", role="button", label="Next", input_type="submit"),
]
