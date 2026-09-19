"""The compact view of a page's controls, as handed to a model. Layer 1.

This is the only place that decides which controls a model gets to see, so a control that is missing
here cannot be acted on by anything downstream. That makes the filter below load-bearing, and it has
one deliberate exception.
"""

from __future__ import annotations

import hashlib
from typing import Any, Literal

from .models import PageElement, PageSnapshot

FieldType = Literal[
    "text", "long_text", "email", "phone", "number", "date",
    "select", "multi_select", "checkbox", "radio", "file", "button", "unknown",
]
PageKind = Literal["form", "job_listing", "confirmation", "captcha", "login", "other"]

FIELD_TYPES = frozenset(FieldType.__args__)  # type: ignore[attr-defined]
PAGE_KINDS = frozenset(PageKind.__args__)  # type: ignore[attr-defined]


def is_capturable(element: PageElement) -> bool:
    """Whether this control is offered to a model at all.

    Anchors are navigation, not application controls, and are never sent.

    A file input is kept even when it is not visible. A site normally hides the real
    `<input type="file">` and styles a button in front of it, so visibility says nothing about
    whether the control matters - and `set_input_files` works on a hidden input perfectly well.
    Dropping it makes the field invisible to every later stage, which is how a required resume
    upload silently never happens and the form refuses to advance.
    """
    if element.role == "link":
        return False
    if element.input_type == "file":
        return True
    return element.visible


def is_actionable(element: PageElement) -> bool:
    """Whether a model could have anything to *do* with this control.

    A file input that already holds a file is the case this exists for, and it is not hypothetical:
    an application wizard keeps its resume widget mounted in the page shell, so the same hidden input
    appears on every step, already attached after the first upload. Showing it invites the model to
    upload the resume again on page four - and because re-uploading the same file changes nothing,
    the identical plan repeats until the run is declared stuck. There is nothing to do with an
    attached file.

    Kept separate from `is_capturable` on purpose: a form's *shape* does not change when a file is
    attached, so the signature that keys the normaliser cache must still include it.
    """
    if element.input_type == "file" and element.file_attached:
        return False
    return is_capturable(element)


def describe(snapshot: PageSnapshot) -> list[dict[str, Any]]:
    """The compact view handed to the normaliser.

    Every option is included: a field cannot be understood, and a dropdown cannot be answered,
    without the values the page actually offers.
    """
    described: list[dict[str, Any]] = []
    for element in snapshot.elements:
        if not is_actionable(element):
            continue
        described.append(
            {
                "id": element.id,
                "tag_role": element.role,
                "input_type": element.input_type,
                "current_label": element.label,
                "group": element.group,
                "help_text": element.help_text,
                "current_value": element.value,
                "checked": element.checked,
                "required_flag": element.required,
                "file_attached": element.file_attached,
                # Everything else here is visible by definition. A file input is the exception, and
                # saying so explains why its label is often empty: a styled button is standing in
                # front of it, so the control the applicant sees is not the control that uploads.
                "visible": element.visible,
                "options": [
                    {"value": option.value, "label": option.label} for option in element.options
                ],
            }
        )
    return described


def form_signature(snapshot: PageSnapshot) -> str:
    """Identifies the *shape* of a form, ignoring the values in it.

    Filling a field changes its value but not what it is asking, so the normalised questions hold for
    as long as the candidate is on that page. Keying on the shape is what keeps normalisation to one
    model call per form instead of one per step.

    Hashed with sha1 rather than `hash()`: the built-in is salted per process, so a signature built on
    it would differ between runs and could not be replayed or tested.
    """
    parts = [
        f"{element.role}|{element.input_type}|{element.group or ''}|{element.label}"
        for element in snapshot.elements
        if is_capturable(element)
    ]
    digest = hashlib.sha1("||".join(parts).encode("utf-8"), usedforsecurity=False).hexdigest()[:16]
    return f"{snapshot.url.split('?')[0]}::{digest}"


def field_type_for(value: Any, input_type: str | None, role: str) -> FieldType:
    """Trust a model's label when it is one of ours, else infer from the element itself."""
    if isinstance(value, str) and value in FIELD_TYPES:
        return value  # type: ignore[return-value]
    kind = (input_type or "").lower()
    if kind == "email":
        return "email"
    if kind in ("tel", "phone"):
        return "phone"
    if kind == "number":
        return "number"
    if kind in ("date", "month", "week", "time", "datetime-local"):
        return "date"
    if kind == "file":
        return "file"
    if kind == "checkbox":
        return "checkbox"
    if kind == "radio":
        return "radio"
    if role == "combobox":
        return "select"
    if role == "textbox":
        return "text"
    if role == "button":
        return "button"
    return "unknown"


def page_kind_of(value: Any) -> PageKind:
    return value if isinstance(value, str) and value in PAGE_KINDS else "other"  # type: ignore[return-value]
