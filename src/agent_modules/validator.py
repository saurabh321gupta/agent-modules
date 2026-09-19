"""Rejects a plan before it can touch the page. Layer 1.

The model proposes; this decides. Every rule here exists because the corresponding mistake reached a
real application at least once, so a rule is not removed without a replacement.
"""

from __future__ import annotations

from typing import Any

from .config import MAX_ACTIONS
from .types import Action, ApplicationPlan, CandidateProfile, PageElement, PageSnapshot

SUBMIT_MARKERS = (
    "submit application",
    "send application",
    "finish application",
    "complete application",
    "complete and submit",
)
SUBMIT_EXACT = {"submit", "final submit"}


class PlanValidationError(ValueError):
    """A plan that must not be executed."""


def resolve_profile_ref(profile: CandidateProfile, ref: str) -> Any:
    """Resolve a dotted profile path to a scalar. Unknown or empty values are an error, not None."""
    current: Any = profile.model_dump(mode="python")
    for part in ref.split("."):
        if not part or not isinstance(current, dict) or part not in current:
            raise PlanValidationError(f"Unknown candidate profile reference: {ref}")
        current = current[part]
    if current is None or current == "":
        raise PlanValidationError(f"Candidate profile value is unknown: {ref}")
    if isinstance(current, (dict, list)):
        raise PlanValidationError(f"Profile reference must resolve to a scalar: {ref}")
    return current


def element_for(snapshot: PageSnapshot, target: str) -> PageElement:
    for item in snapshot.elements:
        if item.id == target:
            return item
    raise PlanValidationError(f"Target {target!r} is not in snapshot {snapshot.snapshot_id}")


def candidate_asset_ids(profile: CandidateProfile) -> set[str]:
    asset_ids: set[str] = set()
    for document in profile.documents.values():
        if isinstance(document, dict) and isinstance(document.get("asset_id"), str):
            asset_ids.add(document["asset_id"])
    return asset_ids


def looks_like_submit(label: str) -> bool:
    text = label.lower()
    return any(term in text for term in SUBMIT_MARKERS) or text.strip() in SUBMIT_EXACT


def unresolved_required(snapshot: PageSnapshot) -> list[str]:
    """Required controls that still have no answer, named the way a human would recognise them."""
    unresolved: list[str] = []
    for element in snapshot.elements:
        if not element.required or not element.enabled:
            continue
        if element.input_type == "file":
            missing = not element.file_attached
        elif element.role in {"checkbox", "radio"}:
            missing = element.checked is not True
        else:
            missing = not (element.value or "").strip()
        if missing:
            unresolved.append(element.label or element.id)
    return unresolved


def _resolved_value(action: Action, profile: CandidateProfile) -> str:
    if action.value_ref:
        return str(resolve_profile_ref(profile, action.value_ref))
    if action.value is None:
        raise PlanValidationError(f"{action.type} requires value or value_ref")
    return action.value


def validate_plan(
    plan: ApplicationPlan,
    snapshot: PageSnapshot,
    profile: CandidateProfile,
    assets: dict[str, str],
    allow_submission: bool,
) -> None:
    """Raise `PlanValidationError` with a recoverable reason, or return cleanly."""
    if plan.snapshot_id != snapshot.snapshot_id:
        raise PlanValidationError("Plan was created for an expired snapshot")
    if len(plan.actions) > MAX_ACTIONS:
        raise PlanValidationError(f"Action batch exceeds the maximum of {MAX_ACTIONS} actions")

    for action in plan.actions:
        if action.type == "scroll":
            if action.target:
                element_for(snapshot, action.target)
            continue

        if not action.target:
            raise PlanValidationError(f"{action.type} requires a target")
        element = element_for(snapshot, action.target)
        if not element.enabled:
            raise PlanValidationError(f"Target {action.target} is disabled")

        if action.type == "fill":
            if element.role != "textbox" or element.input_type in {"file", "checkbox", "radio"}:
                raise PlanValidationError(f"Target {action.target} is not a text field")
            _resolved_value(action, profile)
        elif action.type == "select":
            if element.role != "combobox":
                raise PlanValidationError(f"Target {action.target} is not a dropdown")
            if not action.option_value:
                raise PlanValidationError("select requires option_value")
            if element.options and not any(
                option.value == action.option_value for option in element.options
            ):
                raise PlanValidationError(
                    f"Option {action.option_value!r} is unavailable for {action.target}"
                )
        elif action.type == "set_checked":
            if element.role not in {"checkbox", "radio"}:
                raise PlanValidationError(f"Target {action.target} is not a checkbox or radio")
            if action.checked is None:
                raise PlanValidationError("set_checked requires checked")
        elif action.type == "upload":
            if element.input_type != "file":
                raise PlanValidationError(f"Target {action.target} is not a file input")
            if not action.asset_id or action.asset_id not in assets:
                raise PlanValidationError(f"Unregistered upload asset: {action.asset_id}")
            if action.asset_id not in candidate_asset_ids(profile):
                raise PlanValidationError(
                    f"Upload asset is not referenced by the candidate profile: {action.asset_id}"
                )
        elif action.type == "click":
            if element.role not in {
                "button",
                "link",
                "option",
                "checkbox",
                "radio",
                "combobox",
                "generic",
            }:
                raise PlanValidationError(f"Target {action.target} is not clickable")
            if looks_like_submit(element.label):
                if not allow_submission:
                    raise PlanValidationError("Final submission is disabled; rerun with --no-submit off")
                if snapshot.validation_errors:
                    raise PlanValidationError(
                        "Cannot submit while visible validation errors remain"
                    )
                unresolved = unresolved_required(snapshot)
                if unresolved:
                    raise PlanValidationError(
                        "Required fields are unresolved: " + ", ".join(unresolved[:8])
                    )
