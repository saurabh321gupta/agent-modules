"""Which routine declarations the agent may answer on the candidate's behalf. Layer 1.

Deliberately narrow. Only consent-style checkboxes are auto-accepted; everything that asserts a fact
about the candidate is left to the planner and, if it cannot be grounded, to `needs_input`.
"""

from __future__ import annotations

from .config import AutomationDefaults
from .types import Action, PageElement, PageSnapshot

CONSENT_MARKERS = (
    "consent",
    "agree to the processing",
    "recruitment notice",
    "privacy notice",
)


def is_matching_consent(element: PageElement) -> bool:
    if element.role != "checkbox":
        return False
    label = element.label.lower()
    return any(marker in label for marker in CONSENT_MARKERS)


def default_consent_actions(
    snapshot: PageSnapshot,
    defaults: AutomationDefaults,
) -> list[Action]:
    """Checkboxes this run may tick without asking. Already-ticked boxes are not repeated."""
    if not defaults.accept_matching_consents:
        return []
    return [
        Action(
            type="set_checked",
            target=element.id,
            value=None,
            value_ref=None,
            option_value=None,
            checked=True,
            asset_id=None,
        )
        for element in snapshot.elements
        if is_matching_consent(element) and element.enabled and element.checked is not True
    ]
