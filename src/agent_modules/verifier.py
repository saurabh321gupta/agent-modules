"""Proves an application was actually submitted. Layer 1.

Clicking submit is not evidence; a page that says so is. Nothing else in the system is allowed to
declare success.
"""

from __future__ import annotations

import re

from .types import CompletionEvidence, PageSnapshot

SUCCESS_PATTERNS = (
    r"application\s+(?:has been|was|is)\s+submitted",
    r"application submitted successfully",
    r"your application (?:has been )?received",
    r"we have received your application",
    r"thank you for applying",
    r"successfully applied",
)

REFERENCE_PATTERN = re.compile(
    r"(?:reference|confirmation|application)\s*(?:number|id|#)?\s*[:#]?\s*([A-Z0-9-]{4,})",
    re.I,
)


def verify_completion(snapshot: PageSnapshot) -> CompletionEvidence | None:
    """Return evidence if this page visibly confirms receipt, else None."""
    text = " ".join(snapshot.visible_text)
    for pattern in SUCCESS_PATTERNS:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            ref_match = REFERENCE_PATTERN.search(text)
            return CompletionEvidence(
                text=match.group(0),
                url=snapshot.url,
                reference=ref_match.group(1) if ref_match else None,
            )
    return None
