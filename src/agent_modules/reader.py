"""Turns a live page into a `PageSnapshot`. Layer 2 (browser-bound).

The only module that knows about the DOM. Everything downstream works on the snapshot, which is why
the planners can be tested without a browser at all.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import re
from typing import Any

from playwright.async_api import Page

from .types import Option, PageElement, PageSnapshot

#: An accessibility snapshot line reads `- role "Accessible Name": value`. The name is optional,
#: and quotes and backslashes inside it are escaped.
ARIA_SNAPSHOT_LINE = re.compile(
    r'^-\s+(?P<role>[a-z][a-z0-9-]*)\s*(?:"(?P<name>(?:[^"\\]|\\.)*)")?'
)

#: Roles worth reading an accessible name for. Options are excluded: they are the text inside a
#: combobox we already read in full, and there can be hundreds of them.
ARIA_NAMED_ROLES = frozenset(
    {"textbox", "combobox", "checkbox", "radio", "button", "switch", "searchbox", "spinbutton"}
)
ARIA_SNAPSHOT_TIMEOUT_MS = 2_000

#: Icon fonts put a private-use code point in the accessible name. It carries no meaning for a
#: reader, and it is invisible in every renderer, so it is removed before the name is used.
ICON_GLYPH = re.compile(
    "[\ue000-\uf8ff\U000f0000-\U000ffffd\U00100000-\U0010fffd\u200b-\u200f\u2060-\u206f\ufeff]"
)

#: The observer's own selector list. Anchors are included here and filtered by role later, so that
#: an anchor the page declares a button still works.
CAPTURE_SCRIPT = """
({snapshotId}) => {
  const isVisible = (el) => {
    const style = window.getComputedStyle(el);
    const rect = el.getBoundingClientRect();
    return style.display !== 'none' && style.visibility !== 'hidden' &&
      rect.width > 0 && rect.height > 0;
  };
  const clean = (value, max = 500) => (value || '').replace(/\\s+/g, ' ').trim().slice(0, max);
  const labelFor = (el) => {
    if (el.getAttribute('aria-label')) return clean(el.getAttribute('aria-label'));
    if (el.id) {
      const label = document.querySelector('label[for="' + CSS.escape(el.id) + '"]');
      if (label) return clean(label.innerText);
    }
    const parentLabel = el.closest('label');
    if (parentLabel) return clean(parentLabel.innerText);
    return clean(el.getAttribute('placeholder') || el.getAttribute('name') || el.innerText);
  };
  const describedBy = (el) => {
    const ids = (el.getAttribute('aria-describedby') || '').split(/\\s+/).filter(Boolean);
    return clean(ids.map(id => document.getElementById(id)?.innerText || '').join(' '), 300);
  };
  // A radio answers "Yes" to a question the option text never repeats, so the question has
  // to come from the enclosing group: a fieldset legend, or the radiogroup's own labelling.
  //
  // This is captured for choice controls only. Elsewhere the label already carries the
  // question, while portals routinely wrap ordinary sections in a fieldset whose legend is
  // a framework identifier - eBay's read "cntryFields" and "phoneWidget" - which would be
  // read as the question the applicant is answering.
  const groupFor = (el) => {
    const isChoice = ['radio', 'checkbox'].includes(el.type) ||
      ['radio', 'checkbox'].includes(el.getAttribute('role'));
    if (!isChoice) return '';
    const legend = el.closest('fieldset')?.querySelector('legend');
    if (legend) return clean(legend.innerText, 300);
    const group = el.closest('[role="radiogroup"], [role="group"]');
    if (!group) return '';
    const own = group.getAttribute('aria-label');
    if (own) return clean(own, 300);
    const ids = (group.getAttribute('aria-labelledby') || '').split(/\\s+/).filter(Boolean);
    return clean(ids.map(id => document.getElementById(id)?.innerText || '').join(' '), 300);
  };
  // Anchors are navigation, not application controls, and are never sent to a model.
  // An anchor the page itself declares a button is kept, so an apply control styled as a
  // link still works.
  const keepElement = (el) =>
    el.tagName.toLowerCase() !== 'a' || el.getAttribute('role') === 'button';
  const nodes = [...document.querySelectorAll(
    'input, textarea, select, button, a, [role="button"], [role="combobox"], [role="option"], [role="checkbox"], [role="radio"]'
  )].filter(el => isVisible(el) || (el.tagName.toLowerCase() === 'input' && el.type === 'file')).filter(keepElement);
  const elements = nodes.map((el, index) => {
    const tag = el.tagName.toLowerCase();
    const role = el.getAttribute('role') ||
      (tag === 'select' ? 'combobox' : tag === 'button' ? 'button' : tag === 'a' ? 'link' :
      (el.type === 'checkbox' ? 'checkbox' : el.type === 'radio' ? 'radio' :
      (tag === 'textarea' ? 'textbox' : tag === 'input' ? 'textbox' : 'generic')));
    const options = tag === 'select' ? [...el.options].map(o => ({value: o.value, label: clean(o.textContent)})) : [];
    let value = null;
    if (tag === 'textarea' || (tag === 'input' && !['file', 'password'].includes(el.type))) value = clean(el.value, 300);
    if (tag === 'select') value = el.value || null;
    return {
      id: 'e' + (index + 1),
      role,
      label: labelFor(el),
      input_type: el.type || null,
      required: !!el.required || el.getAttribute('aria-required') === 'true',
      value,
      enabled: !el.disabled && el.getAttribute('aria-disabled') !== 'true',
      checked: ['checkbox', 'radio'].includes(el.type) ? !!el.checked : null,
      options,
      help_text: describedBy(el) || null,
      group: groupFor(el) || null,
      tag,
      visible: isVisible(el),
      file_attached: el.type === 'file' ? !!el.files?.length : false,
      invalid: el.getAttribute('aria-invalid') === 'true',
      readonly: !!el.readOnly,
    };
  });
  const bodyText = (document.body?.innerText || '').split(/\\n+/)
    .map(line => clean(line, 500)).filter(Boolean).slice(0, 500).join('\\n').slice(0, 12000);
  const flagged = nodes
    .filter(el => el.getAttribute('aria-invalid') === 'true')
    .map(el => 'invalid field: ' + (labelFor(el) || el.getAttribute('name') || el.id));
  const validationErrors = [...new Set([
    ...[...document.querySelectorAll(
      '[role="alert"], [aria-live="assertive"], .error, .errors, [class*="error"], [class*="invalid"]'
    )].filter(isVisible).map(el => clean(el.innerText, 300)).filter(Boolean),
    ...flagged,
  ])].slice(0, 30);
  nodes.forEach((el, i) => {
    el.setAttribute('data-agent-snapshot', snapshotId);
    el.setAttribute('data-agent-id', 'e' + (i + 1));
  });
  return {elements, bodyText, validationErrors};
}
"""


def parse_accessible_name(raw: str) -> str:
    """Pull the accessible name out of an accessibility snapshot for a single element."""
    match = ARIA_SNAPSHOT_LINE.match(raw.lstrip())
    if not match or match.group("name") is None:
        return ""
    name = match.group("name").replace('\\"', '"').replace("\\\\", "\\")
    name = ICON_GLYPH.sub(" ", name)
    return re.sub(r"\s+", " ", name).strip()[:300]


def find_blockers(text: str) -> list[str]:
    """Walls that no amount of planning gets past, detected from the visible text."""
    lowered = text.lower()
    blockers: list[str] = []
    if any(
        term in lowered
        for term in ("captcha", "recaptcha", "i'm not a robot", "verify you are human")
    ):
        blockers.append("CAPTCHA or human verification is present")
    if any(
        term in lowered
        for term in ("sign in to apply", "log in to apply", "create an account to apply")
    ):
        blockers.append("Authentication or account creation is required")
    if "verify your email" in lowered or "email verification" in lowered:
        blockers.append("Email verification is required")
    return blockers


def snapshot_fingerprint(snapshot: PageSnapshot) -> str:
    """Identifies the page's *state* including values. Compared to detect a run that is not moving."""
    compact = snapshot.model_dump(exclude={"snapshot_id"})
    return json.dumps(compact, sort_keys=True, separators=(",", ":"))


class PageReader:
    """Extract a compact, accessibility-oriented representation of the live DOM."""

    def __init__(self) -> None:
        self._sequence = itertools.count(1)

    async def capture(self, page: Page) -> PageSnapshot:
        title = await page.title()
        snapshot_id = f"s-{next(self._sequence)}-" + hashlib.sha1(
            f"{page.url}:{title}".encode(), usedforsecurity=False
        ).hexdigest()[:8]
        raw: dict[str, Any] = await page.evaluate(CAPTURE_SCRIPT, {"snapshotId": snapshot_id})

        elements = [
            PageElement(
                id=item["id"],
                role=item["role"],
                label=item["label"],
                group=item["group"],
                input_type=item["input_type"],
                required=item["required"],
                value=item["value"],
                enabled=item["enabled"],
                checked=item["checked"],
                options=[Option(**option) for option in item["options"]],
                help_text=item["help_text"],
                visible=item["visible"],
                file_attached=item["file_attached"],
                invalid=item["invalid"],
                readonly=item["readonly"],
            )
            for item in raw["elements"]
        ]
        await self._apply_accessible_names(page, snapshot_id, elements)
        return PageSnapshot(
            snapshot_id=snapshot_id,
            url=page.url,
            title=title,
            visible_text=[line for line in raw["bodyText"].split("\n") if line.strip()][:250],
            elements=elements,
            validation_errors=raw["validationErrors"],
            blockers=find_blockers(raw["bodyText"]),
        )

    @staticmethod
    async def _apply_accessible_names(
        page: Page, snapshot_id: str, elements: list[PageElement]
    ) -> None:
        """Replace our guessed label with the name the browser itself computes for the control.

        The accessibility tree resolves `aria-labelledby`, `title`, wrapping labels and every other
        rule of the accessible-name algorithm, and it agrees with what a screen reader announces.
        It is authoritative, so it wins whenever it has something to say. It also says nothing for a
        label that is merely *near* the control, which is the one case our own chain still covers -
        so the guess is kept as the fallback rather than thrown away.

        Reads are cheap (about 1.5 ms per control) and never fatal: a control that cannot be
        resolved simply keeps the guessed label.
        """
        for element in elements:
            if element.role not in ARIA_NAMED_ROLES:
                element.label_source = "heuristic" if element.label else "none"
                continue
            locator = page.locator(
                f'[data-agent-snapshot="{snapshot_id}"][data-agent-id="{element.id}"]'
            )
            try:
                name = parse_accessible_name(
                    await locator.aria_snapshot(timeout=ARIA_SNAPSHOT_TIMEOUT_MS)
                )
            except Exception:
                name = ""
            if name:
                element.label = name
                element.label_source = "aria"
            else:
                element.label_source = "heuristic" if element.label else "none"
