"""Closes obstructive page furniture before observation. Layer 2 (browser-bound).

Best-effort by design. A cookie banner or a chat widget can intercept clicks on real controls, so it
is worth clearing; but a mistake here is worse than the problem, so anything resembling the
application form itself is left strictly alone.
"""

from __future__ import annotations

from typing import Any, Callable

from playwright.async_api import Page

NONESSENTIAL_OVERLAY_MARKERS = (
    "cookie",
    "cookies",
    "privacy preference",
    "consent preference",
    "chatbot",
    "career bot",
    "live chat",
    "chat window",
    "chat widget",
    "survey",
    "feedback",
    "newsletter",
)

#: If any of these appear, the "overlay" is the application and must never be dismissed.
ESSENTIAL_FORM_MARKERS = (
    "given name",
    "family name",
    "email address",
    "phone number",
    "upload resume",
    "application form",
)

CLOSE_LABEL_MARKERS = ("close", "dismiss", "not now", "no thanks", "decline")

COOKIE_ACCEPT_LABELS = {"accept all", "accept all cookies", "allow all cookies"}


async def dismiss_nonessential_overlays(page: Page) -> list[dict[str, str]]:
    """Close safe, obstructive overlays without touching required application controls."""
    actions: list[dict[str, str]] = []

    # Portals such as HPE expose a chatbot close control outside a dialog root.
    for _ in range(4):
        closed = await click_matching_control(
            page,
            lambda label: "close" in label
            and any(marker in label for marker in ("chat", "bot", "conversation")),
        )
        if not closed:
            break
        actions.append({"kind": "chat", "label": closed})

    body_text = (await page.locator("body").inner_text()).lower()
    if any(marker in body_text for marker in ("cookie", "cookies")):
        accepted = await click_matching_control(page, lambda label: label in COOKIE_ACCEPT_LABELS)
        if accepted:
            actions.append({"kind": "cookie", "label": accepted})

    roots = page.locator(
        '[role="dialog"], [aria-modal="true"], [class*="modal"], [class*="popup"], [class*="overlay"]'
    )
    for index in range(await roots.count()):
        root = roots.nth(index)
        try:
            if not await root.is_visible():
                continue
            text = (await root.inner_text()).lower()
            class_name = (await root.get_attribute("class") or "").lower()
            context = f"{class_name} {text}"
            if not any(marker in context for marker in NONESSENTIAL_OVERLAY_MARKERS):
                continue
            if any(marker in context for marker in ESSENTIAL_FORM_MARKERS):
                continue
            close_controls = root.locator("button, [role='button'], a")
            for control_index in range(await close_controls.count()):
                control = close_controls.nth(control_index)
                label = await control_label(control)
                if not label or not any(marker in label for marker in CLOSE_LABEL_MARKERS):
                    continue
                await control.click(timeout=2_000)
                actions.append({"kind": "dialog", "label": label[:160]})
                break
        except Exception:
            # Cleanup is best-effort. The normal observer/planner path remains authoritative.
            continue
    return actions


async def click_matching_control(page: Page, predicate: Callable[[str], bool]) -> str | None:
    controls = page.locator("button, [role='button'], a")
    for index in range(await controls.count()):
        control = controls.nth(index)
        try:
            if not await control.is_visible() or not await control.is_enabled():
                continue
            label = await control_label(control)
            if not label or not predicate(label.lower()):
                continue
            await control.click(timeout=2_000)
            return label[:160]
        except Exception:
            continue
    return None


async def control_label(control: Any) -> str:
    parts = [
        await control.get_attribute("aria-label"),
        await control.get_attribute("title"),
        await control.text_content(),
    ]
    return " ".join(part.strip() for part in parts if part and part.strip())
