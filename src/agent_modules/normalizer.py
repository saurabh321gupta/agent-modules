"""Turns a page's controls into canonical questions. Layer 4.

Sits between the reader and the planner, and exists because "what is this field asking?" is reading
comprehension, not a rule. The heuristic label chain resolves most controls but cannot resolve a
label that lives in a sibling div, a `fieldset` legend, or an `aria-labelledby` target - a model
reading the surrounding DOM can.

The result is cached per form *shape*. Filling a field changes its value but not what it is asking,
so questions hold for as long as a candidate is on the page: one model call per form, not per step.
"""

from __future__ import annotations

import json
import time
from typing import Any

from .extractor import describe, field_type_for, form_signature, is_capturable, page_kind_of
from .journey import JourneyLogger
from .llm_client import ModelClient
from .models import NormalisedField, NormalisedForm, PageSnapshot

SYSTEM_PROMPT = """You read a web form and say, for each control, exactly what it is asking the applicant.

You are given the controls a browser can see, with the id, tag, input type, the label the page
exposes, the group a control belongs to, any help text, the current value, and the full list of a
dropdown's options.

For every control you return:
- id: copied exactly from the input. Never invent an id.
- question: what the applicant is being asked, faithful to the form's own wording. If a control
  belongs to a group (a radio or checkbox group, a repeated block), the question is the group's
  question, not the individual option's text.
- field_type: one of text, long_text, email, phone, number, date, select, multi_select, checkbox,
  radio, file, button, unknown.
- required: true when the form genuinely requires an answer - the control is marked required, has
  aria-required, or its label carries a required marker such as an asterisk.
- group: the section or repeated block it belongs to, such as "Work experience #2" or "Address",
  else null. Numbering repeated blocks is important: three rows of From/To dates are three
  different questions.

Rules:
- Describe only. Do not answer the question, do not suggest a value, do not fill anything in.
- Use the surrounding context. A label may be a neighbouring element, a placeholder, a legend, or
  the target of aria-labelledby rather than the control's own text.
- A file control uploads a document. Describe what it is asking for from the surrounding text, and
  give it field_type "file". Never omit it because its label is empty or because it is not visible:
  a site normally hides the real file input behind a styled button, so the control the applicant
  sees is not the control that receives the file. "visible": false is normal for these.
- Buttons are usually not questions for the applicant. For a button, put the button's own label in
  question so it can be told apart (for example "Apply Now"), and use "" only when the control has
  no label at all. field_type should be "button".
- submit_controls must contain only the control that actually submits the completed application -
  the final Submit/Send/Finish control on an application form. Do not list navigation, Next,
  Apply, Save, cookie-consent, alert or search controls. An advert page usually has no submit
  control at all: return an empty list.
- Classify the page as one of: form, job_listing, confirmation, captcha, login, other.

Options are taken from the page itself, so you do not need to return them.

Return only JSON of the shape
{"page_kind": "form", "fields": [{"id": "e1", "question": "...", "field_type": "text",
"required": true, "group": null}], "submit_controls": ["e9"]}"""


class Normalizer:
    """Turns a page's controls into canonical questions, with the exact payloads logged."""

    def __init__(self, client: ModelClient, model: str, journey: JourneyLogger | None = None) -> None:
        self.client = client
        self.model = model
        self.journey = journey
        self._cache: dict[str, NormalisedForm] = {}

    async def normalise(self, snapshot: PageSnapshot) -> NormalisedForm:
        key = form_signature(snapshot)
        cached = self._cache.get(key)
        if cached is not None:
            if self.journey:
                self.journey.log(
                    "normalised_form",
                    cached=True,
                    page_kind=cached.page_kind,
                    field_count=len(cached.fields),
                    form=self.summarise(cached),
                )
            return cached.model_copy(update={"cached": True})

        controls = describe(snapshot)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    {"page": {"url": snapshot.url, "title": snapshot.title}, "controls": controls},
                    ensure_ascii=False,
                ),
            },
        ]
        if self.journey:
            self.journey.log(
                "normalise_request",
                request={"model": self.model, "messages": messages, "response_format": {"type": "json_object"}},
            )

        completion = await self.client.complete(
            messages=messages,
            response_format={"type": "json_object"},
            model=self.model,
        )
        if self.journey:
            self.journey.log(
                "normalise_response",
                model=self.model,
                latency_s=round(completion.latency_s, 2),
                raw_response=completion.text,
                prompt_tokens=completion.prompt_tokens,
                completion_tokens=completion.completion_tokens,
                cached_tokens=completion.cached_tokens,
            )

        form = self.parse(snapshot, completion.text)
        self._cache[key] = form
        if self.journey:
            self.journey.log(
                "normalised_form",
                cached=False,
                page_kind=form.page_kind,
                field_count=len(form.fields),
                dropped_ids=form.dropped_ids,
                latency_s=round(completion.latency_s, 2),
                form=self.summarise(form),
            )
        return form

    def parse(self, snapshot: PageSnapshot, raw: str) -> NormalisedForm:
        """Build a form from a model reply, keeping only ids the page actually has."""
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = {}
        if not isinstance(data, dict):
            data = {}

        known = {element.id: element for element in snapshot.elements}
        fields: list[NormalisedField] = []
        dropped: list[str] = []
        seen: set[str] = set()
        for entry in data.get("fields") or []:
            if not isinstance(entry, dict):
                continue
            field_id = entry.get("id")
            if not isinstance(field_id, str) or field_id not in known or field_id in seen:
                dropped.append(str(field_id))
                continue
            seen.add(field_id)
            element = known[field_id]
            fields.append(
                NormalisedField(
                    id=field_id,
                    question=str(entry.get("question") or ""),
                    field_type=field_type_for(entry.get("field_type"), element.input_type, element.role),
                    required=bool(entry.get("required")),
                    group=(str(entry["group"]) if entry.get("group") else None),
                    # Options always come from the page, complete: the planner needs the real values
                    # (which are often codes), and a model's recollection is not authoritative.
                    options=list(element.options),
                    raw_label=element.label,
                    raw_input_type=element.input_type,
                )
            )

        # Any control the model skipped is still reported, so nothing silently disappears. This uses
        # the same filter as `describe`, so a control the model was never shown is also never lost.
        for element in snapshot.elements:
            if element.id in seen or not is_capturable(element):
                continue
            fields.append(
                NormalisedField(
                    id=element.id,
                    question=element.label,
                    field_type=field_type_for(None, element.input_type, element.role),
                    required=element.required,
                    options=list(element.options),
                    raw_label=element.label,
                    raw_input_type=element.input_type,
                )
            )

        submits = [
            sid for sid in (data.get("submit_controls") or []) if isinstance(sid, str) and sid in known
        ]
        return NormalisedForm(
            page_kind=page_kind_of(data.get("page_kind")),
            fields=fields,
            submit_controls=submits,
            dropped_ids=dropped,
        )

    @staticmethod
    def summarise(form: NormalisedForm) -> list[dict[str, Any]]:
        return [
            {
                "id": f.id,
                "question": f.question,
                "field_type": f.field_type,
                "required": f.required,
                "group": f.group,
                "raw_label": f.raw_label,
            }
            for f in form.fields
        ]


async def normalise(
    snapshot: PageSnapshot,
    client: ModelClient,
    model: str,
    journey: JourneyLogger | None = None,
) -> NormalisedForm:
    """One-shot convenience wrapper around a fresh Normalizer."""
    return await Normalizer(client, model, journey).normalise(snapshot)
