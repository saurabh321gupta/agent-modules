"""Questions, criteria and system prompts for Jev / System One. Layer 2.

Jev answers constrained questions - one Choice over a fixed criteria map, or one 0..1 score - so
almost everything here is about shaping a form into questions it can answer without ever being asked
to generate text. The single exception is the fallback prompt used when a dropdown has more options
than a Choice can carry.
"""

from __future__ import annotations

import re
from typing import Any

from .config import MAX_CHOICES
from .extractor import is_actionable
from .models import CandidateProfile, PageElement, PageSnapshot

#: How many answer-bank entries are offered for one field. Every profile fact is offered; the longer
#: answer bank is shortlisted, or the criteria map would be enormous.
SHORTLIST = 8

SUBMIT_MARKERS = (
    "submit application",
    "send application",
    "finish application",
    "complete application",
    "complete and submit",
    "submit",
    "final submit",
)
ADVANCE_MARKERS = ("next", "continue", "save and continue", "proceed", "review")
APPLY_MARKERS = (
    "apply",
    "apply now",
    "apply for this job",
    "start application",
    "begin application",
    "easy apply",
    "continue application",
    "resume application",
)

CLASSIFY_PROMPT = """You look at one page of a job application website and say what kind of page it is.

Choose exactly one class:
- job_listing: a job advert, search result or careers page that shows a role and offers a control to
  start or continue an application.
- job_review: the final review screen of an application, where the entered answers are shown back
  before the application is sent.
- application_submitted: a page confirming the application was received or submitted. Confirmation
  text, a reference number or a thank-you message.
- application_form: a step of the application where the candidate enters information - fields,
  dropdowns, checkboxes.
- captcha: a CAPTCHA or human-verification challenge.
- login: a sign-in, account creation or email-verification wall.
- other: none of the above.

Also choose which single control the candidate should press next. On an advert that is the control
that starts the application. On a form step it is the control that moves to the next step. Never
choose a button that merely signs up for job alerts, a newsletter or a search.

And say whether the page offers a way to upload a resume, naming the control that would receive the
file. Uploading comes before anything else on a page: a wizard step cannot be answered until the
resume it is waiting for has arrived, and the site often fills fields in from it.

And say whether the page visibly confirms the application was already received."""

PAGE_CLASS_CRITERIA = {
    "job_listing": "A job advert or careers page offering a control to start an application",
    "job_review": "The final review of a completed application, shown back before sending",
    "application_submitted": "A confirmation that the application was received or submitted",
    "application_form": "A step where the candidate enters application information",
    "captcha": "A CAPTCHA or human verification challenge",
    "login": "A sign-in, account creation or email verification wall",
    "other": "None of the above",
}

NEXT_CONTROL_INSTRUCTIONS = (
    "Which single control should the candidate press next? On an advert choose the "
    "control that starts the application. On a form step choose the one that moves "
    "forward. Never choose a control that only signs up for job alerts, a newsletter "
    "or a search. Never choose a control that uploads a file."
)

RESUME_UPLOAD_INSTRUCTIONS = (
    "Does this page offer a way to upload a resume? Answer 'yes' only when the page carries a control "
    "that accepts a file for the candidate's CV. A page that merely mentions a resume in its text "
    "does not offer one, and a file control asking for something else - a cover letter, a transcript, "
    "a portfolio - is not a resume upload."
)

RESUME_CONTROL_INSTRUCTIONS = (
    "Which control receives the resume file? Name the file input itself - the element the browser "
    "would actually be handed the file - not the styled button in front of it. Prefer a control whose "
    "wording mentions a resume or CV over one that does not, and use 'none' if none of them does."
)

FORM_PAGE_KINDS = {
    "form": "An application form with fields the candidate must complete",
    "job_listing": (
        "A job advert, search result or careers page that shows a role and offers a control "
        "to start or continue an application"
    ),
    "confirmation": "A page confirming the application was received or submitted",
    "captcha": "A CAPTCHA or human verification challenge",
    "login": "A sign-in, account creation or email verification wall",
    "other": "None of the above",
}

LARGE_OPTION_SYSTEM = """You pick exactly one option for a single form dropdown.

You are given the field, the candidate's known facts, and the complete list of options.
Choose the option whose meaning best matches the candidate's answer. Match on meaning, not
spelling: "Computer Science" should select "Computer and Information Science" or "Computer
Engineering" if that is the closest thing offered.

Return only a JSON object of the shape {"option_value": "<value>", "reason": "<short reason>"}.
option_value must be copied exactly from the option list. If nothing fits, return
{"option_value": null, "reason": "<why>"}. Never invent a value that is not in the list."""

REPAIR_SYSTEM = """You fix exactly one form field that the site has rejected.

The site showed an error for this field. You are given what the site says, what was written, the
field's own description, and the candidate's known facts. Return a value that satisfies the site's
stated requirement while staying faithful to the candidate's real data.

Return only a JSON object of the shape {"value": "<corrected value>", "reason": "<short reason>"}.
Return {"value": null, "reason": "<why>"} when no faithful correction is possible."""

STOPWORDS = {
    "the", "and", "for", "with", "your", "you", "are", "any", "this", "that", "have", "has",
    "please", "select", "enter", "field", "name", "address",
}


def tokens(text: str) -> set[str]:
    return {part for part in re.split(r"[^a-z0-9]+", (text or "").lower()) if len(part) > 2}


def similarity(label: str, candidate: str) -> float:
    """Token overlap, used only to pick which approved answers to offer. Never to choose an answer."""
    left = tokens(label) - STOPWORDS
    right = tokens(candidate) - STOPWORDS
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def profile_facts(profile: CandidateProfile) -> dict[str, str]:
    """Flatten the profile to dotted-path -> text, so a Choice can name the exact source it picked."""
    facts: dict[str, str] = {}

    def walk(value: Any, prefix: str) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                walk(item, f"{prefix}.{key}" if prefix else key)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                walk(item, f"{prefix}.{index}")
        elif value is not None and value != "":
            facts[prefix] = str(value)

    walk(profile.model_dump(mode="python"), "")
    return facts


def shortlist(label: str, candidates: dict[str, str], limit: int = SHORTLIST) -> dict[str, str]:
    ranked = sorted(candidates.items(), key=lambda item: similarity(label, item[1]), reverse=True)
    return dict(ranked[:limit])


# --------------------------------------------------------------------------- page classification


def classify_controls(snapshot: PageSnapshot) -> list[dict[str, Any]]:
    """The controls the classifier is shown, including a file input that still needs a file.

    A file input is offered even when it is hidden, because the real one usually is - a styled button
    stands in front of it. It is left out once it holds a file, since there is then nothing to do
    with it, which is also what keeps the resume questions off every later page of a wizard whose
    upload widget stays mounted in the page shell.
    """
    return [
        {
            "id": element.id,
            "role": element.role,
            "label": element.label,
            "input_type": element.input_type,
            "enabled": element.enabled,
        }
        for element in snapshot.elements
        if is_actionable(element) and element.role in ("button", "combobox", "textbox")
    ]


def classify_questions(
    buttons: list[dict[str, Any]],
    file_inputs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    questions: dict[str, Any] = {
        "page_class": {
            "type": "choice",
            "instructions": "What kind of page is this?",
            "criteria": dict(PAGE_CLASS_CRITERIA),
        },
        "submitted_evidence": {
            "type": "noul",
            "instructions": (
                "Does this page visibly confirm that an application was already received or "
                "submitted - confirmation text, a reference number, a thank-you message?"
            ),
        },
    }
    # Asked on every page, before the class is even considered: an upload is the one thing that has
    # to happen before a wizard step can be answered, and the control that takes the file is often
    # invisible, so a model reading the page is better placed to find it than any selector rule.
    if file_inputs:
        questions["has_resume_upload"] = {
            "type": "choice",
            "instructions": RESUME_UPLOAD_INSTRUCTIONS,
            "criteria": {
                "yes": "This page offers a way to upload a resume",
                "no": "It does not, or the control accepts some other document",
            },
        }
        questions["resume_upload_control"] = {
            "type": "choice",
            "instructions": RESUME_CONTROL_INSTRUCTIONS,
            "criteria": {c["id"]: (c["label"] or c["id"]) for c in file_inputs},
        }
    if buttons:
        questions["next_control"] = {
            "type": "choice",
            "instructions": NEXT_CONTROL_INSTRUCTIONS,
            "criteria": {c["id"]: (c["label"] or c["id"]) for c in buttons},
        }
    return questions


def classify_state(snapshot: PageSnapshot, controls: list[dict[str, Any]]) -> dict[str, Any]:
    """Deliberately small: classification should cost a fraction of an answering call."""
    return {
        "url": snapshot.url,
        "title": snapshot.title,
        "visible_text": snapshot.visible_text[:20],
        "controls": controls[:60],
        "validation_errors": snapshot.validation_errors[:5],
    }


# ------------------------------------------------------------------------------- field questions


def option_criteria(element: PageElement) -> dict[str, str | None]:
    """A Choice's criteria for a dropdown. The empty placeholder is never an answer, so it is dropped."""
    options = [option for option in element.options if option.value != ""][:MAX_CHOICES]
    criteria: dict[str, str | None] = {option.value: option.label or None for option in options}
    criteria["none"] = "No option fits, or this field should be left alone"
    return criteria


def build_field_questions(
    profile: CandidateProfile,
    snapshot: PageSnapshot,
    answer_bank: dict[str, str],
) -> tuple[dict[str, Any], dict[str, str], list[PageElement], list[PageElement]]:
    """One question per answerable control.

    Returns the questions, the question-key -> element-id map needed to compose a plan, the dropdowns
    too large to offer as a Choice, and the controls the site has already rejected.
    """
    facts = profile_facts(profile)
    bank = {f"answers.{key}": key for key in answer_bank}
    large_dropdowns: list[PageElement] = []
    invalid_elements: list[PageElement] = []

    questions: dict[str, Any] = {
        "page_kind": {
            "type": "choice",
            "instructions": "What kind of page is this?",
            "criteria": dict(FORM_PAGE_KINDS),
        },
        "submit_ready": {
            "type": "noul",
            "instructions": (
                "Is every required field on this page currently filled in, with no visible validation "
                "error? Answer about the state of the form only, not about whether a submit button "
                "is present."
            ),
        },
    }
    element_by_question: dict[str, str] = {}

    for element in snapshot.elements:
        if not element.enabled or not element.visible:
            continue
        if element.input_type in ("file", "password"):
            continue
        if element.invalid:
            # The site has already rejected this value; re-asking "which fact?" cannot fix a format.
            invalid_elements.append(element)
            continue
        key = f"state_{element.id}"
        question = f"{element.label or element.id} (role={element.role}, type={element.input_type})"
        if element.role == "combobox":
            if len(element.options) > MAX_CHOICES:
                # Too many options for a Jev Choice; the LLM resolver handles this one instead.
                large_dropdowns.append(element)
                continue
            questions[key] = {
                "type": "choice",
                "instructions": (
                    f"Which option best answers the form field {question}? Read the candidate profile and "
                    "approved answers: pick the option whose meaning matches, even if the wording differs. "
                    "Use 'none' if nothing fits or the field is optional and already correct."
                ),
                "criteria": option_criteria(element),
            }
            element_by_question[key] = element.id
        elif element.role == "textbox":
            # Every profile fact is offered (there are few, and they are the highest-value sources);
            # the longer answer bank is shortlisted against this field's label. Dotted paths describe
            # themselves, so they carry no description; only the answer bank needs its question text.
            criteria: dict[str, str | None] = {path: None for path in facts}
            criteria.update(shortlist(element.label or element.id, bank, limit=SHORTLIST))
            criteria["none"] = "No candidate fact answers this field"
            questions[key] = {
                "type": "choice",
                "instructions": (
                    f"Which candidate fact answers the form field {question}? "
                    "Names are dotted paths such as location.city; answers.* are approved answers. "
                    "Match on meaning, not spelling. "
                    "Only answer with a fact when this control is part of the job application form. "
                    "Use 'none' for anything that is not an application field - job-alert signups, "
                    "newsletter boxes, search boxes, filters and page furniture - and for a field that "
                    "already holds the right value."
                ),
                "criteria": criteria,
            }
            element_by_question[key] = element.id
        elif element.role in ("checkbox", "radio") and element.checked is not True:
            questions[key] = {
                "type": "noul",
                "instructions": (
                    f"Should the candidate tick the box or select the option {question}? "
                    "Only say yes when the profile or the approved answers support it; a consent, "
                    "privacy or recruitment notice may be accepted."
                ),
            }
            element_by_question[key] = element.id

    buttons = [e for e in snapshot.elements if e.role == "button" and e.enabled and e.visible]
    if buttons:
        questions["advance"] = {
            "type": "choice",
            "instructions": (
                "Which single control should the candidate press next? Prefer the control that starts "
                "or continues the job application - 'Apply', 'Apply Now', 'Apply for this job' - or, on "
                "an application form, the one that moves to the next step ('Next', 'Continue'). A "
                "'Submit' or 'Sign up' button sitting next to a newsletter, job-alert or search box is "
                "not the application and must never be chosen. Use 'none' only if no control makes sense."
            ),
            "criteria": {e.id: (e.label or e.id) for e in buttons},
        }
        element_by_question["advance"] = "advance"
    return questions, element_by_question, large_dropdowns, invalid_elements
