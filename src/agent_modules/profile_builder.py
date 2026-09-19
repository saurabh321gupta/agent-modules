"""Offline tooling: build a `profile.json` from a resume. Layer 6.

Not part of the run loop. It exists so the profile a run depends on can be regenerated from the
document the candidate actually maintains, rather than hand-edited and drifting from the resume.

Every field is required and nullable, and the prompt says so explicitly: a missing fact must arrive
as null, because "unknown" and "no" are very different answers to a screening question.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

from .config import DEFAULT_BASE_URL, DEFAULT_MODEL
from .files import read_api_key
from .llm_client import DeepSeekClient
from .types import CandidateProfile

SYSTEM_PROMPT = """
You extract a structured candidate profile from a resume.
Return only the requested structured data.

Use only facts explicitly present in the resume. When a value is not stated, use null. Never invent,
infer, or embellish: no guessed contact details, no assumed dates, no manufactured role summaries.
Copy values close to verbatim and keep phone numbers exactly as written, including the country code.
total_years is total professional experience in years as a number, computed from the listed dates,
or null when the dates do not support a computation.
skills lists concrete technologies, tools, and languages named in the resume, at most 40 entries.
roles lists the positions found in the resume, most recent first.
authorized_countries lists only countries where the resume explicitly states work authorization;
return an empty list when the resume says nothing about it.
referral_source is a self-reported answer to "how did you hear about us"; a resume never contains it,
so always return null unless the resume literally states it.
""".strip()


def _nullable(type_name: str, description: str) -> dict[str, Any]:
    return {"type": [type_name, "null"], "description": description}


PROFILE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "identity",
        "location",
        "experience",
        "education",
        "skills",
        "work_authorization",
        "preferences",
    ],
    "properties": {
        "identity": {
            "type": "object",
            "additionalProperties": False,
            "required": ["first_name", "last_name", "email", "phone", "linkedin", "headline"],
            "properties": {
                "first_name": _nullable("string", "Given name as printed on the resume."),
                "last_name": _nullable("string", "Family name as printed on the resume."),
                "email": _nullable("string", "Email address."),
                "phone": _nullable("string", "Phone number, including the country code, verbatim."),
                "linkedin": _nullable("string", "LinkedIn profile URL or handle."),
                "headline": _nullable("string", "Professional headline or current title as written."),
            },
        },
        "location": {
            "type": "object",
            "additionalProperties": False,
            "required": ["city", "state", "country", "postal_code"],
            "properties": {
                "city": _nullable("string", "City of residence."),
                "state": _nullable("string", "State or region of residence."),
                "country": _nullable("string", "Country of residence."),
                "postal_code": _nullable("string", "Postal or ZIP code."),
            },
        },
        "experience": {
            "type": "object",
            "additionalProperties": False,
            "required": ["total_years", "current_title", "current_company", "roles"],
            "properties": {
                "total_years": _nullable("number", "Total professional experience in years, or null."),
                "current_title": _nullable("string", "Title of the most recent role."),
                "current_company": _nullable("string", "Employer of the most recent role."),
                "roles": {
                    "type": "array",
                    "description": "Positions found in the resume, most recent first.",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["title", "company", "start", "end", "summary"],
                        "properties": {
                            "title": _nullable("string", "Job title."),
                            "company": _nullable("string", "Employer name."),
                            "start": _nullable("string", "Start date as written, e.g. 'Jan 2021'."),
                            "end": _nullable("string", "End date as written, or 'Present'."),
                            "summary": _nullable("string", "One short line of responsibilities, drawn from the resume."),
                        },
                    },
                },
            },
        },
        "education": {
            "type": "array",
            "description": "Degrees found in the resume, most recent first.",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["degree", "field", "institution", "graduation_year"],
                "properties": {
                    "degree": _nullable("string", "Degree name, e.g. 'B.Tech'."),
                    "field": _nullable("string", "Field of study."),
                    "institution": _nullable("string", "School or university name."),
                    "graduation_year": _nullable("string", "Graduation year as written."),
                },
            },
        },
        "skills": {
            "type": "array",
            "description": "Concrete technologies, tools, and languages named in the resume.",
            "items": {"type": "string"},
        },
        "work_authorization": {
            "type": "object",
            "additionalProperties": False,
            "required": ["authorized_countries"],
            "properties": {
                "authorized_countries": {
                    "type": "array",
                    "description": "Countries where the resume explicitly states work authorization; empty otherwise.",
                    "items": {"type": "string"},
                },
            },
        },
        "preferences": {
            "type": "object",
            "additionalProperties": False,
            "required": ["willing_to_relocate", "expected_salary", "referral_source"],
            "properties": {
                "willing_to_relocate": _nullable("boolean", "Only when the resume states it; otherwise null."),
                "expected_salary": _nullable("string", "Only when the resume states it; otherwise null."),
                "referral_source": _nullable("string", "Self-reported source for 'how did you hear about us'; normally null."),
            },
        },
    },
}


def extract_text(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return path.read_text(encoding="utf-8", errors="replace")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a candidate profile JSON from a resume using a model"
    )
    parser.add_argument("--resume", required=True, help="Resume file to read (.pdf, .txt, or .md)")
    parser.add_argument("--output", default="profile.json", help="Where to write the profile JSON (default: profile.json)")
    parser.add_argument("--asset-id", default="resume_primary", help="Asset id to register for the resume (default: resume_primary)")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key-file", help="Read the provider API key from a local file at runtime")
    return parser


def build_messages(resume_text: str, mode: str) -> list[dict[str, str]]:
    """The request messages for a given schema mode, so the prompt can be rebuilt on a fallback."""
    system = SYSTEM_PROMPT
    if mode != "json_schema":
        system = (
            SYSTEM_PROMPT
            + "\n\nReturn a single JSON object, with no prose and no markdown fences, matching this "
            "schema exactly:\n"
            + json.dumps(PROFILE_SCHEMA, ensure_ascii=False)
        )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Resume text:\n\n{resume_text}"},
    ]


async def extract_profile(client: DeepSeekClient, model: str, resume_text: str) -> dict[str, Any]:
    completion = await client.complete_json(
        build_messages=lambda mode: build_messages(resume_text, mode),
        schema=PROFILE_SCHEMA,
        schema_name="candidate_profile",
        model=model,
    )
    if not completion.text:
        raise RuntimeError("The model returned an empty profile")
    data = json.loads(completion.text)
    if not isinstance(data, dict):
        raise RuntimeError("The model did not return a JSON object")
    return data


async def _run(args: argparse.Namespace) -> int:
    resume = Path(args.resume).expanduser().resolve()
    if not resume.is_file():
        print(f"Resume not found: {resume}", file=sys.stderr)
        return 1
    try:
        resume_text = extract_text(resume)
    except Exception as exc:
        print(f"Could not read {resume}: {exc}", file=sys.stderr)
        return 1
    if len(resume_text.strip()) < 80:
        print(
            "The resume produced almost no readable text (a scanned image PDF needs OCR first).",
            file=sys.stderr,
        )
        return 1

    api_key = read_api_key(args.api_key_file) if args.api_key_file else None
    if not api_key:
        api_key = os.getenv("EXPLABS_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print(
            "No API key: pass --api-key-file or set EXPLABS_API_KEY / OPENAI_API_KEY", file=sys.stderr
        )
        return 1

    client = DeepSeekClient(api_key=api_key, base_url=args.base_url, model=args.model)
    try:
        data = await extract_profile(client, args.model, resume_text)
    except Exception as exc:
        print(f"Profile extraction failed: {exc}", file=sys.stderr)
        return 1

    data["documents"] = {
        "resume": {
            "asset_id": args.asset_id,
            "filename": resume.name,
        }
    }
    profile = CandidateProfile.model_validate(data)
    encoded = json.dumps(profile.model_dump(mode="json"), indent=2, ensure_ascii=False)

    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(encoded + "\n", encoding="utf-8")
    os.chmod(output, 0o600)
    print(encoded)
    print(f"\nWrote {output}", file=sys.stderr)
    return 0


def main() -> None:
    args = build_parser().parse_args()
    try:
        raise SystemExit(asyncio.run(_run(args)))
    except KeyboardInterrupt:
        raise SystemExit(130)


if __name__ == "__main__":
    main()
