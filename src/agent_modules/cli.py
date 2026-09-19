"""Command line entry point. Layer 6.

Flags are translated into a `RunConfig` and nothing else; no behaviour lives here, so the whole run
can be tested by constructing a config directly.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

from .config import DEFAULT_BASE_URL, DEFAULT_JEV_MODEL, DEFAULT_MODEL, DEFAULT_ZOOM, AutomationDefaults, RunConfig
from .files import load_answer_bank, parse_asset, read_api_key
from .orchestrator import run_application
from .types import CandidateProfile

__all__ = [
    "build_parser",
    "config_from_args",
    "load_answer_bank",
    "main",
    "parse_asset",
    "read_api_key",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a generic browser-based job application agent")
    parser.add_argument("--url", required=True, help="External application URL")
    parser.add_argument("--profile", required=True, help="Candidate profile JSON file")
    parser.add_argument("--asset", action="append", type=parse_asset, default=[], help="Registered upload: asset_id=/absolute/path")
    parser.add_argument("--defaults", help="JSON file of pre-approved question/answer pairs sent to the planner alongside the profile")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model for the planner and the normaliser")
    parser.add_argument(
        "--planner",
        choices=["llm", "jev", "staged"],
        default="staged",
        help=(
            "'staged' classifies the page with Jev then normalises and answers only form pages "
            "(default); 'llm' is the single generative planner; 'jev' is the Jev per-field planner"
        ),
    )
    parser.add_argument("--jev-api-key-file", help="Jev API key file, required by the staged and jev planners")
    parser.add_argument("--jev-model", default=DEFAULT_JEV_MODEL)
    parser.add_argument("--jev-confidence", type=float, default=0.35, help="Minimum confidence for a Jev answer to be acted on (default 0.35)")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key-file", help="Read the provider API key from a local file at runtime")
    parser.add_argument("--headless", action="store_true", help="Hide the browser window")
    parser.add_argument("--field-pause", type=float, default=0.0, metavar="SECONDS", help="Pause after typing into a field and before leaving it, to let the site's own validation run (default 0)")
    parser.add_argument("--zoom", type=float, default=DEFAULT_ZOOM, help="Starting browser zoom; the agent zooms out further when a page is taller than the window. 1.0 disables")
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--max-seconds", type=float, default=300.0, help="Hard budget for one application; the run is discarded rather than submitted if it is not done within this (default 300s)")
    parser.add_argument(
        "--effort",
        choices=["default", "low", "medium", "high"],
        default="low",
        help="Planner reasoning effort; 'default' leaves it to the provider (default: low, substantially faster)",
    )
    parser.add_argument("--no-submit", action="store_false", dest="allow_submit", help="Do not click the final submission control (submission happens by default)")
    parser.add_argument("--no-default-consents", action="store_true", help="Do not automatically accept matching consent/privacy checkboxes")
    parser.add_argument("--output", help="Write the run result to JSON")
    parser.add_argument("--journey-log", help="Write the detailed journey JSONL to this path and a readable .log beside it (default: run-artifacts/journey-<timestamp>-<id>.jsonl)")
    return parser


def config_from_args(args: argparse.Namespace) -> RunConfig:
    return RunConfig(
        model=args.model,
        base_url=args.base_url,
        api_key=read_api_key(args.api_key_file) if args.api_key_file else None,
        headless=args.headless,
        field_pause_s=args.field_pause,
        max_steps=args.max_steps,
        allow_submission=args.allow_submit,
        zoom=args.zoom,
        answers=load_answer_bank(args.defaults) if args.defaults else {},
        answers_path=args.defaults,
        planner=args.planner,
        jev_api_key=read_api_key(args.jev_api_key_file) if args.jev_api_key_file else None,
        jev_model=args.jev_model,
        jev_confidence_floor=args.jev_confidence,
        defaults=AutomationDefaults(accept_matching_consents=not args.no_default_consents),
        journey_log_path=args.journey_log,
        max_run_seconds=args.max_seconds,
        reasoning_effort=None if args.effort == "default" else args.effort,
    )


async def _run(args: argparse.Namespace) -> int:
    """Read the inputs, then run.

    Reading the profile and the config files is the step a user is most likely to get wrong, so a bad
    input is reported as a sentence rather than as a traceback.
    """
    try:
        with open(args.profile, encoding="utf-8") as handle:
            profile = CandidateProfile.model_validate_json(handle.read())
        config = config_from_args(args)
    except (argparse.ArgumentTypeError, OSError, ValueError) as exc:
        print(f"Cannot start this run: {exc}", file=sys.stderr)
        return 1
    result = await run_application(args.url, profile, dict(args.asset), config)
    encoded = json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=False)
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(encoded + "\n")
    print(encoded)
    return 0 if result.status == "success" else 1


def main() -> None:
    args = build_parser().parse_args()
    try:
        raise SystemExit(asyncio.run(_run(args)))
    except KeyboardInterrupt:
        raise SystemExit(130)


if __name__ == "__main__":
    main()
