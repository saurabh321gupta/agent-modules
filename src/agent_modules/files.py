"""Readers for the small files a run is configured from. Layer 1.

Separated from the CLI so that every entry point — the run and the profile builder — can read the
same formats without one entry point importing another.
"""

from __future__ import annotations

import argparse
import json
import os


def read_api_key(path: str) -> str:
    """Read an API key from a file, tolerating a `NAME=value` line.

    Both shapes appear in practice, and a stray prefix silently producing an invalid bearer token is
    a confusing failure, so it is unwrapped here rather than at each call site.
    """
    with open(path, encoding="utf-8") as handle:
        raw = handle.read().strip()
    if "=" in raw:
        _, raw = raw.split("=", 1)
    raw = raw.strip()
    if not raw:
        raise argparse.ArgumentTypeError("API key file is empty")
    return raw


def load_answer_bank(path: str) -> dict[str, str]:
    """Read the approved question/answer pairs, with or without an enclosing `answers` key."""
    with open(path, encoding="utf-8") as handle:
        loaded = json.load(handle)
    if not isinstance(loaded, dict):
        raise argparse.ArgumentTypeError(f"{path} must contain a JSON object")
    answers = loaded.get("answers", loaded)
    if not isinstance(answers, dict) or not answers:
        raise argparse.ArgumentTypeError(f"{path} must contain a non-empty 'answers' object")
    bank: dict[str, str] = {}
    for question, answer in answers.items():
        if not isinstance(question, str) or not isinstance(answer, (str, int, float, bool)):
            raise argparse.ArgumentTypeError(
                f"{path}: answers must be plain question/answer pairs"
            )
        bank[question] = str(answer)
    return bank


def parse_asset(value: str) -> tuple[str, str]:
    """Parse an `asset_id=/absolute/path` argument. The path is absolutised, never guessed."""
    asset_id, separator, path = value.partition("=")
    if not separator or not asset_id or not path:
        raise argparse.ArgumentTypeError("asset must use asset_id=/absolute/path")
    return asset_id, os.path.abspath(path)
