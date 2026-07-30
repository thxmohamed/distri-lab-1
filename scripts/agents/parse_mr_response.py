#!/usr/bin/env python3
"""Robustly extract {classification, reason} from the MR-reviewer agent's
raw text response (see mr-reviewer.md). Same tolerant-parsing rationale as
apply_edits.load_json_response: the model is asked for JSON-only in plain
text (no responseFormat: json_schema, see apply_edits.py docstring for
why), so this tolerates a ```json fence or stray prose around the object,
and safely defaults to "human_review" if nothing parseable is found -- an
unparsed response should never be silently read as "mergeable".

Usage:
    python3 parse_mr_response.py <response.txt>
Writes classification and reason to $GITHUB_OUTPUT.
"""
import json
import os
import re
import sys


def write_output(name: str, value: str) -> None:
    gh_output = os.environ.get("GITHUB_OUTPUT")
    delimiter = f"__AGENT_OUTPUT_{name.upper()}__"
    line = f"{name}<<{delimiter}\n{value}\n{delimiter}\n"
    if gh_output:
        with open(gh_output, "a", encoding="utf-8") as fh:
            fh.write(line)
    else:
        sys.stdout.write(line)


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: parse_mr_response.py <response.txt>", file=sys.stderr)
        return 2

    with open(sys.argv[1], encoding="utf-8") as fh:
        raw = fh.read()

    candidates = [raw.strip()]
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if fenced:
        candidates.append(fenced.group(1))
    braces = re.search(r"\{.*\}", raw, re.DOTALL)
    if braces:
        candidates.append(braces.group(0))

    data = None
    for candidate in candidates:
        try:
            data = json.loads(candidate)
            break
        except (json.JSONDecodeError, ValueError):
            continue

    if data is None or "classification" not in data:
        print(f"WARNING: could not parse a JSON object out of the model response in {sys.argv[1]}.", file=sys.stderr)
        print(f"--- raw response ---\n{raw}\n--- end raw response ---", file=sys.stderr)
        write_output("classification", "human_review")
        write_output(
            "reason",
            "El agente no devolvio una respuesta interpretable; se marca para revision humana por seguridad.",
        )
        return 0

    write_output("classification", data.get("classification", "human_review"))
    write_output("reason", data.get("reason", ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
