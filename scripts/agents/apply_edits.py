#!/usr/bin/env python3
"""Validate and apply the structured JSON response from a documentation/bug
agent (see documentador.prompt.yml / bug-reviewer.prompt.yml).

Both agents answer with a fixed JSON schema:
  action:      "none" | "open_issue" | "open_fix_pr" | "human_required"
  title:       short title for the issue/PR
  reason:      why this classification was chosen
  issue_body:  markdown body for the issue
  target_file: path to edit, only used when action == "open_fix_pr"
  edits:       list of {find, replace} exact string replacements

This script is intentionally conservative: since the agent is a single
inference call (no compiler, no test run, no re-reading of its own edits),
"open_fix_pr" is only trusted when every "find" string appears EXACTLY ONCE
in the target file and the target file is in the caller-provided allow-list.
Any mismatch silently degrades the action to "open_issue" instead of risking
a corrupted file, and a note is appended to issue_body explaining why the
automatic patch was not applied.

Usage:
    python3 apply_edits.py <response.json> <comma,separated,allowlist>
Writes the final decision to $GITHUB_OUTPUT as: action, title, reason,
issue_body, target_file (all as GitHub Actions step outputs).
"""
import json
import os
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
    if len(sys.argv) != 3:
        print("Usage: apply_edits.py <response.json> <allowlist,comma,separated>", file=sys.stderr)
        return 2

    response_path, allowlist_raw = sys.argv[1], sys.argv[2]
    allowlist = {p.strip() for p in allowlist_raw.split(",") if p.strip()}

    with open(response_path, encoding="utf-8") as fh:
        data = json.load(fh)

    action = data.get("action", "none")
    title = data.get("title", "") or ""
    reason = data.get("reason", "") or ""
    issue_body = data.get("issue_body", "") or ""
    target_file = data.get("target_file", "") or ""
    edits = data.get("edits", []) or []

    def is_allowed(path: str) -> bool:
        if not path or os.path.isabs(path) or ".." in path.split("/"):
            return False
        for entry in allowlist:
            if entry.endswith("/"):
                if path.startswith(entry):
                    return True
            elif path == entry:
                return True
        return False

    if action == "open_fix_pr":
        degrade_reason = None
        if not target_file or not is_allowed(target_file):
            degrade_reason = (
                f"El agente propuso editar `{target_file}`, fuera de la lista "
                f"permitida ({', '.join(sorted(allowlist))})."
            )
        elif not os.path.isfile(target_file):
            degrade_reason = f"El agente propuso editar `{target_file}`, que no existe en el repo."
        elif not edits:
            degrade_reason = "El agente marco open_fix_pr pero no entrego ningun edit."
        else:
            with open(target_file, encoding="utf-8") as fh:
                content = fh.read()
            new_content = content
            for i, edit in enumerate(edits):
                find, replace = edit.get("find", ""), edit.get("replace", "")
                occurrences = new_content.count(find)
                if not find or occurrences != 1:
                    degrade_reason = (
                        f"El edit #{i + 1} propuesto no aparece exactamente una vez en "
                        f"`{target_file}` (apariciones encontradas: {occurrences}); se "
                        "descarta el parche automatico por seguridad."
                    )
                    break
                new_content = new_content.replace(find, replace, 1)
            if degrade_reason is None:
                with open(target_file, "w", encoding="utf-8") as fh:
                    fh.write(new_content)

        if degrade_reason is not None:
            action = "open_issue"
            issue_body = (
                f"{issue_body}\n\n---\n_Nota del agente: se intento un fix automatico, pero "
                f"se degrado a issue manual. Motivo: {degrade_reason}_"
            ).strip()

    write_output("action", action)
    write_output("title", title)
    write_output("reason", reason)
    write_output("issue_body", issue_body)
    write_output("target_file", target_file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
