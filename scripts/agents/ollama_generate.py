#!/usr/bin/env python3
"""Call a local Ollama server's /api/generate and write the raw response
text to a file. Used by run_ollama.sh; kept as a separate .py (instead of
an inline heredoc) so it can be unit-tested/inspected on its own.

Usage:
    python3 ollama_generate.py <system_file> <user_file> <model> \
        <num_predict> <temperature> <host_url> <output_file>
"""
import json
import sys
import urllib.error
import urllib.request

TIMEOUT_SECONDS = 30 * 60  # CPU-only inference on a 7B model can be slow.


def main() -> int:
    if len(sys.argv) != 8:
        print(
            "Usage: ollama_generate.py <system_file> <user_file> <model> "
            "<num_predict> <temperature> <host_url> <output_file>",
            file=sys.stderr,
        )
        return 2

    system_file, user_file, model, num_predict, temperature, host_url, output_file = sys.argv[1:8]

    with open(system_file, encoding="utf-8") as fh:
        system_prompt = fh.read()
    with open(user_file, encoding="utf-8") as fh:
        user_prompt = fh.read()

    payload = {
        "model": model,
        "system": system_prompt,
        "prompt": user_prompt,
        "format": "json",
        "stream": False,
        "options": {
            "num_predict": int(num_predict),
            "temperature": float(temperature),
        },
    }

    request = urllib.request.Request(
        f"{host_url}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        print(f"ERROR: request to {host_url}/api/generate failed: {exc}", file=sys.stderr)
        # Leave behind an empty/action-none-ish response instead of crashing
        # the workflow outright; apply_edits.py already treats unparseable
        # content as a safe no-op.
        with open(output_file, "w", encoding="utf-8") as fh:
            fh.write("")
        return 1

    with open(output_file, "w", encoding="utf-8") as fh:
        fh.write(body.get("response", ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
