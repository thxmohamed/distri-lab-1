#!/usr/bin/env bash
# Calls a local Ollama server to run one of the agent prompts.
#
# Usage: run_ollama.sh <system_prompt_file> <user_prompt_file> <output_file>
#
# Env vars (all optional):
#   OLLAMA_MODEL        default: qwen2.5-coder:7b-instruct-q4_K_M
#                        (fallback: qwen2.5-coder:3b if the runner times out)
#   OLLAMA_NUM_PREDICT   default: 1200 (max tokens to generate)
#   OLLAMA_TEMPERATURE   default: 0.2
#   OLLAMA_HOST_URL      default: http://localhost:11434
#
# Starts `ollama serve` if it is not already answering, pulls the model
# (a no-op if already cached under ~/.ollama), then calls /api/generate
# with format=json (Ollama enforces syntactically valid JSON output, which
# apply_edits.py/parse_mr_response.py still parse defensively on top of).
set -euo pipefail

if [ "$#" -ne 3 ]; then
  echo "Usage: run_ollama.sh <system_prompt_file> <user_prompt_file> <output_file>" >&2
  exit 2
fi

SYSTEM_PROMPT_FILE="$1"
USER_PROMPT_FILE="$2"
OUTPUT_FILE="$3"

MODEL="${OLLAMA_MODEL:-qwen2.5-coder:7b-instruct-q4_K_M}"
NUM_PREDICT="${OLLAMA_NUM_PREDICT:-1200}"
TEMPERATURE="${OLLAMA_TEMPERATURE:-0.2}"
HOST_URL="${OLLAMA_HOST_URL:-http://localhost:11434}"
LOG_DIR="${RUNNER_TEMP:-/tmp}"

if ! curl -s -o /dev/null "$HOST_URL/api/version"; then
  echo "El servidor de Ollama no responde en $HOST_URL, iniciando 'ollama serve'..."
  nohup ollama serve > "$LOG_DIR/ollama-serve.log" 2>&1 &
  for _ in $(seq 1 60); do
    if curl -s -o /dev/null "$HOST_URL/api/version"; then
      break
    fi
    sleep 2
  done
  if ! curl -s -o /dev/null "$HOST_URL/api/version"; then
    echo "El servidor de Ollama no llego a responder tras 120s." >&2
    cat "$LOG_DIR/ollama-serve.log" >&2 || true
    exit 1
  fi
fi

echo "Modelo: $MODEL (se descarga solo si no esta en cache local)"
ollama pull "$MODEL"

python3 "$(dirname "$0")/ollama_generate.py" \
  "$SYSTEM_PROMPT_FILE" "$USER_PROMPT_FILE" "$MODEL" "$NUM_PREDICT" "$TEMPERATURE" "$HOST_URL" "$OUTPUT_FILE"

echo "Respuesta guardada en $OUTPUT_FILE"
