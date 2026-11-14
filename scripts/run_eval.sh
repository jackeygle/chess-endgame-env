#!/usr/bin/env bash
# Run every evaluation the judges run: reference, do-nothing, stop-early and three agent runs.
#
#   MODEL=openai/gpt-5.5 ./scripts/run_eval.sh
#
# ENV is a Harbor environment (docker, or modal to run without local Docker; with modal,
# first run scripts/make_modal_variant.sh and set TASK=build/chess-full-modal).
# TASK defaults to the full-game task; set TASK=task/chess-endgame for the endgames.
# The agent runs need the model provider's API key in the environment (e.g. OPENAI_API_KEY).
set -euo pipefail
cd "$(dirname "$0")/.."
TASK="${TASK:-task/chess-full}"
ENV="${ENV:-docker}"
MODEL="${MODEL:?set MODEL to provider/model, e.g. openai/gpt-5.5}"
RUNS="${RUNS:-3}"

harbor run --path "$TASK" --env "$ENV" --agent oracle -y                       # reference
harbor run --path "$TASK" --env "$ENV" --agent nop -y                          # do nothing
PYTHONPATH=. harbor run --path "$TASK" --env "$ENV" -y \
  --agent-import-path baselines.stop_early:StopEarlyAgent                       # stop early
for i in $(seq 1 "$RUNS"); do                                                   # the agent
  harbor run --path "$TASK" --env "$ENV" --agent opencode --model "$MODEL" -y
  sleep 2   # Harbor names job folders by the second
done
python3 scripts/summarize.py jobs
