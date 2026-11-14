#!/usr/bin/env bash
# Run every evaluation the judges run: reference, do-nothing, stop-early and three agent runs.
#
#   MODEL=anthropic/claude-sonnet-5-5 ENV=docker ./scripts/run_eval.sh
#
# ENV is a Harbor environment (docker, or modal to run without local Docker).
# The agent runs need the model provider's API key in the environment (e.g. ANTHROPIC_API_KEY).
set -euo pipefail
cd "$(dirname "$0")/.."
TASK=task/chess-endgame
ENV="${ENV:-docker}"
MODEL="${MODEL:?set MODEL to provider/model, e.g. anthropic/claude-sonnet-5-5}"
RUNS="${RUNS:-3}"

harbor run --path "$TASK" --env "$ENV" --agent oracle -y                       # reference
harbor run --path "$TASK" --env "$ENV" --agent nop -y                          # do nothing
PYTHONPATH=. harbor run --path "$TASK" --env "$ENV" -y \
  --agent-import-path baselines.stop_early:StopEarlyAgent                       # stop early
for i in $(seq 1 "$RUNS"); do                                                   # the agent
  harbor run --path "$TASK" --env "$ENV" --agent opencode --model "$MODEL" -y
done
python3 scripts/summarize.py jobs
