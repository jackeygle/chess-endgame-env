#!/usr/bin/env bash
# Ablation copy of the full-game task with the one-move-lookahead tool `check_move` enabled.
#
#   scripts/make_hint_variant.sh <seed>      # -> build/chess-full-hints-modal
#
# Same as build/chess-full-modal (public networking, for Modal) plus CHESS_HINTS=1 and a
# fixed CHESS_SEED, so the hint run gets the same openings as the baseline run with that seed.
# Only for the experiment in the README; the shipped task never enables the tool.
set -euo pipefail
cd "$(dirname "$0")/.."
SEED="${1:?seed of the baseline run to match}"
scripts/make_modal_variant.sh >/dev/null
OUT=build/chess-full-hints-modal
rm -rf "$OUT" && cp -r build/chess-full-modal "$OUT"
python3 - "$OUT" "$SEED" <<'PY'
import sys
from pathlib import Path
out, seed = Path(sys.argv[1]), sys.argv[2]
c = out / "environment/docker-compose.yaml"
c.write_text(c.read_text().replace(
    "  mcp-server:\n    build:\n      context: ./mcp-server\n",
    f"  mcp-server:\n    build:\n      context: ./mcp-server\n    environment:\n"
    f"      CHESS_HINTS: \"1\"\n      CHESS_SEED: \"{seed}\"\n"))
t = out / "task.toml"
t.write_text(t.read_text().replace('name = "junction2026/chess-full-game"',
                                   'name = "junction2026/chess-full-game-hints"'))
i = out / "instruction.md"
i.write_text(i.read_text().replace(
    "  and cost nothing. Moves cannot be taken back.\n",
    "  and cost nothing. Moves cannot be taken back.\n"
    "- `check_move(game_id, move)` — without playing it, lists which of your pieces Black\n"
    "  could capture right after that move (attacked and undefended, or attacked by a\n"
    "  cheaper piece). It looks one reply ahead only.\n"))
PY
grep -n "CHESS_" "$OUT/environment/docker-compose.yaml"
