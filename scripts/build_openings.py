"""Keep the balanced candidate openings and write the book the full-game server samples from.

    python3 scripts/build_openings.py

An opening is kept if Stockfish (fixed node budget, one thread) puts it within MAX_CP
centipawns of equality with White to move, and the reference wins from it against every
opponent level (checked with experiments/verify_reference.py).
"""
import json
from pathlib import Path

import chess
import chess.engine

from chessenv.game import new_engine
from chessenv.openings import CANDIDATES

MAX_CP = 80
# Openings the reference did not win against every level (experiments/verify_reference.py):
# the book only holds openings the reference wins in all four games.
REFERENCE_FAILS = {"Two Knights Defence"}       # draw by repetition against Level 2
NODES = 2_000_000
OUT = Path(__file__).resolve().parent.parent / "task/chess-full/environment/mcp-server/openings.json"

eng = new_engine()
book = []
for name, line in CANDIDATES.items():
    b = chess.Board()
    for san in line.split():
        b.push_san(san)
    assert b.turn == chess.WHITE, name
    cp = eng.analyse(b, chess.engine.Limit(nodes=NODES))["score"].white().score(mate_score=100_000)
    keep = abs(cp) <= MAX_CP and name not in REFERENCE_FAILS
    print(f"{'keep' if keep else 'drop'} {cp:+5d}  {name}", flush=True)
    if keep:
        book.append({"name": name, "moves": line, "fen": b.fen(), "eval_cp": cp})
eng.quit()
OUT.write_text(json.dumps(book, indent=1))
print(f"{len(book)} of {len(CANDIDATES)} openings kept -> {OUT}")
