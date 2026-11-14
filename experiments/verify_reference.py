"""Check that the reference wins every opening of the book against every opponent level.

    modal run experiments/verify_reference.py          # 40 openings x 4 levels, in parallel

Runs on Modal in the server's base image (Debian bookworm's Stockfish), with the same Game
class and the same reference budget as the server's oracle route. Writes
results/reference_check.json. Openings the reference does not win are listed; the book
should only contain openings it wins against all four levels.
"""
import json
from pathlib import Path

import modal

app = modal.App("chess-reference-check")
image = (modal.Image.debian_slim(python_version="3.12").apt_install("stockfish")
         .pip_install("chess==1.11.2").add_local_python_source("chessenv"))
ROOT = Path(__file__).resolve().parent.parent
REFERENCE_NODES = 2_000_000


@app.function(image=image, cpu=1, memory=1024, timeout=3600)
def play(opening: dict, pos: dict, seed: str) -> dict:
    import chess.engine
    from chessenv.game import Game, new_engine
    g = Game(opening["fen"], None, pos["move_limit"], mode="full",
             opponent=pos.get("opponent_kind", "engine"), seed=seed,
             **({"opponent_nodes": pos["opponent_nodes"]} if "opponent_nodes" in pos else {}))
    ref = new_engine()
    try:
        while not g.done:
            g.play(ref.play(g.board, chess.engine.Limit(nodes=REFERENCE_NODES)).move.uci())
        return {"opening": opening["name"], "game": pos["id"], "opponent": pos["opponent"],
                "result": g.result, "moves": g.moves_used, "reward": g.reward()}
    finally:
        ref.quit()
        g.close()


@app.local_entrypoint()
def main():
    srv = ROOT / "task/chess-full/environment/mcp-server"
    book = json.loads((srv / "openings.json").read_text())
    positions = json.loads((srv / "positions.json").read_text())
    jobs = [(o, p, f"check:{o['name']}:{p['id']}") for o in book for p in positions]
    out = list(play.starmap(jobs))
    (ROOT / "results/reference_check.json").write_text(json.dumps(out, indent=1))
    bad = [r for r in out if r["reward"] < 1.0]
    for r in bad:
        print("NOT WON:", r)
    print(f"{len(out) - len(bad)}/{len(out)} games won by the reference")
