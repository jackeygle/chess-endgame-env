"""Build a self-contained HTML replay of one or more runs.

    python3 scripts/replay.py out.html "Reference=jobs/<job>/<trial>/verifier/games.json" \
                                       "Kimi K3.1 Pro=jobs/<job>/<trial>/verifier/games.json"
"""
import json
import sys
from pathlib import Path

import chess

TEMPLATE = Path(__file__).with_name("replay_template.html")


def frames_for(game: dict) -> list[dict]:
    b = chess.Board(game["start_fen"])
    out = [{"fen": b.board_fen(), "label": "Start", "lost": False, "side": None}]
    for i, h in enumerate(game["moves"], 1):
        mv = b.parse_san(h["move"])
        b.push(mv)
        out.append({"fen": b.board_fen(), "label": f"{i}. {h['move']}", "side": "w",
                    "lost": False, "from": chess.square_name(mv.from_square),
                    "to": chess.square_name(mv.to_square)})
        if h["reply"]:
            mv = b.parse_san(h["reply"])
            captured = b.piece_at(mv.to_square)
            b.push(mv)
            out.append({"fen": b.board_fen(), "label": f"{i}... {h['reply']}", "side": "b",
                        "lost": bool(captured and captured.color == chess.WHITE),
                        "from": chess.square_name(mv.from_square),
                        "to": chess.square_name(mv.to_square)})
    return out


def main():
    out_path = Path(sys.argv[1])
    runs = []
    for arg in sys.argv[2:]:
        label, path = arg.split("=", 1)
        data = json.loads(Path(path).read_text())
        runs.append({"label": label, "reward": data["reward"],
                     "games": [{"id": g["game_id"], "material": g["material"],
                                "status": g["status"], "reward": g["reward"],
                                "moves_used": g["moves_used"], "mate": g["shortest_mate_in"],
                                "limit": g["move_limit"], "frames": frames_for(g)}
                               for g in data["games"]]})
    html = TEMPLATE.read_text().replace("/*DATA*/null", json.dumps(runs))
    out_path.write_text(html)
    print(f"wrote {out_path} ({len(runs)} runs)")


if __name__ == "__main__":
    main()
