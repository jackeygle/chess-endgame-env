"""Classify every agent move with Stockfish: why did the agent lose?

    python3 scripts/analyze_blunders.py "GPT-5.5 run 1=path/games.json" ... [--out file.json]

For each White move: Stockfish's evaluation before and after it (fixed node budget, one
thread), so the evaluation drop is the cost of that move. A move that loses 300 centipawns
or more (three pawns) is a blunder. A blunder is a one-move oversight when after it the
opponent can immediately capture a piece that is undefended or worth more than its
attacker (chessenv.tactics.en_prise): a mistake that looking one reply ahead would catch.
Otherwise it is a deeper mistake (a tactic two or more moves deep, or a bad plan).
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import chess
import chess.engine

from chessenv.game import new_engine
from chessenv.tactics import en_prise

NODES = 300_000
BLUNDER_CP = 300
CLAMP = 2000          # mate scores count as +-20 pawns, so one mate does not dominate sums


def evaluate(eng, board):
    if board.is_game_over(claim_draw=True):
        if board.is_checkmate():
            return CLAMP if board.turn == chess.BLACK else -CLAMP
        return 0
    s = eng.analyse(board, chess.engine.Limit(nodes=NODES))["score"].white()
    return max(-CLAMP, min(CLAMP, s.score(mate_score=100_000)))


def analyse_game(eng, g):
    board = chess.Board(g["start_fen"])
    rows = []
    before = evaluate(eng, board)
    for n, h in enumerate(g["moves"], 1):
        move = board.parse_san(h["move"])
        board.push(move)
        after = evaluate(eng, board)
        hanging = en_prise(board, chess.WHITE) if board.turn == chess.BLACK else []
        drop = before - after
        kind = None
        if drop >= BLUNDER_CP:
            kind = "one-move oversight" if hanging else "deeper mistake"
        rows.append({"n": n, "move": h["move"], "eval_before": before, "eval_after": after,
                     "drop": drop, "kind": kind, "hanging": hanging})
        if h["reply"]:
            board.push(board.parse_san(h["reply"]))
        before = evaluate(eng, board)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="Label=path/to/games.json")
    ap.add_argument("--out")
    a = ap.parse_args()
    eng = new_engine()
    report, kinds, moves_total = [], Counter(), 0
    first_blunder = []
    for spec in a.runs:
        label, path = spec.split("=", 1)
        data = json.loads(Path(path).read_text())
        for g in data["games"]:
            if not g["moves"]:
                continue
            rows = analyse_game(eng, g)
            moves_total += len(rows)
            bl = [r for r in rows if r["kind"]]
            kinds.update(r["kind"] for r in bl)
            if bl:
                first_blunder.append(bl[0]["n"])
            report.append({"run": label, "game": g["game_id"], "opponent": g.get("opponent"),
                           "opening": g.get("opening"), "result": g["status"],
                           "reward": g["reward"], "moves": len(rows), "blunders": len(bl),
                           "one_move_oversights": sum(r["kind"] == "one-move oversight" for r in bl),
                           "first_blunder_move": bl[0]["n"] if bl else None,
                           "lost_cp_in_blunders": sum(r["drop"] for r in bl),
                           "rows": rows})
            print(f"{label:22s} {g['game_id']} {str(g.get('opponent'))[:22]:22s} "
                  f"{g['status']:18s} moves={len(rows):3d} blunders={len(bl):2d} "
                  f"one-move={report[-1]['one_move_oversights']:2d} "
                  f"first={report[-1]['first_blunder_move']}", flush=True)
    eng.quit()
    total = sum(kinds.values())
    summary = {"games": len(report), "agent_moves": moves_total, "blunders": total,
               "one_move_oversights": kinds["one-move oversight"],
               "deeper_mistakes": kinds["deeper mistake"],
               "share_one_move": kinds["one-move oversight"] / total if total else None,
               "games_with_blunder": sum(r["blunders"] > 0 for r in report),
               "median_first_blunder_move": sorted(first_blunder)[len(first_blunder) // 2]
               if first_blunder else None}
    print(json.dumps(summary, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps({"summary": summary, "games": report}, indent=1))


if __name__ == "__main__":
    main()
