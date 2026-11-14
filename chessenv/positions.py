"""Generate seeded won-for-White positions and verify each is a forced mate with Stockfish.

    python3 -m chessenv.positions --per-class 6 --out positions.json
"""
from __future__ import annotations

import argparse
import json
import random

import chess
import chess.engine

from chessenv.game import new_engine

# White pieces besides the king; Black has a bare king unless stated.
CLASSES = {
    "KQK":  ("Q", ""),
    "KRK":  ("R", ""),
    "KBBK": ("BB", ""),
    "KBNK": ("BN", ""),
    "KQKR": ("Q", "r"),
}
PIECE = {"Q": chess.QUEEN, "R": chess.ROOK, "B": chess.BISHOP, "N": chess.KNIGHT}


def random_position(rng: random.Random, white: str, black: str) -> chess.Board | None:
    b = chess.Board(None)
    squares = rng.sample(range(64), 2 + len(white) + len(black))
    b.set_piece_at(squares[0], chess.Piece(chess.KING, chess.WHITE))
    b.set_piece_at(squares[1], chess.Piece(chess.KING, chess.BLACK))
    for sq, p in zip(squares[2:], white + black):
        b.set_piece_at(sq, chess.Piece(PIECE[p.upper()], p.isupper()))
    b.turn = chess.WHITE
    if white == "BB":           # two bishops must be on opposite colours
        bs = list(b.pieces(chess.BISHOP, chess.WHITE))
        if chess.square_rank(bs[0]) % 2 ^ chess.square_file(bs[0]) % 2 == \
           chess.square_rank(bs[1]) % 2 ^ chess.square_file(bs[1]) % 2:
            return None
    if not b.is_valid() or b.is_check() or b.is_game_over():
        return None
    return b


def mate_length(engine, board, seconds: float) -> int | None:
    info = engine.analyse(board, chess.engine.Limit(time=seconds))
    m = info["score"].white().mate()
    return m if m is not None and m > 0 else None


def generate(per_class: int, seed: int, seconds: float, min_mate: dict) -> list[dict]:
    rng = random.Random(seed)
    eng = new_engine()
    out = []
    try:
        for name, (white, black) in CLASSES.items():
            got, tries = 0, 0
            while got < per_class and tries < 5000:
                tries += 1
                b = random_position(rng, white, black)
                if b is None:
                    continue
                n = mate_length(eng, b, seconds)
                if n is None or n < min_mate.get(name, 5):
                    continue
                out.append({"id": f"{name}-{got+1}", "class": name, "fen": b.fen(), "mate_len": n})
                got += 1
    finally:
        eng.quit()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=6)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--seconds", type=float, default=2.0)
    ap.add_argument("--out", default="positions.json")
    a = ap.parse_args()
    min_mate = {"KQK": 7, "KRK": 12, "KBBK": 14, "KBNK": 20, "KQKR": 12}
    pos = generate(a.per_class, a.seed, a.seconds, min_mate)
    with open(a.out, "w") as f:
        json.dump(pos, f, indent=1)
    for p in pos:
        print(p["id"], p["mate_len"], p["fen"])


if __name__ == "__main__":
    main()
