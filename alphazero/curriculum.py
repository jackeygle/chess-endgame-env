"""Training start positions, generated and verified by the rules alone (no engine).

Kinds:
  mate1   White to move has a checkmate in one          (verified by trying every move)
  mate2   White to move mates in two against any defence (verified by exhaustive search)
  KQK     king and queen against king, White to move
  KRK     king and rook against king, White to move
  start   the standard initial position

Test positions are excluded from training by FEN (`exclude`), so the evaluation set is never
a training start position.
"""
from __future__ import annotations

import random

import chess

MATERIAL = {"KQK": "Q", "KRK": "R", "mate1": "QR", "mate2": "QR"}
PIECE = {"Q": chess.QUEEN, "R": chess.ROOK}


def _random_board(rng: random.Random, white: str) -> chess.Board | None:
    b = chess.Board(None)
    sq = rng.sample(range(64), 2 + len(white))
    b.set_piece_at(sq[0], chess.Piece(chess.KING, chess.WHITE))
    b.set_piece_at(sq[1], chess.Piece(chess.KING, chess.BLACK))
    for s, p in zip(sq[2:], white):
        b.set_piece_at(s, chess.Piece(PIECE[p], chess.WHITE))
    b.turn = chess.WHITE
    if not b.is_valid() or b.is_game_over() or b.is_check():
        return None
    return b


def has_mate_in_1(b: chess.Board) -> bool:
    for m in b.legal_moves:
        b.push(m)
        mate = b.is_checkmate()
        b.pop()
        if mate:
            return True
    return False


def has_mate_in_2(b: chess.Board) -> bool:
    """White mates in exactly two moves (and not in one) against every Black reply."""
    if has_mate_in_1(b):
        return False
    for m in b.legal_moves:
        b.push(m)
        ok = not b.is_game_over() and all(_reply_allows_mate(b, r) for r in list(b.legal_moves))
        b.pop()
        if ok:
            return True
    return False


def _reply_allows_mate(b: chess.Board, r: chess.Move) -> bool:
    b.push(r)
    ok = has_mate_in_1(b)
    b.pop()
    return ok


def sample(kind: str, rng: random.Random, exclude: set[str] = frozenset(),
           tries: int = 20000) -> chess.Board:
    if kind == "start":
        return chess.Board()
    for _ in range(tries):
        pieces = MATERIAL[kind]
        if kind in ("mate1", "mate2"):
            pieces = rng.choice(["Q", "R", "QR"])
        b = _random_board(rng, pieces)
        if b is None or b.fen() in exclude:
            continue
        if kind == "mate1" and not has_mate_in_1(b):
            continue
        if kind == "mate2" and not has_mate_in_2(b):
            continue
        if kind in ("KQK", "KRK") and has_mate_in_1(b):
            continue                       # keep the plain endgames non-trivial
        return b
    raise RuntimeError(f"could not sample a {kind} position in {tries} tries")


def mixture(weights: dict[str, float], rng: random.Random, exclude=frozenset()) -> tuple[str, chess.Board]:
    kinds = [k for k, w in weights.items() if w > 0]
    kind = rng.choices(kinds, [weights[k] for k in kinds])[0]
    return kind, sample(kind, rng, exclude)


def test_set(n_per_kind: int, seed: int, kinds=("mate1", "mate2", "KQK", "KRK")) -> list[dict]:
    rng = random.Random(seed)
    out, seen = [], set()
    for k in kinds:
        got = 0
        while got < n_per_kind:
            b = sample(k, rng)
            if b.fen() in seen:
                continue
            seen.add(b.fen())
            out.append({"kind": k, "fen": b.fen()})
            got += 1
    return out
