"""Baseline and reference players. A policy maps a chess.Board (White to move) to a move."""
from __future__ import annotations

import random

import chess
import chess.engine

from chessenv.game import new_engine

REFERENCE_NODES = 2_000_000


def _mate_now(board):
    for m in board.legal_moves:
        board.push(m)
        mate = board.is_checkmate()
        board.pop()
        if mate:
            return m
    return None


def random_policy(board, rng):
    return rng.choice(list(board.legal_moves))


def checker(board, rng):
    """Mate if possible, else give a check, else play anything: aimless checking."""
    if (m := _mate_now(board)):
        return m
    checks = [m for m in board.legal_moves if board.gives_check(m)]
    return rng.choice(checks or list(board.legal_moves))


def squeeze(board, rng):
    """One-ply greed: minimise the defender's legal moves, blind to stalemate."""
    if (m := _mate_now(board)):
        return m
    best, best_n = [], 99
    for m in board.legal_moves:
        board.push(m)
        n = board.legal_moves.count()
        board.pop()
        if n < best_n:
            best, best_n = [m], n
        elif n == best_n:
            best.append(m)
    return rng.choice(best)


def _hangs(board, move) -> bool:
    """After `move`, can Black capture a White piece that is not defended?"""
    board.push(move)
    try:
        for sq in board.pieces(chess.QUEEN, chess.WHITE) | board.pieces(chess.ROOK, chess.WHITE) \
                | board.pieces(chess.BISHOP, chess.WHITE) | board.pieces(chess.KNIGHT, chess.WHITE):
            if board.is_attacked_by(chess.BLACK, sq) and not board.is_attacked_by(chess.WHITE, sq):
                return True
        return False
    finally:
        board.pop()


def safe_squeeze(board, rng):
    """Sensible one-ply play: never hang a piece or stalemate; squeeze the king, bring ours closer."""
    if (m := _mate_now(board)):
        return m
    bk = board.king(chess.BLACK)
    scored = []
    for m in board.legal_moves:
        if _hangs(board, m):
            continue
        board.push(m)
        stalemate = board.is_stalemate()
        mobility = board.legal_moves.count()
        wk = board.king(chess.WHITE)
        board.pop()
        if stalemate:
            continue
        scored.append((mobility, chess.square_distance(wk, bk), rng.random(), m))
    if not scored:
        return rng.choice(list(board.legal_moves))
    return min(scored)[-1]


class Reference:
    """Full-strength Stockfish on its own process (never the defender's)."""

    def __init__(self):
        self.engine = new_engine()

    def __call__(self, board, rng=None):
        return self.engine.play(board, chess.engine.Limit(nodes=REFERENCE_NODES)).move

    def close(self):
        self.engine.quit()
