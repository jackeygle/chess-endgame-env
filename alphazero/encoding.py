"""Board and move encoding.

Canonical orientation: the network always sees the board from the side to move. When Black
is to move the position is mirrored (ranks flipped, colours swapped) with
`chess.Board.mirror()`, which keeps castling rights and the en-passant square consistent,
so this is a change of coordinates, not a data augmentation: every position has exactly
one encoding and no rule changes. A plane records the true colour to move.

Input planes, shape (21, 8, 8), from the side to move's point of view ("us"):
   0-5   our pawns, knights, bishops, rooks, queens, king
   6-11  their pawns, knights, bishops, rooks, queens, king
   12-15 castling: our king side, our queen side, their king side, their queen side
   16    en-passant target square
   17    half-move clock / 100 (fifty-move rule progress)
   18    this position has occurred once before
   19    this position has occurred twice before (a third time would be a draw)
   20    1 if White is to move in the real game, else 0

Actions: from-square x to-square x promotion, 64 * 64 * 5 = 20480 indices, in the
canonical orientation. promotion index 0 = none, 1..4 = knight, bishop, rook, queen.
Castling is the king's two-square move (python-chess convention, e1g1), en passant is the
pawn's diagonal move to the target square: both are plain from/to moves, so every legal
move has exactly one index and decoding against the legal moves is exact.
"""
from __future__ import annotations

import chess
import numpy as np

N_PLANES = 21
N_ACTIONS = 64 * 64 * 5
PROMOS = [None, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]
_PROMO_IDX = {p: i for i, p in enumerate(PROMOS)}
PIECES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]


def canonical(board: chess.Board) -> chess.Board:
    return board if board.turn == chess.WHITE else board.mirror()


def _flip_move(move: chess.Move) -> chess.Move:
    return chess.Move(chess.square_mirror(move.from_square), chess.square_mirror(move.to_square),
                      move.promotion)


def move_to_index(board: chess.Board, move: chess.Move) -> int:
    """Index of a legal move of `board`, in the side-to-move orientation."""
    m = move if board.turn == chess.WHITE else _flip_move(move)
    return (m.from_square * 64 + m.to_square) * 5 + _PROMO_IDX[m.promotion]


def index_to_move(board: chess.Board, index: int) -> chess.Move:
    """Inverse of move_to_index. Raises ValueError if the index is not a legal move."""
    fr, rest = divmod(index, 320)
    to, p = divmod(rest, 5)
    m = chess.Move(fr, to, PROMOS[p])
    if board.turn == chess.BLACK:
        m = _flip_move(m)
    if m not in board.legal_moves:
        raise ValueError(f"index {index} is not a legal move in {board.fen()}")
    return m


def legal_indices(board: chess.Board) -> tuple[list[chess.Move], np.ndarray]:
    moves = list(board.legal_moves)
    return moves, np.array([move_to_index(board, m) for m in moves], dtype=np.int64)


def repetition_count(board: chess.Board) -> int:
    """How many times the current position occurred before (0, 1 or 2+), by the rules'
    definition of a repeated position (pieces, side to move, castling, en passant)."""
    key = board._transposition_key()
    count = 0
    switchyard = []
    try:
        while board.move_stack:
            switchyard.append(board.pop())
            if board._transposition_key() == key:
                count += 1
                if count >= 2:
                    break
    finally:
        while switchyard:
            board.push(switchyard.pop())
    return count


def encode(board: chess.Board) -> np.ndarray:
    reps = repetition_count(board)
    c = canonical(board)
    x = np.zeros((N_PLANES, 8, 8), dtype=np.float32)
    for i, pt in enumerate(PIECES):
        for sq in c.pieces(pt, chess.WHITE):
            x[i, sq // 8, sq % 8] = 1.0
        for sq in c.pieces(pt, chess.BLACK):
            x[6 + i, sq // 8, sq % 8] = 1.0
    x[12] = float(c.has_kingside_castling_rights(chess.WHITE))
    x[13] = float(c.has_queenside_castling_rights(chess.WHITE))
    x[14] = float(c.has_kingside_castling_rights(chess.BLACK))
    x[15] = float(c.has_queenside_castling_rights(chess.BLACK))
    if c.ep_square is not None and c.has_legal_en_passant():
        x[16, c.ep_square // 8, c.ep_square % 8] = 1.0
    x[17] = min(board.halfmove_clock, 100) / 100.0
    x[18] = float(reps >= 1)
    x[19] = float(reps >= 2)
    x[20] = float(board.turn == chess.WHITE)
    return x


def cache_key(board: chess.Board) -> tuple:
    """Everything the encoding depends on: position, side, castling, en passant (via the
    transposition key), the half-move clock and the repetition count."""
    return (board._transposition_key(), min(board.halfmove_clock, 100), repetition_count(board))
