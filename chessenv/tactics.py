"""One-move lookahead: which of the mover's pieces can the opponent win right after a move.

A piece counts as *en prise* (left hanging) when an enemy piece attacks it and either no
friendly piece defends it, or the cheapest attacker is worth less than it. This is the
check a player makes by looking one reply ahead; it ignores pins and longer sequences.
"""
from __future__ import annotations

import chess

VALUE = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}


def en_prise(board: chess.Board, color: chess.Color) -> list[dict]:
    """Pieces of `color` the side to move (the opponent) can win by capturing now."""
    out = []
    for sq, piece in board.piece_map().items():
        if piece.color != color or piece.piece_type == chess.KING:
            continue
        attackers = [s for s in board.attackers(not color, sq)
                     if board.is_legal(chess.Move(s, sq)) or board.piece_at(s).piece_type == chess.PAWN
                     and board.is_legal(chess.Move(s, sq, chess.QUEEN))]
        if not attackers:
            continue
        cheapest = min(VALUE.get(board.piece_at(s).piece_type, 100) for s in attackers)
        defended = bool(board.attackers(color, sq))
        if not defended or cheapest < VALUE[piece.piece_type]:
            out.append({"square": chess.square_name(sq), "piece": piece.symbol(),
                        "value": VALUE[piece.piece_type], "defended": defended,
                        "attacked_by": sorted(board.piece_at(s).symbol() + chess.square_name(s)
                                              for s in attackers)})
    return sorted(out, key=lambda x: -x["value"])


def hanging_after(board: chess.Board, move: chess.Move) -> list[dict]:
    """What the opponent could win immediately if the side to move played `move`."""
    b = board.copy(stack=False)
    color = b.turn
    b.push(move)
    return en_prise(b, color)
