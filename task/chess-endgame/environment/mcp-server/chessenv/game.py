"""One game: the agent (White) must checkmate a deterministic Stockfish defender.

Determinism: the defender is a fresh Stockfish process per game with Threads=1, a fixed
hash size and a fixed node budget per move, so the same moves always get the same replies.
"""
from __future__ import annotations

import chess
import chess.engine

STOCKFISH = "/usr/games/stockfish"
DEFENDER_NODES = 1_000_000


class IllegalMove(ValueError):
    pass


def new_engine(path: str = STOCKFISH) -> chess.engine.SimpleEngine:
    eng = chess.engine.SimpleEngine.popen_uci(path)
    eng.configure({"Threads": 1, "Hash": 16})
    return eng


def game_reward(won: bool, moves_used: int, mate_len: int, move_limit: int) -> float:
    """1.0 for the fastest possible mate, 0.5 for mating on the last allowed move, 0 otherwise."""
    if not won:
        return 0.0
    if move_limit <= mate_len:
        return 1.0
    slack = (move_limit - moves_used) / (move_limit - mate_len)
    return 0.5 + 0.5 * max(0.0, min(1.0, slack))


class Game:
    def __init__(self, fen: str, mate_len: int, move_limit: int, engine=None):
        self.board = chess.Board(fen)
        if self.board.turn != chess.WHITE:
            raise ValueError("the agent always plays White")
        self.start_fen = fen
        self.mate_len = mate_len
        self.move_limit = move_limit
        self._own_engine = engine is None
        self.engine = engine or new_engine()
        self.engine.protocol.send_line("ucinewgame")
        self.moves_used = 0
        self.result = None            # "mate" | "stalemate" | "draw" | "move_limit" | "lost"
        self.history: list[dict] = []

    # ------------------------------------------------------------------ #
    @property
    def done(self) -> bool:
        return self.result is not None

    @property
    def won(self) -> bool:
        return self.result == "mate"

    def reward(self) -> float:
        return game_reward(self.won, self.moves_used, self.mate_len, self.move_limit)

    def close(self):
        if self._own_engine:
            self.engine.quit()

    # ------------------------------------------------------------------ #
    def parse(self, text: str) -> chess.Move:
        text = str(text).strip()
        try:
            move = chess.Move.from_uci(text)
            if move in self.board.legal_moves:
                return move
        except ValueError:
            pass
        try:
            return self.board.parse_san(text)
        except ValueError:
            raise IllegalMove(f"'{text}' is not a legal move here. Legal moves: "
                              + " ".join(m.uci() for m in self.board.legal_moves)) from None

    def _draw_reason(self) -> str | None:
        b = self.board
        if b.is_stalemate():
            return "stalemate"
        if b.is_insufficient_material():
            return "draw"
        if b.can_claim_threefold_repetition() or b.halfmove_clock >= 100:
            return "draw"
        return None

    def play(self, text: str) -> dict:
        if self.done:
            raise IllegalMove("the game is over")
        move = self.parse(text)
        san = self.board.san(move)
        self.board.push(move)
        self.moves_used += 1
        reply_san = None
        if self.board.is_checkmate():
            self.result = "mate"
        elif (reason := self._draw_reason()):
            self.result = reason
        else:
            res = self.engine.play(self.board, chess.engine.Limit(nodes=DEFENDER_NODES))
            reply_san = self.board.san(res.move)
            self.board.push(res.move)
            if self.board.is_checkmate():
                self.result = "lost"
            elif (reason := self._draw_reason()):
                self.result = reason
            elif self.moves_used >= self.move_limit:
                self.result = "move_limit"
        self.history.append({"move": san, "reply": reply_san, "fen": self.board.fen()})
        return {"your_move": san, "opponent_reply": reply_san, "result": self.result,
                "moves_used": self.moves_used, "moves_left": self.move_limit - self.moves_used}

    def state(self) -> dict:
        return {"fen": self.board.fen(), "board": str(self.board),
                "to_move": "white" if self.board.turn else "black",
                "legal_moves": sorted(m.uci() for m in self.board.legal_moves),
                "moves_used": self.moves_used, "move_limit": self.move_limit,
                "in_check": self.board.is_check(), "result": self.result}
