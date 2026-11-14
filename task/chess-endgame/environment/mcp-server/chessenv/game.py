"""One game: the agent plays White against a deterministic Stockfish opponent.

Two modes:
  endgame  the start position is a forced mate; only mating within the move limit scores
  full     a whole game from any position; win 1, draw 0.5, loss or unfinished 0. A game
           that reaches the move limit is adjudicated by Stockfish's evaluation.

Opponents: Stockfish with a fixed node budget per move (a fresh process per game, Threads=1,
fixed hash: the same moves always get the same replies), or a random mover that picks a
uniformly random legal move from a seeded generator (reproducible given the seed).
"""
from __future__ import annotations

import random

import chess
import chess.engine

STOCKFISH = "/usr/games/stockfish"
DEFENDER_NODES = 1_000_000
ADJUDICATE_NODES = 2_000_000
ADJUDICATE_CP = 300          # a lead of three pawns or more at the move limit counts as decided
FULL_REWARD = {"mate": 1.0, "adjudicated_win": 1.0, "draw": 0.5, "stalemate": 0.5,
               "adjudicated_draw": 0.5}


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
    def __init__(self, fen: str, mate_len: int | None, move_limit: int, engine=None,
                 opponent_nodes: int = DEFENDER_NODES, mode: str = "endgame",
                 opponent: str = "engine", seed: int | str | None = None):
        if mode not in ("endgame", "full"):
            raise ValueError(f"mode must be 'endgame' or 'full', got {mode!r}")
        self.board = chess.Board(fen)
        if self.board.turn != chess.WHITE:
            raise ValueError("the agent always plays White")
        self.start_fen = fen
        self.mate_len = mate_len
        self.move_limit = move_limit
        self.opponent_nodes = opponent_nodes
        self.mode = mode
        if opponent not in ("engine", "random"):
            raise ValueError(f"opponent must be 'engine' or 'random', got {opponent!r}")
        self.opponent = opponent
        self._rng = random.Random(seed)
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
        if self.mode == "full":
            return FULL_REWARD.get(self.result, 0.0)
        return game_reward(self.won, self.moves_used, self.mate_len, self.move_limit)

    def _adjudicate(self) -> str:
        info = self.engine.analyse(self.board, chess.engine.Limit(nodes=ADJUDICATE_NODES))
        cp = info["score"].white().score(mate_score=100_000)
        if cp >= ADJUDICATE_CP:
            return "adjudicated_win"
        if cp <= -ADJUDICATE_CP:
            return "adjudicated_loss"
        return "adjudicated_draw"

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

    def _reply(self) -> chess.Move:
        if self.opponent == "random":
            return self._rng.choice(sorted(self.board.legal_moves, key=chess.Move.uci))
        return self.engine.play(self.board, chess.engine.Limit(nodes=self.opponent_nodes)).move

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
            reply = self._reply()
            reply_san = self.board.san(reply)
            self.board.push(reply)
            if self.board.is_checkmate():
                self.result = "lost"
            elif (reason := self._draw_reason()):
                self.result = reason
            elif self.moves_used >= self.move_limit:
                self.result = self._adjudicate() if self.mode == "full" else "move_limit"
        self.history.append({"move": san, "reply": reply_san, "fen": self.board.fen()})
        return {"your_move": san, "opponent_reply": reply_san, "result": self.result,
                "moves_used": self.moves_used, "moves_left": self.move_limit - self.moves_used}

    def state(self) -> dict:
        return {"fen": self.board.fen(), "board": str(self.board),
                "to_move": "white" if self.board.turn else "black",
                "legal_moves": sorted(m.uci() for m in self.board.legal_moves),
                "moves_used": self.moves_used, "move_limit": self.move_limit,
                "in_check": self.board.is_check(), "result": self.result}
