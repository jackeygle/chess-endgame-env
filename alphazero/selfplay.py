"""Self-play: many games in lockstep so leaf evaluations batch across games.

Each move stores a sample (planes, legal action indices, MCTS visit distribution, side to
move). When a game ends by the rules, every sample gets z = the result from the point of
view of the side to move in that sample (+1 win, 0 draw, -1 loss). Games cut off at
`max_plies` have no true result; their samples are discarded, not labelled as draws.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import chess
import numpy as np

from alphazero import curriculum
from alphazero.encoding import encode
from alphazero.mcts import SearchConfig, Tree, search, terminal_value, visit_policy


@dataclass
class Sample:
    planes: np.ndarray       # (21, 8, 8) float16, a fresh array (never a view of a board)
    actions: np.ndarray      # legal action indices, int32
    pi: np.ndarray           # visit distribution over `actions`, float32
    z: float
    version: int             # model version that produced the game


def play_games(evaluator, scfg: SearchConfig, n_games: int, weights: dict, rng: random.Random,
               np_rng: np.random.Generator, version: int, temp_moves: int = 10,
               max_plies: int = 120, reuse_tree: bool = True, exclude=frozenset()):
    games = []
    for _ in range(n_games):
        kind, board = curriculum.mixture(weights, rng, exclude)
        games.append({"kind": kind, "tree": Tree(board, np_rng), "records": [], "plies": 0,
                      "result": None})
    active = [g for g in games if terminal_value(g["tree"].board) is None]
    while active:
        search([g["tree"] for g in active], evaluator, scfg, noise=True)
        still = []
        for g in active:
            t = g["tree"]
            pi = visit_policy(t.root, 1.0)
            g["records"].append((encode(t.board).astype(np.float16), t.root.actions.astype(np.int32),
                                 pi.astype(np.float32), t.board.turn))
            play = visit_policy(t.root, 1.0 if g["plies"] < temp_moves else 0.0)
            move = t.root.moves[int(np_rng.choice(len(play), p=play))]
            t.advance(move, reuse_tree)
            g["plies"] += 1
            v = terminal_value(t.board)
            if v is not None:
                # v is for the side to move now; convert to White's point of view
                g["result"] = v if t.board.turn == chess.WHITE else -v
            elif g["plies"] >= max_plies:
                g["result"] = "truncated"
            else:
                still.append(g)
        active = still

    samples, stats = [], {"games": n_games, "truncated": 0, "white_win": 0, "draw": 0,
                          "black_win": 0, "plies": 0, "by_kind": {}}
    for g in games:
        stats["plies"] += g["plies"]
        k = stats["by_kind"].setdefault(g["kind"], {"games": 0, "white_win": 0})
        k["games"] += 1
        r = g["result"]
        if r == "truncated":
            stats["truncated"] += 1
            continue
        if r is None:                                    # start position was already over
            continue
        stats["white_win" if r > 0 else "black_win" if r < 0 else "draw"] += 1
        if r > 0:
            k["white_win"] += 1
        samples.extend(assign_labels(g["records"], r, version))
    return samples, stats


def assign_labels(records, white_result: float, version: int) -> list[Sample]:
    """z for each record = the final result from the side to move's point of view."""
    return [Sample(planes, acts, pi, float(white_result if turn == chess.WHITE else -white_result),
                   version)
            for planes, acts, pi, turn in records]
