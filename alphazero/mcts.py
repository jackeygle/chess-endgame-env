"""PUCT Monte Carlo tree search with batched leaf evaluation across many trees.

Value convention:
  * a network value belongs to the side to move at the evaluated node;
  * an edge's W and Q (stored on the parent) belong to the parent's side to move;
  * so the value backed up into the edge that leads to a leaf is minus the leaf's value,
    and the sign flips again at every level above.

Terminal positions are scored by the rules (`terminal_value`), never by the network.

Batching: `search` advances one simulation in every tree, sends all the leaves that need
the network as one batch, then expands and backs up. One pending leaf per tree means no
virtual loss is needed, so there is no in-flight state to clean up on errors.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import chess
import numpy as np
import torch

from alphazero.encoding import cache_key, encode, legal_indices


def terminal_value(board: chess.Board) -> float | None:
    """Exact result for the side to move, or None if the game goes on.

    Project rules (chessenv.game): a claimable threefold repetition and the fifty-move
    rule end the game at once as draws, as do stalemate and insufficient material.
    """
    if board.is_checkmate():
        return -1.0
    if (board.is_stalemate() or board.is_insufficient_material()
            or board.halfmove_clock >= 100 or board.can_claim_threefold_repetition()):
        return 0.0
    return None


@dataclass
class Node:
    moves: list = field(default_factory=list)
    actions: np.ndarray | None = None      # action index of each move
    prior: np.ndarray | None = None        # network prior over the legal moves (never noised)
    N: np.ndarray | None = None
    W: np.ndarray | None = None
    children: list = field(default_factory=list)
    terminal: float | None = None
    expanded: bool = False

    def expand(self, moves, actions, prior):
        self.moves, self.actions, self.prior = moves, actions, prior
        self.N = np.zeros(len(moves), dtype=np.float64)
        self.W = np.zeros(len(moves), dtype=np.float64)
        self.children = [None] * len(moves)
        self.expanded = True

    def child_index(self, move) -> int | None:
        for i, m in enumerate(self.moves):
            if m == move:
                return i
        return None


class Evaluator:
    """Batched network evaluation with a cache of raw (un-noised) priors and values."""

    def __init__(self, net, device="cpu", cache_size: int = 200_000, use_cache: bool = True):
        self.net, self.device = net, device
        self.cache: dict = {}
        self.cache_size, self.use_cache = cache_size, use_cache
        self.n_evals = 0

    @torch.no_grad()
    def __call__(self, boards: list[chess.Board]):
        out = [None] * len(boards)
        todo, keys = [], []
        for i, b in enumerate(boards):
            k = cache_key(b) if self.use_cache else None
            if k is not None and k in self.cache:
                out[i] = self.cache[k]
            else:
                todo.append(i)
                keys.append(k)
        if todo:
            x = torch.from_numpy(np.stack([encode(boards[i]) for i in todo])).to(self.device)
            self.net.eval()
            logits, v = self.net(x)
            v = self.net.scalar_value(v).float().cpu().numpy()
            logits = logits.float().cpu().numpy()
            self.n_evals += len(todo)
            if self.use_cache and len(self.cache) > self.cache_size:
                self.cache.clear()
            for j, i in enumerate(todo):
                moves, acts = legal_indices(boards[i])
                lg = logits[j, acts].astype(np.float64)
                p = np.exp(lg - lg.max())
                p /= p.sum()                     # softmax over the legal moves only
                res = (moves, acts, p, float(v[j]))
                out[i] = res
                if self.use_cache:
                    self.cache[keys[j]] = res
        return out


@dataclass
class SearchConfig:
    sims: int = 128
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_eps: float = 0.25


class Tree:
    """Search tree for one game. `board` is the real game position (with its move stack)."""

    def __init__(self, board: chess.Board, rng: np.random.Generator):
        self.board = board
        self.root = Node()
        self.rng = rng
        self.root_noise: np.ndarray | None = None

    def advance(self, move: chess.Move, reuse: bool):
        """Play `move` on the real board; keep its subtree if `reuse`."""
        nxt = None
        if reuse and self.root.expanded:
            i = self.root.child_index(move)
            if i is not None and self.root.children[i] is not None:
                nxt = self.root.children[i]
        self.board.push(move)
        self.root = nxt if nxt is not None else Node()
        self.root_noise = None


def _root_prior(tree: Tree, cfg: SearchConfig, noise: bool) -> np.ndarray:
    p = tree.root.prior
    if not noise or len(p) < 2 or cfg.dirichlet_eps <= 0:
        return p
    if tree.root_noise is None or len(tree.root_noise) != len(p):
        tree.root_noise = tree.rng.dirichlet([cfg.dirichlet_alpha] * len(p))
    return (1 - cfg.dirichlet_eps) * p + cfg.dirichlet_eps * tree.root_noise


def _select(node: Node, prior: np.ndarray, c_puct: float) -> int:
    total = node.N.sum()
    q = np.divide(node.W, node.N, out=np.zeros_like(node.W), where=node.N > 0)
    u = c_puct * prior * math.sqrt(1.0 + total) / (1.0 + node.N)
    return int(np.argmax(q + u))


def _backup(path, leaf_value: float):
    v = -leaf_value                          # edge into the leaf belongs to the leaf's parent
    for node, a in reversed(path):
        node.N[a] += 1
        node.W[a] += v
        v = -v


def search(trees: list[Tree], evaluator: Evaluator, cfg: SearchConfig, noise: bool,
           sims: int | None = None):
    """Run `sims` simulations in every tree (roots are expanded first)."""
    sims = cfg.sims if sims is None else sims
    # expand unexpanded, non-terminal roots in one batch
    need = [t for t in trees if not t.root.expanded and terminal_value(t.board) is None]
    if need:
        for t, (moves, acts, p, _) in zip(need, evaluator([t.board for t in need])):
            t.root.expand(moves, acts, p)
    live = [t for t in trees if t.root.expanded and len(t.root.moves) > 0]
    for _ in range(sims):
        pending = []
        for t in live:
            node, board, path = t.root, t.board.copy(), []
            prior = _root_prior(t, cfg, noise)
            while True:
                a = _select(node, prior, cfg.c_puct)
                path.append((node, a))
                board.push(node.moves[a])
                child = node.children[a]
                if child is None:
                    child = node.children[a] = Node()
                    child.terminal = terminal_value(board)
                if child.terminal is not None:
                    _backup(path, child.terminal)
                    break
                if not child.expanded:
                    pending.append((child, board, path))
                    break
                node, prior = child, child.prior
        if pending:
            for (child, board, path), (moves, acts, p, v) in zip(
                    pending, evaluator([b for _, b, _ in pending])):
                child.expand(moves, acts, p)
                _backup(path, v)


def visit_policy(root: Node, temperature: float) -> np.ndarray:
    """pi(a|s) proportional to N(s,a)^(1/temperature); argmax when temperature == 0."""
    n = root.N
    if temperature <= 1e-6:
        pi = np.zeros_like(n)
        pi[int(np.argmax(n))] = 1.0
        return pi
    x = np.power(n, 1.0 / temperature)
    s = x.sum()
    return x / s if s > 0 else np.full_like(n, 1.0 / len(n))
