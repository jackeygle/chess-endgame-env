"""Evaluate checkpoints as White against the environment's Stockfish defender.

    python3 -m alphazero.evaluate --models runs/az/model_000.pkl runs/az/model_050.pkl

Uses the environment's own Game class (rules, defender, move limits), search without noise
and the most-visited move. Two position sets:
  * held-out curriculum positions (alphazero/test_positions.json), scored by mate rate;
  * the six positions of the endgame task (positions.json), scored by the task's reward.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch

from alphazero.mcts import Evaluator, SearchConfig, Tree, search
from alphazero.net import PolicyValueNet
from chessenv.game import Game

LIMITS = {"mate1": 5, "mate2": 8, "KQK": 30, "KRK": 40}


def load(path, device):
    with open(path, "rb") as f:
        st = pickle.load(f)
    net = PolicyValueNet(**st["config"]["net"]).to(device)
    net.load_state_dict(st["model"])
    net.eval()
    return net


def play_all(net, device, positions, sims, defender_nodes, c_puct=1.5):
    """positions: dicts with fen, move_limit and optional mate_len. Games run in lockstep."""
    ev = Evaluator(net, device)
    cfg = SearchConfig(sims=sims, c_puct=c_puct)
    rng = np.random.default_rng(0)
    games = [Game(p["fen"], p.get("mate_len"), p["move_limit"], opponent_nodes=defender_nodes)
             for p in positions]
    try:
        active = list(range(len(games)))
        while active:
            trees = [Tree(games[i].board.copy(), rng) for i in active]
            search(trees, ev, cfg, noise=False)
            for i, t in zip(active, trees):
                games[i].play(t.root.moves[int(np.argmax(t.root.N))].uci())
            active = [i for i in active if not games[i].done]
        return [{"result": g.result, "moves": g.moves_used,
                 "reward": g.reward() if g.mate_len is not None else None} for g in games]
    finally:
        for g in games:
            g.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--sims", type=int, default=128)
    ap.add_argument("--defender-nodes", type=int, default=100_000)
    ap.add_argument("--test-set", default="alphazero/test_positions.json")
    ap.add_argument("--task-positions", default="positions.json")
    ap.add_argument("--out")
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    held = json.loads(Path(a.test_set).read_text())
    for p in held:
        p["move_limit"] = LIMITS[p["kind"]]
    task = json.loads(Path(a.task_positions).read_text())
    report = []
    for m in a.models:
        net = load(m, device)
        h = play_all(net, device, held, a.sims, a.defender_nodes)
        by_kind = {}
        for p, r in zip(held, h):
            k = by_kind.setdefault(p["kind"], {"n": 0, "mates": 0})
            k["n"] += 1
            k["mates"] += r["result"] == "mate"
        t = play_all(net, device, task, a.sims, a.defender_nodes)
        rec = {"model": m, "sims": a.sims, "defender_nodes": a.defender_nodes,
               "held_out": by_kind,
               "held_out_mate_rate": sum(k["mates"] for k in by_kind.values()) / len(held),
               "task_reward": float(np.mean([r["reward"] for r in t])),
               "task_games": [{"id": p["id"], "class": p["class"], **r} for p, r in zip(task, t)]}
        report.append(rec)
        print(json.dumps(rec), flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
