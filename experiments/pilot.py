"""Play each policy on every position and report the reward per class.

    python3 -m experiments.pilot --positions positions.json
"""
from __future__ import annotations

import argparse
import json
import random
import statistics as st
from collections import defaultdict

from chessenv.game import Game
from chessenv.policies import Reference, checker, random_policy, safe_squeeze, squeeze


def move_limit(mate_len: int) -> int:
    return min(50, 2 * mate_len + 10)


def play(policy, pos, seed=0):
    rng = random.Random(seed)
    g = Game(pos["fen"], pos["mate_len"], move_limit(pos["mate_len"]))
    try:
        while not g.done:
            g.play(policy(g.board, rng).uci())
        return g.reward(), g.result, g.moves_used
    finally:
        g.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--positions", required=True)
    ap.add_argument("--policies", nargs="+", default=["random", "checker", "squeeze", "reference"])
    a = ap.parse_args()
    positions = json.load(open(a.positions))
    ref = Reference()
    pols = {"random": random_policy, "checker": checker, "squeeze": squeeze,
            "safe_squeeze": safe_squeeze, "reference": ref}
    try:
        for name in a.policies:
            per_class, results = defaultdict(list), defaultdict(int)
            for p in positions:
                r, res, n = play(pols[name], p)
                per_class[p["class"]].append(r)
                results[res] += 1
            allr = [x for v in per_class.values() for x in v]
            cls = "  ".join(f"{k}={st.mean(v):.2f}" for k, v in per_class.items())
            print(f"{name:10s} mean={st.mean(allr):.3f}  {cls}  outcomes={dict(results)}", flush=True)
    finally:
        ref.close()


if __name__ == "__main__":
    main()
