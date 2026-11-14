"""AlphaZero-style training: self-play -> replay buffer -> network update -> evaluation.

    python3 -m alphazero.train --config alphazero/configs/endgame.json
    python3 -m alphazero.train --config alphazero/configs/endgame.json --resume

Loss: L = cross_entropy(pi, p) + beta * (v - z)^2 + l2 * sum(w^2) over conv/linear weights.
L2 is applied only here (Adam's weight_decay stays 0) so it is not applied twice.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import pickle
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from alphazero.encoding import N_ACTIONS
from alphazero.mcts import Evaluator, SearchConfig
from alphazero.net import PolicyValueNet, n_params
from alphazero.selfplay import play_games

DEFAULT = {
    "seed": 0, "device": "auto", "iters": 50, "out": "runs/az_endgame",
    "net": {"blocks": 6, "channels": 64, "wdl": False},
    "search": {"sims": 64, "c_puct": 1.5, "dirichlet_alpha": 0.3, "dirichlet_eps": 0.25,
               "reuse_tree": True, "cache": True},
    "selfplay": {"games_per_iter": 32, "temp_moves": 10, "max_plies": 120},
    "curriculum": {"mate1": 0.3, "mate2": 0.2, "KQK": 0.3, "KRK": 0.2, "start": 0.0},
    "train": {"buffer": 100_000, "batch": 256, "steps_per_iter": 100, "lr": 1e-3,
              "beta": 1.0, "l2": 1e-4, "min_buffer": 512},
    "test_set": "alphazero/test_positions.json",
}


def merge(base, over):
    out = json.loads(json.dumps(base))
    for k, v in over.items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def pick_device(name):
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name


def l2_term(net):
    return sum((p ** 2).sum() for n, p in net.named_parameters() if p.dim() > 1)


def batch_tensors(samples, device):
    x = torch.from_numpy(np.stack([s.planes for s in samples]).astype(np.float32)).to(device)
    mask = torch.zeros(len(samples), N_ACTIONS, dtype=torch.bool)
    pi = torch.zeros(len(samples), N_ACTIONS)
    for i, s in enumerate(samples):
        idx = torch.from_numpy(s.actions.astype(np.int64))
        mask[i, idx] = True
        pi[i, idx] = torch.from_numpy(s.pi)
    z = torch.tensor([s.z for s in samples], dtype=torch.float32)
    return x, mask.to(device), pi.to(device), z.to(device)


def loss_fn(net, batch, beta, l2):
    x, mask, pi, z = batch
    logits, v = net(x)
    logp = F.log_softmax(logits.masked_fill(~mask, -1e9), dim=1)   # legal moves only
    policy = -(pi * logp).sum(1).mean()
    if net.wdl:
        target = torch.where(z > 0, 0, torch.where(z < 0, 2, 1)).long()
        value = F.cross_entropy(v, target)
    else:
        value = ((v - z) ** 2).mean()
    reg = l2_term(net)
    return policy + beta * value + l2 * reg, {"policy": policy.item(), "value": value.item(),
                                             "l2": reg.item()}


def save(path, state):
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(state, f)
    os.replace(tmp, path)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--iters", type=int)
    a = ap.parse_args(argv)
    cfg = merge(DEFAULT, json.loads(Path(a.config).read_text()) if a.config else {})
    if a.iters is not None:
        cfg["iters"] = a.iters
    out = Path(cfg["out"])
    out.mkdir(parents=True, exist_ok=True)
    device = pick_device(cfg["device"])

    net = PolicyValueNet(**cfg["net"]).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=cfg["train"]["lr"], weight_decay=0.0)
    buffer = collections.deque(maxlen=cfg["train"]["buffer"])
    start_iter = 0
    rng, np_rng = random.Random(cfg["seed"]), np.random.default_rng(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    ckpt = out / "checkpoint.pkl"
    if a.resume and ckpt.exists():
        with open(ckpt, "rb") as f:
            st = pickle.load(f)
        net.load_state_dict(st["model"])
        opt.load_state_dict(st["optim"])
        buffer.extend(st["buffer"])
        start_iter = st["iter"] + 1
        rng.setstate(st["py_rng"])
        np_rng.bit_generator.state = st["np_rng"]
        torch.set_rng_state(st["torch_rng"])
        print(f"resumed at iteration {start_iter}, buffer {len(buffer)}", flush=True)
    else:
        (out / "config.json").write_text(json.dumps(cfg, indent=1))
        save(out / "model_000.pkl", {"model": net.state_dict(), "iter": -1, "config": cfg})
    print(f"device {device}, parameters {n_params(net):,}", flush=True)

    exclude = set()
    if cfg.get("test_set") and Path(cfg["test_set"]).exists():
        exclude = {p["fen"] for p in json.loads(Path(cfg["test_set"]).read_text())}
    scfg = SearchConfig(**{k: cfg["search"][k] for k in ("sims", "c_puct", "dirichlet_alpha",
                                                         "dirichlet_eps")})
    tcfg = cfg["train"]
    for it in range(start_iter, cfg["iters"]):
        t0 = time.time()
        version = it                       # self-play uses the weights from before update `it`
        evaluator = Evaluator(net, device, use_cache=cfg["search"]["cache"])
        samples, stats = play_games(evaluator, scfg, cfg["selfplay"]["games_per_iter"],
                                    cfg["curriculum"], rng, np_rng, version,
                                    cfg["selfplay"]["temp_moves"], cfg["selfplay"]["max_plies"],
                                    cfg["search"]["reuse_tree"], exclude)
        buffer.extend(samples)
        t_play = time.time() - t0
        losses = []
        if len(buffer) >= tcfg["min_buffer"]:
            net.train()
            buf = list(buffer)
            for _ in range(tcfg["steps_per_iter"]):
                batch = batch_tensors(rng.sample(buf, min(tcfg["batch"], len(buf))), device)
                loss, parts = loss_fn(net, batch, tcfg["beta"], tcfg["l2"])
                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append(parts)
        rec = {"iter": it, "model_version": version, "samples": len(samples),
               "buffer": len(buffer), "selfplay": stats, "net_evals": evaluator.n_evals,
               "sec_selfplay": round(t_play, 1), "sec_total": round(time.time() - t0, 1)}
        if losses:
            rec["loss"] = {k: float(np.mean([l[k] for l in losses])) for k in losses[0]}
        with open(out / "metrics.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)
        save(out / f"model_{it + 1:03d}.pkl", {"model": net.state_dict(), "iter": it, "config": cfg})
        save(ckpt, {"model": net.state_dict(), "optim": opt.state_dict(), "buffer": list(buffer),
                    "iter": it, "py_rng": rng.getstate(), "np_rng": np_rng.bit_generator.state,
                    "torch_rng": torch.get_rng_state(), "config": cfg})


if __name__ == "__main__":
    main()
