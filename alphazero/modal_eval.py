"""Evaluate checkpoints from the Modal volume on a GPU, one model per container, in parallel.

    modal run alphazero/modal_eval.py --models model_000.pkl,model_020.pkl,model_060.pkl

Same evaluation as alphazero/evaluate.py (held-out positions and the six endgame-task
positions, against the environment's Stockfish defender, no search noise). Results go to
results/alphazero/eval_<models>.json.
"""
import json
from pathlib import Path

import modal

app = modal.App("chess-az-eval")
image = (modal.Image.debian_slim(python_version="3.12").apt_install("stockfish")
         .pip_install("torch==2.5.1", "chess==1.11.2", "numpy")
         .add_local_python_source("alphazero", "chessenv"))
vol = modal.Volume.from_name("chess-az")


@app.function(image=image, gpu="T4", cpu=4, memory=8192, timeout=3 * 3600, volumes={"/vol": vol})
def evaluate(model: str, held: list, task: list, sims: int, defender_nodes: int) -> dict:
    import numpy as np
    from alphazero.evaluate import LIMITS, load, play_all
    net = load(f"/vol/az_endgame/{model}", "cuda")
    for p in held:
        p["move_limit"] = LIMITS[p["kind"]]
    h = play_all(net, "cuda", held, sims, defender_nodes)
    by_kind = {}
    for p, r in zip(held, h):
        k = by_kind.setdefault(p["kind"], {"n": 0, "mates": 0})
        k["n"] += 1
        k["mates"] += r["result"] == "mate"
    t = play_all(net, "cuda", task, sims, defender_nodes)
    return {"model": model, "sims": sims, "defender_nodes": defender_nodes, "held_out": by_kind,
            "held_out_mate_rate": sum(k["mates"] for k in by_kind.values()) / len(held),
            "task_reward": float(np.mean([r["reward"] for r in t])),
            "task_games": [{"id": p["id"], "class": p["class"], **r} for p, r in zip(task, t)]}


@app.local_entrypoint()
def main(models: str = "model_000.pkl", sims: int = 128, defender_nodes: int = 100_000):
    held = json.loads(Path("alphazero/test_positions.json").read_text())
    task = json.loads(Path("positions.json").read_text())
    names = models.split(",")
    out = list(evaluate.starmap([(m, held, task, sims, defender_nodes) for m in names]))
    d = Path("results/alphazero")
    d.mkdir(parents=True, exist_ok=True)
    tag = "_".join(n.replace("model_", "").replace(".pkl", "") for n in names)
    (d / f"eval_{tag}.json").write_text(json.dumps(out, indent=1))
    for r in out:
        print(r["model"], "held-out mate rate", f"{r['held_out_mate_rate']:.3f}",
              {k: f"{v['mates']}/{v['n']}" for k, v in r["held_out"].items()},
              "task reward", f"{r['task_reward']:.3f}",
              [f"{g['id']}:{g['result']}" for g in r["task_games"]], flush=True)
