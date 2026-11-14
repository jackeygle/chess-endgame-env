"""Train on a Modal T4 GPU. Checkpoints go to the Modal volume `chess-az`.

    modal run --detach alphazero/modal_train.py --config alphazero/configs/endgame.json
    modal volume get chess-az az_endgame runs/          # download models and metrics

Training runs in chunks of iterations and commits the volume after each, so a stopped or
timed-out run resumes from its last checkpoint when started again.
"""
import json
from pathlib import Path

import modal

app = modal.App("chess-az-train")
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.5.1", "chess==1.11.2", "numpy")
         .add_local_python_source("alphazero", "chessenv"))
vol = modal.Volume.from_name("chess-az", create_if_missing=True)


@app.function(image=image, gpu="T4", cpu=8, memory=8192, timeout=4 * 3600, volumes={"/vol": vol})
def train(cfg: dict, test_positions: list, chunk: int = 2):
    from alphazero import train as tr
    cfg = dict(cfg)
    name = Path(cfg.get("out", "runs/az")).name
    cfg["out"] = f"/vol/{name}"
    cfg["test_set"] = "/tmp/test_positions.json"
    Path(cfg["test_set"]).write_text(json.dumps(test_positions))
    Path("/tmp/cfg.json").write_text(json.dumps(cfg))
    total = cfg["iters"]
    done = 0
    metrics = Path(cfg["out"]) / "metrics.jsonl"
    if metrics.exists():
        done = len(metrics.read_text().splitlines())
    while done < total:
        done = min(done + chunk, total)
        args = ["--config", "/tmp/cfg.json", "--iters", str(done)]
        if (Path(cfg["out"]) / "checkpoint.pkl").exists():
            args.append("--resume")
        tr.main(args)
        vol.commit()


@app.local_entrypoint()
def main(config: str = "alphazero/configs/endgame.json"):
    cfg = json.loads(Path(config).read_text())
    tests = json.loads(Path("alphazero/test_positions.json").read_text())
    train.remote(cfg, tests)
