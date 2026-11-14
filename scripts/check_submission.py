"""Pre-submission checks, grouped by the judging criteria. Needs no Docker.

    python3 scripts/check_submission.py          # everything (~10 minutes)
    python3 scripts/check_submission.py --quick  # skip the full-game reference replay

Each check prints PASS or FAIL; the exit code is non-zero if any check fails.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import socket
import subprocess
import sys
import time
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TASKS = ["chess-full", "chess-endgame"]
results: list[tuple[str, str, bool, str]] = []


def check(group, name, ok, detail=""):
    results.append((group, name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {group}: {name}" + (f" -- {detail}" if detail else ""),
          flush=True)


def run(cmd, env=None, timeout=1800):
    return subprocess.run(cmd, cwd=ROOT, env={**os.environ, **(env or {})}, timeout=timeout,
                          capture_output=True, text=True)


class Server:
    def __init__(self, task, port, seed=None):
        self.task, self.port, self.seed = task, port, seed

    def __enter__(self):
        self.p = subprocess.Popen([sys.executable, "server.py"],
                                  cwd=ROOT / "task" / self.task / "environment" / "mcp-server",
                                  env={**os.environ, "PORT": str(self.port),
                                       **({"CHESS_SEED": str(self.seed)} if self.seed else {})},
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(60):
            try:
                socket.create_connection(("127.0.0.1", self.port), 1).close()
                return self
            except OSError:
                time.sleep(1)
        raise RuntimeError(f"server for {self.task} did not start")

    def __exit__(self, *exc):
        self.p.terminate()
        self.p.wait(timeout=30)
        time.sleep(1)


def score(task, port, logs):
    logs.mkdir(parents=True, exist_ok=True)
    r = run([sys.executable, f"task/{task}/tests/score.py"],
            {"CHESS_SERVER": f"http://127.0.0.1:{port}", "VERIFIER_LOGS": str(logs)})
    return float((logs / "reward.txt").read_text()), json.loads((logs / "games.json").read_text())


def craft(tmp):
    from harbor.models.task.config import TaskConfig
    for t in TASKS:
        d = ROOT / "task" / t
        need = ["instruction.md", "task.toml", "environment/Dockerfile", "tests/test.sh",
                "solution/solve.sh", "environment/docker-compose.yaml"]
        missing = [n for n in need if not (d / n).exists()]
        check("Craft", f"{t}: Harbor task layout complete", not missing, ", ".join(missing))
        try:
            cfg = TaskConfig.model_validate(tomllib.loads((d / "task.toml").read_text()))
            check("Craft", f"{t}: task.toml valid for Harbor", True,
                  f"{cfg.task.name}, network {cfg.environment.network_mode.value}")
        except Exception as e:                                    # noqa: BLE001
            check("Craft", f"{t}: task.toml valid for Harbor", False, str(e)[:200])
        same = (d / "environment/mcp-server/chessenv/game.py").read_bytes() == \
            (ROOT / "chessenv/game.py").read_bytes()
        check("Craft", f"{t}: server ships the current rules (chessenv/game.py)", same)
    readme = (ROOT / "README.md").read_text()
    check("Craft", "README gives the one-command evaluation", "./scripts/run_eval.sh" in readme)
    r = run(["bash", "-n", "scripts/run_eval.sh"])
    check("Craft", "run_eval.sh parses", r.returncode == 0, r.stderr[:200])
    r = run([sys.executable, "-m", "pytest", "tests_unit", "-q"], timeout=1800)
    tail = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-200:]
    check("Craft", "unit tests (rules, reward, AlphaZero)", r.returncode == 0, tail)


def integrity(tmp):
    for t, port in (("chess-endgame", 8781), ("chess-full", 8782)):
        with Server(t, port):
            rew, _ = score(t, port, tmp / f"{t}-nop")
            check("Reward integrity", f"{t}: doing nothing scores 0", rew == 0.0, f"{rew}")
        with Server(t, port):
            r = run([sys.executable, "-m", "experiments.exploit_tests"],
                    {"CHESS_SERVER": f"http://127.0.0.1:{port}"})
            tail = [l for l in r.stdout.splitlines() if "checks passed" in l]
            check("Reward integrity", f"{t}: exploit checks (illegal moves, ids, stalling, "
                  "token, early stop, finished games)", r.returncode == 0,
                  tail[0] if tail else r.stdout[-300:] + r.stderr[-300:])
    for t in TASKS:
        toml = (ROOT / "task" / t / "task.toml").read_text()
        check("Reward integrity", f"{t}: agent network restricted to the model API",
              'network_mode = "allowlist"' in toml)
        df = (ROOT / "task" / t / "environment/Dockerfile").read_text().lower()
        check("Reward integrity", f"{t}: no engine or chess library in the agent's image",
              "stockfish" not in df and "chess" not in re.sub(r"#.*", "", df))
    pattern = re.compile(r"sk-proj-|sk-[A-Za-z0-9]{32,}|dc_[0-9a-f]{40,}|\bak-[A-Za-z0-9]{20,}|"
                         r"\bas-[A-Za-z0-9]{20,}")
    files = run(["git", "ls-files"]).stdout.split()
    hits = [f for f in files if f != "scripts/check_submission.py" and (ROOT / f).is_file() and (ROOT / f).stat().st_size < 5_000_000
            and pattern.search((ROOT / f).read_text(errors="ignore"))]
    check("Reward integrity", "no API keys committed", not hits, ", ".join(hits))


def gap(tmp, quick):
    tasks = ["chess-endgame"] + ([] if quick else ["chess-full"])
    for t in tasks:
        moves = []
        for k in range(2):
            port = 8790 + k
            with Server(t, port, seed=20261115):
                r = run([sys.executable, f"task/{t}/solution/reference.py"],
                        {"CHESS_SERVER": f"http://127.0.0.1:{port}"}, timeout=3600)
                rew, games = score(t, port, tmp / f"{t}-ref{k}")
            moves.append([g["moves"] for g in games["games"]])
            check("Gap", f"{t}: reference run {k + 1} scores 1.000", rew == 1.0 and r.returncode == 0,
                  f"{rew}")
        check("Gap", f"{t}: reference is reproducible move for move (same seed)",
              moves[0] == moves[1])
    with Server("chess-full", 8793, seed=1) as _a:
        a = json.loads(run(["curl", "-s", "localhost:8793/score"]).stdout)
    with Server("chess-full", 8794, seed=2) as _b:
        b = json.loads(run(["curl", "-s", "localhost:8794/score"]).stdout)
    check("Gap", "chess-full: a new seed gives new openings (no memorisable line)",
          [g["start_fen"] for g in a["games"]] != [g["start_fen"] for g in b["games"]])
    res = ROOT / "results"
    agent = [float(p.read_text()) for p in sorted(res.glob("v2/gpt55_run*.reward.txt"))]
    ref = float((res / "v2/reference.reward.txt").read_text())
    check("Gap", "recorded agent runs are far below the recorded reference",
          len(agent) == 3 and ref == 1.0 and max(agent) <= 0.5, f"agent {agent}, reference {ref}")
    check("Craft", "a full Harbor run directory is included",
          any((res / "harbor_runs").glob("*/*/agent/trajectory.json")))


def difficulty(tmp):
    for t in TASKS:
        ins = (ROOT / "task" / t / "instruction.md").read_text()
        for word in ("list_games", "get_position", "make_move", "legal", "Scoring"):
            if word not in ins:
                check("Meaningful difficulty", f"{t}: instruction explains '{word}'", False)
                break
        else:
            check("Meaningful difficulty", f"{t}: instruction states tools, legal moves and scoring",
                  True)
    src = (ROOT / "task/chess-full/environment/mcp-server/server.py").read_text()
    check("Meaningful difficulty", "every position view lists all legal moves",
          '"legal_moves"' in src and '"board"' in src and '"fen"' in src)
    check("Meaningful difficulty", "README explains the agent's mistakes",
          "Why the agent loses" in (ROOT / "README.md").read_text()
          and "RESULTS:FAILURES" not in (ROOT / "README.md").read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    tmp = Path(os.environ.get("TMPDIR", "/tmp")) / "chess-check"
    tmp.mkdir(parents=True, exist_ok=True)
    craft(tmp)
    integrity(tmp)
    difficulty(tmp)
    gap(tmp, a.quick)
    failed = [r for r in results if not r[2]]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
