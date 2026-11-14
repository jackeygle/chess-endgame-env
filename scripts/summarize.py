"""Summarise Harbor job folders: agent, reward and per-game outcome.

    python3 scripts/summarize.py jobs
"""
import json
import sys
from pathlib import Path

root = Path(sys.argv[1] if len(sys.argv) > 1 else "jobs")
rows = []
for trial in sorted(root.glob("*/*/")):
    reward_f = trial / "verifier" / "reward.txt"
    cfg_f = trial / "config.json"
    if not reward_f.exists() or not cfg_f.exists():
        continue
    cfg = json.loads(cfg_f.read_text())
    agent = cfg.get("agent", {})
    name = agent.get("name") or agent.get("import_path") or "?"
    model = agent.get("model_name") or ""
    games_f = trial / "verifier" / "games.json"
    detail = ""
    if games_f.exists():
        g = json.loads(games_f.read_text())["games"]
        detail = " ".join(f"{x['game_id']}:{x['status']}" for x in g)
    rows.append((trial.parent.name, name, model, float(reward_f.read_text()), detail))

for job, name, model, r, detail in rows:
    print(f"{job}  {name:28s} {model:32s} reward={r:.3f}  {detail}")
