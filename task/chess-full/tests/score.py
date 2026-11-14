"""Read the authoritative result from the game server and write Harbor's reward."""
import json
import os
import urllib.request

SERVER = os.environ.get("CHESS_SERVER", "http://mcp-server:8000")
LOGS = os.environ.get("VERIFIER_LOGS", "/logs/verifier")

with urllib.request.urlopen(f"{SERVER}/score", timeout=30) as r:
    result = json.load(r)

reward = float(result["reward"])
if not 0.0 <= reward <= 1.0:
    raise SystemExit(f"reward out of range: {reward}")

with open(f"{LOGS}/games.json", "w") as f:
    json.dump(result, f, indent=1)
with open(f"{LOGS}/reward.txt", "w") as f:
    f.write(f"{reward:.6f}\n")

for g in result["games"]:
    print(f"{g['game_id']} {g['material']:5s} {g['status']:12s} moves={g['moves_used']:3d} "
          f"reward={g['reward']:.3f}")
print(f"reward={reward:.6f}")
