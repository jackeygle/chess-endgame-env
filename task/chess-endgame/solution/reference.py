"""Reference run: Stockfish chooses every move, played through the same MCP tools.

The engine runs inside the game server. This script asks it for a move over an
authenticated route (the token exists only in solution/, which Harbor copies into the
container only for oracle runs) and then plays that move with `make_move`, exactly as an
agent would. It never touches the game state any other way.
"""
import asyncio
import os
import json
import urllib.request
from pathlib import Path

from mcp import ClientSession

try:
    from mcp.client.streamable_http import streamablehttp_client
except ImportError:  # renamed in newer mcp releases
    from mcp.client.streamable_http import streamable_http_client as streamablehttp_client

SERVER = os.environ.get("CHESS_SERVER", "http://mcp-server:8000")
TOKEN = (Path(__file__).parent / "oracle_token").read_text().strip()


def engine_move(game_id: str) -> str:
    req = urllib.request.Request(f"{SERVER}/oracle/move",
                                 data=json.dumps({"game_id": game_id, "token": TOKEN}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)["move"]


def payload(result):
    if getattr(result, "structuredContent", None):
        sc = result.structuredContent
        return sc.get("result", sc)
    return json.loads(result.content[0].text)


async def main():
    async with streamablehttp_client(f"{SERVER}/mcp") as streams:
        async with ClientSession(streams[0], streams[1]) as s:
            await s.initialize()
            games = payload(await s.call_tool("list_games", {}))
            for g in games:
                gid = g["game_id"]
                while True:
                    out = payload(await s.call_tool("make_move", {"game_id": gid,
                                                                 "move": engine_move(gid)}))
                    if "error" in out:
                        raise SystemExit(out["error"])
                    if out["result"] is not None:
                        print(gid, out["result"], out["moves_used"], flush=True)
                        break


asyncio.run(main())
