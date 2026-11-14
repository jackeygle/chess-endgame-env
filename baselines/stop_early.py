"""Stop-early baseline: plays two legal moves in the first game, then quits.

    harbor run --path task/chess-endgame --agent-import-path baselines.stop_early:StopEarlyAgent
"""
from typing import override

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

SCRIPT = r'''
import asyncio, json
from mcp import ClientSession
try:
    from mcp.client.streamable_http import streamablehttp_client as client
except ImportError:
    from mcp.client.streamable_http import streamable_http_client as client

def data(r):
    sc = getattr(r, "structuredContent", None)
    return sc.get("result", sc) if sc else json.loads(r.content[0].text)

async def main():
    async with client("http://mcp-server:8000/mcp") as st:
        async with ClientSession(st[0], st[1]) as s:
            await s.initialize()
            for _ in range(2):
                pos = data(await s.call_tool("get_position", {"game_id": "g1"}))
                print(data(await s.call_tool("make_move", {"game_id": "g1",
                                                           "move": pos["legal_moves"][0]})))
asyncio.run(main())
'''


class StopEarlyAgent(BaseAgent):
    @staticmethod
    @override
    def name() -> str:
        return "stop-early"

    @override
    def version(self) -> str:
        return "1.0.0"

    @override
    async def setup(self, environment: BaseEnvironment) -> None:
        pass

    @override
    async def run(self, instruction: str, environment: BaseEnvironment,
                  context: AgentContext) -> None:
        await environment.exec(f"/opt/mcp/bin/python - <<'EOF'\n{SCRIPT}\nEOF")
