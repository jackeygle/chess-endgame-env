#!/bin/bash
# Reference run: Stockfish picks every move, played through the same MCP tools as the agent.
set -euo pipefail
/opt/mcp/bin/python /solution/reference.py
