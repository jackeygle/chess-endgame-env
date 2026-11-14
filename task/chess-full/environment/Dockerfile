# The agent's container. It has no chess library, no engine and (via task.toml) no internet
# except the model API, so the only way to play is to think. OpenCode is installed at build
# time so nothing has to be downloaded at run time.
FROM node:22-bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-venv ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN npm install -g opencode-ai@1.19.5 && opencode --version
# An MCP client for the reference run (solution/). It is only a protocol client; it gives
# no help at chess.
RUN python3 -m venv /opt/mcp && /opt/mcp/bin/pip install --no-cache-dir mcp==2.3.0

WORKDIR /app
