#!/usr/bin/env bash
# Modal builds the task's images inside the sandbox, where the allowlist would also block
# the image pulls. This makes a copy for Modal runs that differs only in the network mode.
# The shipped task (task/chess-endgame) keeps the allowlist for local Docker.
set -euo pipefail
cd "$(dirname "$0")/.."
rm -rf build/chess-endgame-modal && mkdir -p build
cp -r task/chess-endgame build/chess-endgame-modal
sed -i 's/^network_mode = "allowlist"/network_mode = "public"/; /^allowed_hosts = \[\]/d' \
  build/chess-endgame-modal/task.toml
grep -n "network_mode" build/chess-endgame-modal/task.toml
