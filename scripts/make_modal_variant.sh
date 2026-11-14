#!/usr/bin/env bash
# Modal builds the task's images inside the sandbox, where the allowlist would also block
# the image pulls. This makes copies for Modal runs that differ only in the network mode.
# The shipped tasks (task/*) keep the allowlist for local Docker.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p build
for t in chess-full chess-endgame; do
  rm -rf "build/$t-modal" && cp -r "task/$t" "build/$t-modal"
  sed -i 's/^network_mode = "allowlist"/network_mode = "public"/; /^allowed_hosts = \[\]/d' \
    "build/$t-modal/task.toml"
  echo "build/$t-modal: $(grep network_mode "build/$t-modal/task.toml")"
done
