# Winning Won Endgames: a chess conversion environment

**The agent is handed six chess positions that are already won, and has to actually
deliver checkmate.** Every start position is a verified forced mate for White. The
defender is Stockfish, deterministic, resisting as long as it can. Frontier agents know
the rules and usually know the theory ("drive the king to the edge"), yet they turn sure
wins into stalemates, repetitions, fifty-move draws and timeouts. Converting a won
position is pure *planning several steps ahead*: no single move wins, and greedy moves
(checks, squeezing the king's squares) go in circles.

| Game | Material | Shortest forced mate | Move limit |
|---|---|---|---|
| g1 | K+Q vs K | 8 | 26 |
| g2 | K+Q vs K | 10 | 30 |
| g3 | K+R vs K | 14 | 38 |
| g4 | K+B+B vs K | 16 | 42 |
| g5 | K+Q vs K+R | 19 | 48 |
| g6 | K+B+N vs K | 24 | 50 |

## What the agent can do

An MCP server (`chess`) with three tools:

- `list_games()` — material, shortest forced mate, move limit and status of each game
- `get_position(game_id)` — board, FEN, **every legal move**, moves used
- `make_move(game_id, move)` — one White move (UCI or SAN); Black replies at once

Illegal moves are rejected and cost nothing; moves cannot be taken back. All rules, the
move limits and the scoring are in `instruction.md`. Nothing is hidden except the future.

## Scoring

Each game scores 0–1; the reward is the mean over the six games.

- Checkmate within the limit: `0.5 + 0.5 * (limit - moves_used) / (limit - shortest_mate)`,
  so 1.0 for an optimal mate and 0.5 for mating on the last allowed move.
- Everything else is 0: stalemate, threefold repetition, fifty-move rule, move limit,
  being mated, or an unfinished game.

The score is computed by the game server, in its own container. The verifier only reads
it (`tests/score.py`), so nothing the agent writes in its container can change it.

## Why this is hard, and what the mistakes look like

<!-- RESULTS:FAILURES -->

## Results

<!-- RESULTS:TABLE -->

## Reward integrity

What we tried, and what happens (`experiments/exploit_tests.py`, 11/11 pass, plus
`tests_unit/`, 6/6):

| Attempt | Result |
|---|---|
| Do nothing | 0.000 |
| Stop early (two moves, then quit) | 0.000 |
| Illegal, malformed or 5 000-character moves | rejected, no move used |
| Unknown or path-like game ids | rejected |
| Moving after a game ended | rejected |
| Shuffling pieces to stall | threefold repetition or fifty-move rule → 0 |
| Calling the reference-engine route without the token | HTTP 403 |
| Using an engine or chess library | none in the agent's container; `network_mode = "allowlist"` lets it reach only the model API, so nothing can be installed or downloaded |
| Editing reward files in its container | the verifier rewrites them from the server's record |

The reference run asks the server's Stockfish for moves through an authenticated route.
Its token exists only in `environment/mcp-server/` (the server image) and `solution/`,
which Harbor copies into the container only for oracle runs; the agent's container never
holds it and cannot reach the internet to find it.

## Reproduce

Requirements: Docker (Linux, or macOS with OrbStack — Harbor's allowlist uses nftables,
which Docker Desktop for macOS may lack), Python 3.12+, `pip install harbor`.

```bash
MODEL=anthropic/claude-sonnet-5-5 ./scripts/run_eval.sh
```

This runs the reference (oracle), do-nothing (nop), stop-early and three agent runs, then
prints a table. Single runs:

```bash
harbor run --path task/chess-endgame --agent oracle
harbor run --path task/chess-endgame --agent opencode --model anthropic/claude-sonnet-5-5
```

Without local Docker, `ENV=modal` runs the same evaluation on Modal; build the Modal copy
first with `scripts/make_modal_variant.sh` (Modal builds images inside the sandbox, so the
copy uses public networking) and point `--path` at `build/chess-endgame-modal`.

## Repository

| Path | What |
|---|---|
| `task/chess-endgame/` | the Harbor task (instruction, task.toml, server, verifier, reference) |
| `chessenv/` | game rules, defender, reward; position generator; baseline players |
| `positions.json` | the six positions and their verified mate lengths |
| `experiments/pilot.py` | offline comparison of baseline players and the reference |
| `experiments/exploit_tests.py` | reward-integrity checks against a live server |
| `tests_unit/` | rule and reward unit tests |
| `baselines/stop_early.py` | the stop-early Harbor agent |
| `scripts/` | one-command evaluation, Modal copy, results summary |
