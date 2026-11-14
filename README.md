# Chess against a ladder: an RL environment for planning under a real opponent

**The agent plays three full games of chess as White, from the normal starting position,
against Stockfish at three fixed strengths.** It gets the board, the FEN and every legal
move at every turn, and plays through three MCP tools. Frontier agents know the rules,
the openings and the principles, yet they hang their queen, walk into forks and get
mated by opponents a reference player beats every time. Every move is a decision whose
consequences arrive several moves later: that is the skill this environment trains.

| Game | Opponent | Stockfish node budget per move | Your move limit |
|---|---|---|---|
| g1 | Level 1 (weak) | 1 | 80 |
| g2 | Level 2 (club) | 100 | 80 |
| g3 | Level 3 (strong) | 3 000 | 80 |

Each opponent is a fresh Stockfish process with one thread, a fixed hash and a fixed node
budget, so the same moves always get the same replies: the environment is deterministic
and every difference between runs comes from the agent.

A second task, `task/chess-endgame`, uses the same server on six won endgames (mate with
queen, rook, two bishops, bishop and knight, queen against rook) and is described at the
end.

## What the agent can do

An MCP server (`chess`) with three tools:

- `list_games()` — opponent, move limit and status of each game
- `get_position(game_id)` — board, FEN, **every legal move**, moves played so far
- `make_move(game_id, move)` — one White move (UCI or SAN); Black replies at once

Illegal moves are rejected and cost nothing; moves cannot be taken back. The rules, the
move limit and the scoring are all in `instruction.md`.

## Scoring

Each game scores 1 for a win, 0.5 for a draw, 0 for a loss; the reward is the mean of the
three games.

- **Win:** checkmate, or a lead of at least three pawns (Stockfish evaluation) when the
  80-move limit is reached.
- **Draw:** stalemate, threefold repetition, fifty-move rule, insufficient material, or an
  evaluation within three pawns at the limit.
- **Loss:** checkmated, three pawns or more behind at the limit, or the game left unfinished.

The score is computed by the game server in its own container. The verifier only reads it
(`tests/score.py`), so nothing the agent writes in its own container can change it.

## Results

<!-- RESULTS:TABLE -->

## What the agent gets wrong

<!-- RESULTS:FAILURES -->

## Reward integrity

What we tried, and what happens (`experiments/exploit_tests.py`: 11/11 pass;
`tests_unit/`: 10/10 pass):

| Attempt | Result |
|---|---|
| Do nothing | 0.000 (Harbor `nop` agent) |
| Stop early (two moves, then quit) | 0.000 (`baselines/stop_early.py`) |
| Illegal, malformed or 5 000-character moves | rejected, no move used |
| Unknown or path-like game ids | rejected |
| Moving after a game ended | rejected |
| Stalling until the move limit | adjudicated on the position: stalling does not win |
| Calling the reference-engine route without the token | HTTP 403 |
| Using an engine or chess library | none in the agent's container; `network_mode = "allowlist"` lets it reach only the model API, so nothing can be installed or downloaded |
| Editing reward files in its container | the verifier rewrites them from the server's record |

The reference run asks the server's Stockfish for moves through an authenticated route.
Its token is in `environment/mcp-server/` (the server image) and `solution/`, which
Harbor copies into the agent's container only for oracle runs. An agent in the sandbox
can neither read it nor reach the internet to find it.

## Reproduce

Requirements: Docker (Linux, or macOS with OrbStack: Harbor's allowlist uses nftables,
which Docker Desktop for macOS may lack), Python 3.12+, `pip install harbor`, and the
model provider's API key in the environment (e.g. `OPENAI_API_KEY`).

```bash
MODEL=openai/gpt-5.5 ./scripts/run_eval.sh
```

This runs the reference (oracle), do-nothing (nop), stop-early and three agent runs on the
full-game task, then prints a table. Single runs:

```bash
harbor run --path task/chess-full --agent oracle
harbor run --path task/chess-full --agent opencode --model openai/gpt-5.5
```

Without local Docker, run on Modal: `scripts/make_modal_variant.sh` makes copies under
`build/` that differ only in using public networking (Modal builds the images inside the
sandbox, where the allowlist would block the image pulls), then
`ENV=modal TASK=build/chess-full-modal MODEL=openai/gpt-5.5 ./scripts/run_eval.sh`.
The results above were produced this way.

Replays: `python3 scripts/replay.py out.html "Label=jobs/<job>/<trial>/verifier/games.json" ...`
writes a step-through HTML replay of any runs.

## The endgame task

`task/chess-endgame` gives six positions that are already forced mates for White (shortest
mates of 8 to 24 moves) against a resisting Stockfish. Only mating within the move limit
scores (1.0 for an optimal mate, 0.5 for mating on the last allowed move); stalemate,
repetition, the fifty-move rule and timeouts score 0. The reference mates in all six
(1.000). Kimi K3.1 Pro scored 0.000: it put its queen next to the black king in two games
and lost it, gave away its rook and a bishop, then ended the session with two games
unfinished.

## Repository

| Path | What |
|---|---|
| `task/chess-full/` | the full-game Harbor task (instruction, task.toml, server, verifier, reference) |
| `task/chess-endgame/` | the endgame Harbor task |
| `chessenv/` | game rules, opponent, reward; position generator; baseline players |
| `experiments/` | offline comparison of baseline players; reward-integrity checks |
| `tests_unit/` | rule and reward unit tests |
| `baselines/stop_early.py` | the stop-early Harbor agent |
| `scripts/` | one-command evaluation, Modal copies, results summary, replays |
