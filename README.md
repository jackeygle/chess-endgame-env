# Chess against a ladder: an RL environment for planning under a real opponent

**The agent plays four games of chess as White against four opponents, from random moves
to Stockfish.** Each game starts from a well-known balanced opening, a different one in
every game and every trial. The agent gets the board, the FEN and every legal move at
every turn, and plays through three MCP tools. Frontier agents know the rules and the
openings, and they beat a random mover, but they leave pieces where the opponent can take
them on the next move and lose to every Stockfish level that punishes this. The skill
under test is the one a game of chess demands at every move: working out what the
opponent can do next before committing.

| Game | Opponent | Your move limit |
|---|---|---|
| g1 | Level 0: uniformly random legal moves (seeded) | 80 |
| g2 | Level 1 (weak): Stockfish, 1 node per move | 80 |
| g3 | Level 2 (club): Stockfish, 100 nodes per move | 80 |
| g4 | Level 3 (strong): Stockfish, 3 000 nodes per move | 80 |

**Per-trial variation.** Each trial draws one seed (recorded in the score). It picks a
different opening for each game from a book of 39 balanced openings
(`environment/mcp-server/openings.json`, built by `scripts/build_openings.py`) and drives
the random mover. Stockfish opponents run one thread, a fixed hash and a fixed node
budget. So a trial is fully reproducible from its seed (`CHESS_SEED`), but no fixed
sequence of moves wins across trials: an agent cannot memorise a line and replay it.

A second task, `task/chess-endgame`, uses the same server on six won endgames (mate with
queen, rook, two bishops, bishop and knight, queen against rook) and is described at the
end.

## What the agent can do

An MCP server (`chess`) with three tools:

- `list_games()` — opening, opponent, move limit and status of each game
- `get_position(game_id)` — board, FEN, **every legal move**, moves played so far
- `make_move(game_id, move)` — one White move (UCI or SAN); Black replies at once

Illegal moves are rejected and cost nothing; moves cannot be taken back. The rules, the
move limit and the scoring are all in `instruction.md`.

## Scoring

Each game scores 1 for a win, 0.5 for a draw, 0 for a loss; the reward is the mean of the
four games.

- **Win:** checkmate, or a lead of at least three pawns (Stockfish evaluation) when the
  80-move limit is reached.
- **Draw:** stalemate, threefold repetition, fifty-move rule, insufficient material, or an
  evaluation within three pawns at the limit.
- **Loss:** checkmated, three pawns or more behind at the limit, or the game left unfinished.

The score is computed by the game server in its own container. The verifier only reads it
(`tests/score.py`), so nothing the agent writes in its own container can change it.

## Results

Run with Harbor on Modal (`build/chess-full-modal`, identical to `task/chess-full` except
public networking), OpenCode with `openai/gpt-5.5`, 14 Nov 2026. Each run has its own seed,
so its own four openings. Files: `results/v2/`; full Harbor run directories (trajectory,
logs, verifier output) in `results/harbor_runs/`.

| Run | Reward | g1 vs random | g2 vs Level 1 | g3 vs Level 2 | g4 vs Level 3 |
|---|---|---|---|---|---|
| Reference (Stockfish, oracle) | **1.000** | mate, move 13 | mate, move 25 | mate, move 38 | mate, move 24 |
| GPT-5.5, run 1 | 0.375 | mate, move 15 | mated | draw | mated |
| GPT-5.5, run 2 | 0.375 | mate, move 25 | mated | draw | mated |
| GPT-5.5, run 3 | 0.250 | mate, move 32 | mated | mated | mated |
| **GPT-5.5, mean of 3** | **0.333** | 3 wins | 0 of 3 | 2 draws | 0 of 3 |
| Do nothing (nop) | 0.000 | not started | not started | not started | not started |
| Stop early (2 moves) | 0.000 | unfinished | not started | not started | not started |

- **The task is solvable from every opening.** The reference played all 39 book openings
  against all four levels on Modal (`experiments/verify_reference.py`) and won all
  156 games. A 40th opening, where it drew once, was removed from the book.
- **The score has a gradient.** The agent beats the random mover every time and loses to
  every Stockfish level except two draws at Level 2. Three runs are a small sample: they
  show a large, consistent gap (0.250–0.375 against 1.000), not a precise rating.
- **Level 1 beat the agent more often than Level 2.** In all three runs it lost to
  Level 1 but drew twice with Level 2. One node per move is enough for Stockfish to take
  a piece left hanging, and that is how the agent lost those games (below).
- **Earlier version.** The first version had three games from the starting position
  against Levels 1–3, and GPT-5.5 scored 0.056 over three runs (`results/v1/`). That
  scale left no room between agents, and a fixed starting position could be memorised.

## Why the agent loses: measured, not guessed

`scripts/analyze_blunders.py` replays every agent move with Stockfish:
- It takes the evaluation before and after the move; a **blunder** is a move that loses
  three pawns or more.
- A blunder is a **one-move oversight** when the opponent can immediately capture a piece
  that is undefended or worth more than its attacker (`chessenv/tactics.py`). Looking one
  reply ahead would catch it.

| | GPT-5.5 (3 runs, 12 games) | GPT-5.5, first version (9 games) | Reference (4 games) |
|---|---|---|---|
| Agent moves | 444 | 328 | 100 |
| Blunders | 37 (8.3% of moves) | 27 (8.2%) | 1 (in a game it mated) |
| of which one-move oversights | **28 (76%)** | 16 (59%) | 0 |
| Games with at least one blunder | 12 of 12 | 9 of 9 | 1 of 4 |
| Median move of the first blunder | 14 | 14 | — |

The agent never fails on the interface:
- Every legal move is listed at every turn.
- 3 of its 447 move submissions were rejected as illegal, at no cost.
- It finished all 12 games in these runs.

What it gets wrong is the decision: about one move in twelve throws away three pawns or
more, and three in four of those blunders leave a piece the opponent takes on the very
next move.

### Ablation: is one-move lookahead the missing skill?

We gave the same model a fourth tool, `check_move(game_id, move)`. It does not play the
move; it lists which White pieces Black could capture right after it. That is exactly the
one-reply check behind the oversights above. The run used the seed of run 1, so it had
the same four openings. The tool is enabled only in this experiment's copy
(`scripts/make_hint_variant.sh`); the shipped task never offers it, and a unit test
asserts that.

| | Run 1 (no tool) | Same seed, with `check_move` |
|---|---|---|
| Reward | 0.375 | 0.250 (g4 cut off at move 29: API credit ran out) |
| Results g1 / g2 / g3 | win / lost / draw | win / lost / lost |
| Blunders per move | 15 / 179 = 8.4% | 7 / 153 = 4.6% |
| One-move oversights per move | 7 / 179 = 3.9% | 5 / 153 = 3.3% |
| `check_move` calls | — | 206 for 153 moves |

The tool roughly halved the blunder rate but **did not improve the result**. Of the five
one-move oversights it still made:
- three were moves the tool had just flagged as unsafe, which the agent played anyway;
- two were played without checking.

So the agent is not only blind one move ahead. It also does not weigh what it sees, and
it keeps losing to mistakes two or more moves deep. A one-reply lookahead tool cannot
paper over this, which is why the environment measures a real gap.

This is one run cut short by the provider's credit limit: it shows the direction of the
effect, not its size.

Replays of all runs, the ablation included: `results/v2/replays.html`
(`scripts/replay.py`).

## Reward integrity

What we tried, and what happens (`experiments/exploit_tests.py`: 11/11 pass;
`tests_unit/`: 37/37 pass; `scripts/check_submission.py` runs all checks):

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
| Memorising a winning line | openings and the random mover change with every trial's seed |

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
full-game task, then prints a table. Run the agent runs one after another (the script
does): three parallel GPT-5.5 runs exceeded the provider's tokens-per-minute limit. Single runs:

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
| `experiments/` | baseline players; reward-integrity checks; reference check over the opening book |
| `results/` | `v2/` current runs, blunder analyses, ablation, replays; `harbor_runs/` full Harbor run directories; `v1/` first version |
| `tests_unit/` | rule and reward unit tests |
| `baselines/stop_early.py` | the stop-early Harbor agent |
| `scripts/` | one-command evaluation, Modal copies, opening book, blunder analysis, ablation copy, replays, pre-submission checks |
