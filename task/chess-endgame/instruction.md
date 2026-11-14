# Win the won endgames

You play **White** in six chess games. Every starting position is a **forced win** for
White: Stockfish has verified a checkmate in each one. Your job is to actually deliver
checkmate against a defender (Stockfish) that resists as long as it can.

Use the `chess` MCP server:

- `list_games()` — the six games, their material, the length of the shortest forced mate,
  your move limit and their status.
- `get_position(game_id)` — the board (White is uppercase), FEN, every legal move, moves
  used so far.
- `make_move(game_id, move)` — play one move as UCI (`e2e4`, `d7d8q`) or SAN (`Qh5+`).
  Black replies immediately and you get the new position back. Illegal moves are rejected
  and cost nothing. Moves cannot be taken back.

## Scoring

Each game scores between 0 and 1, and your reward is the average over all six games:

- **Checkmate within the move limit:** 0.5 to 1.0 — 1.0 if you mate as fast as the shortest
  forced mate, falling to 0.5 if you mate on your last allowed move.
- **Anything else scores 0:** stalemate, a draw by threefold repetition or the fifty-move
  rule, running out of moves, being checkmated, or leaving the game unfinished.

A game counts as drawn as soon as a position repeats for the third time or fifty moves pass
without a capture or pawn move, so keep making progress.

Play all six games; a game you never start scores 0. You have no chess engine or chess
library here and no internet access beyond your model.
