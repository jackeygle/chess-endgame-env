# Win at chess

You play **White** in three full games of chess, each from the normal starting position.
Your opponents are Stockfish at three fixed strengths: Level 1 (weak), Level 2 (club) and
Level 3 (strong). Each opponent is deterministic: the same moves always get the same
replies.

Use the `chess` MCP server:

- `list_games()` — the three games, their opponent, your move limit and their status.
- `get_position(game_id)` — the board (White is uppercase), FEN, every legal move, the moves
  played so far.
- `make_move(game_id, move)` — play one move as UCI (`e2e4`, `e7e8q`) or SAN (`Nf3`, `O-O`).
  Black replies immediately and you get the new position back. Illegal moves are rejected
  and cost nothing. Moves cannot be taken back.

## Scoring

Each game scores 1 for a win, 0.5 for a draw and 0 for a loss; your reward is the average
over the three games.

- **Win (1):** checkmate, or a lead of at least three pawns (Stockfish's evaluation) when
  your move limit is reached.
- **Draw (0.5):** stalemate, threefold repetition, the fifty-move rule, insufficient
  material, or an evaluation within three pawns at the move limit.
- **Loss (0):** being checkmated, trailing by three pawns or more at the move limit, or
  leaving the game unfinished.

Each game allows **80 of your moves**. Play all three games to the end; a game you never
start or abandon scores 0. You have no chess engine or chess library here and no internet
access beyond your model.
