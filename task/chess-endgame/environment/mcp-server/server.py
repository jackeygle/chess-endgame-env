"""MCP game server: the agent plays White against Stockfish, in the games of positions.json.

positions.json decides the task: won endgames (mode "endgame") or whole games against a
ladder of opponents (mode "full"). The same server runs both tasks.

Per-trial variation (full games): a game whose position says "opening": "book" starts from
an opening drawn from openings.json, each game a different one, and a random-mover
opponent draws its moves from a generator; both are seeded by one trial seed (CHESS_SEED if
set, otherwise random at start-up, recorded in /score). A run can be replayed exactly from
its seed, but a fixed sequence of moves cannot be memorised across trials.

Everything that decides the score lives in this container: the positions, the defender
and the game records. The agent's container only reaches it over HTTP.

Tools (MCP, for the agent):  list_games, get_position, make_move
Routes (HTTP, not tools):    GET /score          read-only results, used by the verifier
                             POST /oracle/move   reference engine move; needs the token that
                                                 only the oracle's solution/ directory holds
"""
from __future__ import annotations

import hmac
import json
import os
import random
import secrets
import threading
from pathlib import Path

import chess
import chess.engine
from fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse

from chessenv.game import Game, IllegalMove, new_engine

HERE = Path(__file__).parent
SEED = int(os.environ.get("CHESS_SEED") or secrets.randbits(32))


def _resolve(positions: list[dict], seed: int) -> list[dict]:
    """Give every "opening": "book" game its own opening, drawn with the trial seed."""
    book_games = [p for p in positions if p.get("opening") == "book"]
    if not book_games:
        return positions
    book = json.loads((HERE / "openings.json").read_text())
    picks = random.Random(f"openings:{seed}").sample(book, len(book_games))
    for p, o in zip(book_games, picks):
        p.update(fen=o["fen"], opening_name=o["name"], opening_moves=o["moves"])
    return positions


POSITIONS = _resolve(json.loads((HERE / "positions.json").read_text()), SEED)
ORACLE_TOKEN = (HERE / "oracle_token").read_text().strip()
REFERENCE_NODES = 2_000_000

mcp = FastMCP("chess-endgame")
_lock = threading.Lock()
_games: dict[str, Game] = {}
_oracle_engines: dict[str, chess.engine.SimpleEngine] = {}


def _game(game_id: str) -> Game:
    pos = next((p for p in POSITIONS if p["id"] == game_id), None)
    if pos is None:
        raise ValueError(f"unknown game_id '{game_id}'. Valid ids: "
                         + ", ".join(p["id"] for p in POSITIONS))
    if game_id not in _games:
        kw = {"mode": pos.get("mode", "endgame"), "opponent": pos.get("opponent_kind", "engine"),
              "seed": f"{SEED}:{game_id}"}
        if "opponent_nodes" in pos:
            kw["opponent_nodes"] = pos["opponent_nodes"]
        _games[game_id] = Game(pos["fen"], pos.get("mate_len"), pos["move_limit"], **kw)
    return _games[game_id]


def _board_text(board: chess.Board) -> str:
    rows = []
    for rank in range(7, -1, -1):
        cells = []
        for file in range(8):
            p = board.piece_at(chess.square(file, rank))
            cells.append(p.symbol() if p else ".")
        rows.append(f"{rank + 1} " + " ".join(cells))
    rows.append("  a b c d e f g h")
    return "\n".join(rows)


def _view(game_id: str, g: Game) -> dict:
    b = g.board
    return {"game_id": game_id, "board": _board_text(b), "fen": b.fen(),
            "to_move": "white" if b.turn else "black", "in_check": b.is_check(),
            "legal_moves": sorted(m.uci() for m in b.legal_moves) if not g.done else [],
            "moves_used": g.moves_used, "move_limit": g.move_limit,
            "result": g.result or "in_progress",
            "moves_so_far": [f"{h['move']} {h['reply'] or ''}".strip() for h in g.history]}


def _summary(p: dict) -> dict:
    g = _games.get(p["id"])
    out = {"game_id": p["id"], "material": p["class"], "move_limit": p["move_limit"],
           "status": "not_started" if g is None else (g.result or "in_progress"),
           "moves_used": 0 if g is None else g.moves_used}
    if p.get("mode", "endgame") == "endgame":
        out["shortest_mate_in"] = p["mate_len"]
    else:
        out["opponent"] = p["opponent"]
        if "opening_name" in p:
            out["opening"] = f"{p['opening_name']}: {p['opening_moves']}"
    return out


@mcp.tool()
def list_games() -> list[dict]:
    """List the games with your move limit and status (and the opponent or shortest mate)."""
    with _lock:
        return [_summary(p) for p in POSITIONS]


@mcp.tool()
def get_position(game_id: str) -> dict:
    """Show a game's board (White = uppercase, you), FEN, legal moves and move count."""
    with _lock:
        return _view(game_id, _game(game_id))


@mcp.tool()
def make_move(game_id: str, move: str) -> dict:
    """Play one White move (UCI like 'e2e4' or SAN like 'Qh5+'). Black replies at once.

    Illegal moves are rejected and cost nothing. A finished game cannot be changed.
    """
    with _lock:
        g = _game(game_id)
        try:
            out = g.play(move)
        except IllegalMove as e:
            return {"error": str(e)}
        if g.done:
            g.close()
            if game_id in _oracle_engines:
                _oracle_engines.pop(game_id).quit()
        return {**out, "position": _view(game_id, g)}


def score() -> dict:
    games = []
    for p in POSITIONS:
        g = _games.get(p["id"])
        games.append({**_summary(p), "reward": 0.0 if g is None else g.reward(),
                      "moves": [] if g is None else g.history, "start_fen": p["fen"]})
    return {"reward": sum(x["reward"] for x in games) / len(POSITIONS), "seed": SEED,
            "games": games}


@mcp.custom_route("/score", methods=["GET"])
async def score_route(request: Request) -> JSONResponse:
    with _lock:
        return JSONResponse(score())


@mcp.custom_route("/oracle/move", methods=["POST"])
async def oracle_route(request: Request) -> JSONResponse:
    body = await request.json()
    if not hmac.compare_digest(str(body.get("token", "")), ORACLE_TOKEN):
        return JSONResponse({"error": "forbidden"}, status_code=403)
    with _lock:
        g = _game(str(body.get("game_id")))
        if g.done:
            return JSONResponse({"error": "game over"}, status_code=400)
        gid = str(body.get("game_id"))
        if gid not in _oracle_engines:          # one fresh engine per game: no state carried over
            _oracle_engines[gid] = new_engine()
        move = _oracle_engines[gid].play(g.board, chess.engine.Limit(nodes=REFERENCE_NODES)).move
        return JSONResponse({"move": move.uci()})


if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
