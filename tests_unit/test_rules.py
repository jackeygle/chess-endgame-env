"""Unit tests for the rules and the reward (no server needed; uses the local Stockfish).

    python3 -m pytest tests_unit -q
"""
import chess
from pathlib import Path

from chessenv.game import Game, IllegalMove, game_reward

KQK = "8/8/k7/8/8/8/5Q1K/8 w - - 0 1"


def test_reward_shape():
    assert game_reward(False, 10, 8, 26) == 0.0
    assert game_reward(True, 8, 8, 26) == 1.0          # fastest mate
    assert game_reward(True, 26, 8, 26) == 0.5         # mate on the last allowed move
    assert 0.5 < game_reward(True, 17, 8, 26) < 1.0
    assert game_reward(True, 5, 8, 26) == 1.0          # faster than "shortest" is capped


def test_illegal_move_costs_nothing():
    g = Game(KQK, 8, 26)
    try:
        try:
            g.play("a1a8")
            assert False, "illegal move accepted"
        except IllegalMove:
            pass
        assert g.moves_used == 0 and g.board.fen() == KQK
    finally:
        g.close()


def test_threefold_repetition_is_a_draw():
    # Kings shuffle; the position repeats a third time -> draw worth 0.
    g = Game("7k/8/8/8/8/8/8/KQ6 w - - 0 1", 1, 30)
    try:
        b = g.board
        for mv in ["a1a2", "h8g8", "a2a1", "g8h8", "a1a2", "h8g8", "a2a1", "g8h8"]:
            b.push_uci(mv)
        assert g._draw_reason() == "draw"
    finally:
        g.close()


def test_stalemate_detected():
    g = Game("7k/8/6Q1/8/8/8/8/K7 w - - 0 1", 1, 30)
    try:
        g.board.push_uci("g6f7")          # Black king on h8 has no moves and is not in check
        assert g.board.is_stalemate()
        assert g._draw_reason() == "stalemate"
    finally:
        g.close()


def test_mate_wins_and_ends_game():
    g = Game("7k/8/6K1/8/8/8/8/Q7 w - - 0 1", 1, 10)
    try:
        out = g.play("a1a8")
        assert out["result"] == "mate" and g.won and g.reward() == 1.0
        try:
            g.play("a8a7")
            assert False, "move accepted after the game ended"
        except IllegalMove:
            pass
    finally:
        g.close()


def test_move_limit_ends_game_with_zero():
    g = Game(KQK, 8, 1)
    try:
        g.play("f2f3")
        assert g.result == "move_limit" and g.reward() == 0.0
    finally:
        g.close()


# ---- full-game mode ------------------------------------------------------------------ #
START = chess.STARTING_FEN


def test_full_mode_rewards():
    from chessenv.game import FULL_REWARD
    assert FULL_REWARD["mate"] == 1.0 and FULL_REWARD["adjudicated_win"] == 1.0
    assert FULL_REWARD["draw"] == 0.5 and FULL_REWARD["adjudicated_draw"] == 0.5
    assert FULL_REWARD.get("lost", 0.0) == 0.0 and FULL_REWARD.get("adjudicated_loss", 0.0) == 0.0


def test_full_mode_unfinished_scores_zero():
    g = Game(START, None, 80, mode="full", opponent_nodes=1)
    try:
        g.play("e2e4")
        assert g.result is None and g.reward() == 0.0
    finally:
        g.close()


def test_full_mode_win_by_mate():
    g = Game("7k/8/6K1/8/8/8/8/Q7 w - - 0 1", None, 80, mode="full", opponent_nodes=1)
    try:
        g.play("a1a8")
        assert g.result == "mate" and g.reward() == 1.0
    finally:
        g.close()


def test_full_mode_adjudication_at_move_limit():
    # White is a queen up when the single allowed move is used: adjudicated win.
    g = Game("4k3/8/8/8/8/8/8/3QK3 w - - 0 1", None, 1, mode="full", opponent_nodes=1)
    try:
        g.play("d1d2")
        assert g.result == "adjudicated_win" and g.reward() == 1.0
    finally:
        g.close()
    # Kings only would be insufficient material; a bare rook down is an adjudicated loss.
    g = Game("3rk3/8/8/8/8/8/8/4K3 w - - 0 1", None, 1, mode="full", opponent_nodes=1)
    try:
        g.play("e1f2")
        assert g.result in ("adjudicated_loss", "lost") and g.reward() == 0.0
    finally:
        g.close()


# ---- per-trial variation: random opponent and seeded openings -------------------------- #
def _random_game(seed, n=6):
    g = Game(START, None, 80, mode="full", opponent="random", seed=seed)
    try:
        for _ in range(n):
            if g.done:
                break
            g.play(sorted(m.uci() for m in g.board.legal_moves)[0])
        return [h["reply"] for h in g.history]
    finally:
        g.close()


def test_random_opponent_is_reproducible_from_its_seed_only():
    assert _random_game("7:g1") == _random_game("7:g1")
    assert _random_game("7:g1") != _random_game("8:g1")


def test_unknown_opponent_kind_is_rejected():
    import pytest
    with pytest.raises(ValueError):
        Game(START, None, 80, mode="full", opponent="human")


def _server_positions(seed):
    import importlib.util
    import os
    import sys
    from pathlib import Path
    srv = Path(__file__).resolve().parent.parent / "task/chess-full/environment/mcp-server"
    os.environ["CHESS_SEED"] = str(seed)
    sys.path.insert(0, str(srv))
    try:
        spec = importlib.util.spec_from_file_location(f"srv{seed}", srv / "server.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.POSITIONS
    finally:
        sys.path.remove(str(srv))
        del os.environ["CHESS_SEED"]


def test_openings_are_seeded_distinct_and_balanced():
    a, b, c = _server_positions(1), _server_positions(1), _server_positions(2)
    assert [p["fen"] for p in a] == [p["fen"] for p in b]               # same seed, same games
    assert [p["fen"] for p in a] != [p["fen"] for p in c]               # new seed, new openings
    assert len({p["fen"] for p in a}) == len(a)                         # a different one per game
    for p in a:
        board = chess.Board(p["fen"])
        assert board.turn == chess.WHITE and board.fullmove_number > 1
        replay = chess.Board()
        for san in p["opening_moves"].split():
            replay.push_san(san)
        assert replay.fen() == p["fen"]


# ---- one-move lookahead (analysis and the hint ablation) -------------------------------- #
def test_hanging_after_finds_pieces_left_en_prise():
    from chessenv.tactics import hanging_after
    b = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3")
    out = hanging_after(b, chess.Move.from_uci("f3g5"))            # Ng5?? Qxg5
    assert [x["square"] for x in out] == ["g5"] and not out[0]["defended"]
    assert hanging_after(b, chess.Move.from_uci("f1c4")) == []      # Bc4 is safe
    # Defended, but attacked by a cheaper piece: a queen on d5 attacked by the pawn on e6.
    b = chess.Board("rnbqkbnr/pppp1ppp/4p3/8/8/2N5/PPPPPPPP/R1BQKBNR w KQkq - 0 2")
    out = hanging_after(b, chess.Move.from_uci("c3d5"))
    assert out and out[0]["piece"] == "N" and out[0]["defended"] is False


def test_hint_tool_is_off_unless_enabled():
    src = (Path(__file__).resolve().parent.parent
           / "task/chess-full/environment/mcp-server/server.py").read_text()
    assert 'if os.environ.get("CHESS_HINTS") == "1":' in src
    for t in ("chess-full", "chess-endgame"):
        compose = (Path(__file__).resolve().parent.parent
                   / f"task/{t}/environment/docker-compose.yaml").read_text()
        assert "CHESS_HINTS" not in compose
