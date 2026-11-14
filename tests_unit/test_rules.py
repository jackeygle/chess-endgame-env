"""Unit tests for the rules and the reward (no server needed; uses the local Stockfish).

    python3 -m pytest tests_unit -q
"""
import chess

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
