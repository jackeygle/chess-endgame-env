"""Correctness tests for the AlphaZero system: the failure modes that silently break training.

    python3 -m pytest tests_unit/test_alphazero.py -q
"""
import json
import random

import chess
import numpy as np
import pytest
import torch

from alphazero import curriculum
from alphazero.encoding import (N_ACTIONS, cache_key, encode, index_to_move, legal_indices,
                                move_to_index)
from alphazero.mcts import Evaluator, SearchConfig, Tree, search, terminal_value, visit_policy
from alphazero.net import PolicyValueNet
from alphazero.selfplay import assign_labels, play_games
from alphazero.train import batch_tensors, loss_fn

SPECIAL = {
    "castling both sides, both colours": "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1",
    "castling, black to move": "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R b KQkq - 0 1",
    "en passant (white)": "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
    "en passant (black)": "4k3/8/8/8/3Pp3/8/8/4K3 b - d3 0 1",
    "promotions incl. capture (white)": "1n2k3/P7/8/8/8/8/8/4K3 w - - 0 1",
    "promotions (black)": "4k3/8/8/8/8/8/p7/4K3 b - - 0 1",
}


def tiny_net():
    torch.manual_seed(0)
    return PolicyValueNet(blocks=1, channels=16)


@pytest.mark.parametrize("name", SPECIAL)
def test_encoding_round_trip_special_moves(name):
    b = chess.Board(SPECIAL[name])
    moves, idx = legal_indices(b)
    assert len(set(idx.tolist())) == len(moves)                  # one index per move
    for m, i in zip(moves, idx):
        assert 0 <= i < N_ACTIONS and index_to_move(b, int(i)) == m
    if "promotion" in name:
        promos = {m.promotion for m in moves if m.promotion}
        assert promos == {chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN}
    if "castling" in name:
        assert any(b.is_castling(m) and b.is_kingside_castling(m) for m in moves)
        assert any(b.is_castling(m) and b.is_queenside_castling(m) for m in moves)
    if "en passant" in name:
        assert any(b.is_en_passant(m) for m in moves)


def test_encoding_round_trip_random_games():
    rng = random.Random(1)
    for _ in range(30):
        b = chess.Board()
        for _ in range(60):
            moves, idx = legal_indices(b)
            if not moves:
                break
            assert len(set(idx.tolist())) == len(moves)
            for m, i in zip(moves, idx):
                assert index_to_move(b, int(i)) == m
            b.push(rng.choice(moves))


def test_illegal_index_is_rejected():
    b = chess.Board()
    with pytest.raises(ValueError):
        index_to_move(b, move_to_index(b, chess.Move.from_uci("e2e4")) + 1)


def test_priors_are_a_distribution_over_legal_moves():
    b = chess.Board(SPECIAL["promotions incl. capture (white)"])
    moves, acts, p, v = Evaluator(tiny_net())([b])[0]
    assert len(p) == len(moves) == b.legal_moves.count()
    assert np.all(p >= 0) and abs(p.sum() - 1) < 1e-9 and -1 <= v <= 1


def test_training_loss_gives_illegal_moves_zero_probability():
    net = tiny_net()
    b = chess.Board()
    _, acts = legal_indices(b)
    from alphazero.selfplay import Sample
    s = Sample(encode(b).astype(np.float16), acts.astype(np.int32),
               np.full(len(acts), 1 / len(acts), dtype=np.float32), 1.0, 0)
    x, mask, pi, z = batch_tensors([s], "cpu")
    logits, _ = net(x)
    p = torch.softmax(logits.masked_fill(~mask, -1e9), dim=1)
    assert p[~mask].sum().item() == 0.0
    assert abs(p[mask].sum().item() - 1) < 1e-5


def test_encoding_planes():
    b = chess.Board()
    x = encode(b)
    assert x.shape == (21, 8, 8) and x[0].sum() == 8 and x[11].sum() == 1
    assert x[12:16].min() == 1 and x[20].max() == 1               # castling rights, White to move
    b.push_san("e4")
    y = encode(b)                                                  # Black to move: mirrored
    assert y[20].max() == 0 and y[0].sum() == 8
    assert y[16].sum() == 0                                        # e3 is not capturable: no ep plane


def test_cache_key_covers_clock_and_repetition():
    a = chess.Board("4k3/8/8/8/8/8/8/R3K3 w - - 0 1")
    b = chess.Board("4k3/8/8/8/8/8/8/R3K3 w - - 7 1")
    assert cache_key(a) != cache_key(b)
    c = chess.Board("4k3/8/8/8/8/8/8/R3K3 w - - 0 1")
    for m in ["a1a2", "e8d8", "a2a1", "d8e8"]:
        c.push_uci(m)
    assert cache_key(c) != cache_key(a)                            # same placement, repeated once


def test_terminal_values_follow_project_rules():
    assert terminal_value(chess.Board("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1")) == -1.0   # mated
    assert terminal_value(chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")) == 0.0    # stalemate
    assert terminal_value(chess.Board("4k3/8/8/8/8/8/8/R3K3 w - - 100 80")) == 0.0  # fifty moves
    assert terminal_value(chess.Board("4k3/8/8/8/8/8/8/4K3 w - - 0 1")) == 0.0      # bare kings
    # Project rule (chessenv uses the same call): a draw that can be *claimed* ends the game.
    # python-chess allows the claim as soon as the side to move has a move that would make
    # the third occurrence, i.e. one ply before the position actually occurs a third time.
    b = chess.Board("4k3/8/8/8/8/8/8/R3K3 w - - 0 1")
    plies = ["a1a2", "e8d8", "a2a1", "d8e8"] * 2
    for m in plies[:7]:
        assert terminal_value(b) is None
        b.push_uci(m)
    assert terminal_value(b) == 0.0          # d8e8 would repeat the start position a third time
    b.push_uci(plies[7])
    assert b.is_repetition(3) and terminal_value(b) == 0.0
    assert terminal_value(chess.Board()) is None


def test_mcts_finds_mate_in_one_and_backs_up_the_right_sign():
    b = chess.Board("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1")              # Ra8#
    t = Tree(b, np.random.default_rng(0))
    search([t], Evaluator(tiny_net()), SearchConfig(sims=200), noise=False)
    best = t.root.moves[int(np.argmax(t.root.N))]
    assert best == chess.Move.from_uci("a1a8")
    i = t.root.child_index(best)
    assert t.root.W[i] / t.root.N[i] == pytest.approx(1.0)        # mating edge, mover's view


def test_root_noise_is_not_written_into_the_cache():
    net = tiny_net()
    ev = Evaluator(net)
    b = chess.Board()
    t = Tree(b, np.random.default_rng(0))
    search([t], ev, SearchConfig(sims=8, dirichlet_eps=0.5), noise=True)
    assert np.allclose(ev.cache[cache_key(b)][2], t.root.prior)
    assert np.allclose(t.root.prior, ev.__class__(net, use_cache=False)([b])[0][2])


def test_visit_policy():
    from alphazero.mcts import Node
    n = Node()
    n.expand([1, 2, 3], np.array([0, 1, 2]), np.ones(3) / 3)
    n.N[:] = [1, 3, 0]
    assert visit_policy(n, 0).tolist() == [0, 1, 0]
    assert np.allclose(visit_policy(n, 1), [0.25, 0.75, 0])


def test_labels_for_both_sides():
    recs = [(np.zeros((21, 8, 8), np.float16), np.zeros(1, np.int32), np.ones(1, np.float32), t)
            for t in (chess.WHITE, chess.BLACK, chess.WHITE)]
    assert [s.z for s in assign_labels(recs, 1.0, 0)] == [1.0, -1.0, 1.0]     # White won
    assert [s.z for s in assign_labels(recs, -1.0, 0)] == [-1.0, 1.0, -1.0]   # Black won
    assert [s.z for s in assign_labels(recs, 0.0, 0)] == [0.0, 0.0, 0.0]


def test_truncated_games_are_not_used_as_draws():
    ev = Evaluator(tiny_net())
    samples, stats = play_games(ev, SearchConfig(sims=4), 3, {"KRK": 1.0}, random.Random(0),
                                np.random.default_rng(0), version=0, max_plies=1)
    assert stats["truncated"] == 3 and samples == []


def test_samples_are_independent_of_the_board():
    ev = Evaluator(tiny_net())
    samples, stats = play_games(ev, SearchConfig(sims=8), 4, {"mate1": 1.0}, random.Random(0),
                                np.random.default_rng(0), version=7, max_plies=20)
    assert samples and all(s.version == 7 for s in samples)
    first = [s.planes.copy() for s in samples]
    play_games(ev, SearchConfig(sims=8), 4, {"mate1": 1.0}, random.Random(1),
               np.random.default_rng(1), version=8, max_plies=20)
    assert all(np.array_equal(a, s.planes) for a, s in zip(first, samples))
    assert all(abs(s.pi.sum() - 1) < 1e-5 for s in samples)


def test_curriculum_positions_are_verified_by_rules():
    rng = random.Random(0)
    assert curriculum.has_mate_in_1(curriculum.sample("mate1", rng))
    assert curriculum.has_mate_in_2(curriculum.sample("mate2", rng))
    exclude = {curriculum.sample("KQK", random.Random(5)).fen()}
    assert curriculum.sample("KQK", random.Random(5), exclude).fen() not in exclude


def test_checkpoint_resume(tmp_path):
    from alphazero import train
    cfg = {"out": str(tmp_path / "run"), "iters": 1, "device": "cpu", "test_set": "",
           "net": {"blocks": 1, "channels": 8},
           "search": {"sims": 4}, "selfplay": {"games_per_iter": 2, "max_plies": 12},
           "curriculum": {"mate1": 1.0, "mate2": 0, "KQK": 0, "KRK": 0},
           "train": {"batch": 8, "steps_per_iter": 2, "min_buffer": 1}}
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(cfg))
    train.main(["--config", str(p)])
    train.main(["--config", str(p), "--resume", "--iters", "2"])
    lines = (tmp_path / "run" / "metrics.jsonl").read_text().splitlines()
    assert [json.loads(l)["iter"] for l in lines] == [0, 1]
    assert (tmp_path / "run" / "model_002.pkl").exists()


def test_loss_parts_are_finite():
    net = tiny_net()
    ev = Evaluator(net)
    samples, _ = play_games(ev, SearchConfig(sims=8), 4, {"mate1": 1.0}, random.Random(0),
                            np.random.default_rng(0), version=0, max_plies=20)
    loss, parts = loss_fn(net, batch_tensors(samples, "cpu"), 1.0, 1e-4)
    assert torch.isfinite(loss) and all(np.isfinite(v) for v in parts.values())
