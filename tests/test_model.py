import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from staffing.ctmc import generator, prob_backlog_at, prob_backlog_within
from staffing.simulate import Config, run


def test_generator_rows_sum_to_zero():
    Q = generator(2.0, 0.5, 4, 50)
    assert np.allclose(Q.sum(axis=1), 0.0)


def test_probabilities_are_valid_and_ordered():
    for t in [1, 15, 30, 60]:
        a = prob_backlog_at(20, t, 2.6, 0.5, 5, 40, 120)
        w = prob_backlog_within(20, t, 2.6, 0.5, 5, 40, 120)
        assert 0 <= a <= 1 and 0 <= w <= 1
        assert w >= a - 1e-9          # reaching B by t includes being there at t


def test_first_passage_increases_with_horizon():
    w = [prob_backlog_within(20, t, 2.6, 0.5, 5, 40, 120) for t in (5, 15, 30)]
    assert w[0] <= w[1] <= w[2]


def test_simulation_is_reproducible_and_conserves_staff():
    a, b = run("probability", Config(seed=3)), run("probability", Config(seed=3))
    assert np.array_equal(a["queue"], b["queue"])
    assert (a["staff"].sum(axis=0) == sum(Config().staff)).all()
