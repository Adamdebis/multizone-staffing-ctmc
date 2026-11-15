"""Continuous-time Markov chain model of one zone's queue.

Each zone is modelled as an M/M/c queue truncated at a buffer limit K:
state n = parcels (or pallets) in the zone, arrivals at rate lam,
service at rate mu per worker with c workers.

Two quantities are computed from the Kolmogorov forward equations
dp/dt = p Q, whose solution is p(t) = p(0) expm(Q t):

1. P(in backlog at time t)       = P(X(t) >= B)
2. P(backlog reached by time t)  = P(tau_B <= t), where tau_B is the first
   time the queue reaches B. This is computed by making states >= B
   absorbing and reading the absorbed mass at time t.

The second is the quantity a staffing trigger actually needs: a backlog that
forms and clears before t is missed by (1) but caught by (2). In general
(2) >= (1).
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import expm


def generator(lam: float, mu: float, c: int, K: int) -> np.ndarray:
    """Generator matrix Q of a birth-death M/M/c/K queue (states 0..K)."""
    if lam < 0 or mu <= 0 or c < 1 or K < 1:
        raise ValueError("need lam >= 0, mu > 0, c >= 1, K >= 1")
    Q = np.zeros((K + 1, K + 1))
    for n in range(K + 1):
        if n < K:
            Q[n, n + 1] = lam
        if n > 0:
            Q[n, n - 1] = mu * min(n, c)
        Q[n, n] = -Q[n].sum()
    return Q


def _start(n0: int, K: int) -> np.ndarray:
    p0 = np.zeros(K + 1)
    p0[min(max(n0, 0), K)] = 1.0
    return p0


def prob_backlog_at(n0: int, t: float, lam: float, mu: float, c: int,
                    B: int, K: int) -> float:
    """P(X(t) >= B | X(0) = n0) from the forward equations."""
    Q = generator(lam, mu, c, K)
    pt = _start(n0, K) @ expm(Q * t)
    return float(np.clip(pt[B:].sum(), 0.0, 1.0))


def prob_backlog_within(n0: int, t: float, lam: float, mu: float, c: int,
                        B: int, K: int) -> float:
    """P(queue reaches B at some time in [0, t] | X(0) = n0).

    States B..K are made absorbing (their rows of Q set to zero), so the
    probability mass sitting in them at time t is the first-passage
    probability.
    """
    if n0 >= B:
        return 1.0
    Q = generator(lam, mu, c, K)
    Q[B:, :] = 0.0
    pt = _start(n0, K) @ expm(Q * t)
    return float(np.clip(pt[B:].sum(), 0.0, 1.0))
