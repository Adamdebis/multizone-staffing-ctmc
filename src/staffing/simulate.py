"""Four-zone simulation comparing three staffing policies.

Time step: one minute. Arrivals per minute in each zone are Poisson with a
time-varying rate; each worker clears parcels at rate mu per minute.
Every `review` minutes a policy may move workers between zones.

Policies
--------
fixed        : staff never move.
reactive     : move a worker into any zone whose current queue exceeds a
               fixed length, taking one from the shortest queue.
probability  : for each zone compute P(backlog reached within `horizon`
               minutes) from the CTMC model at the current queue and current
               arrival rate. Zones above `p_in` (70%) request a worker; zones
               below `p_out` (30%) may give one. A small integer programme
               chooses the moves.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from .ctmc import prob_backlog_within


@dataclass
class Config:
    zones: int = 4
    staff: tuple = (5, 5, 5, 5)
    min_staff: int = 2
    mu: float = 0.5                 # parcels per worker per minute
    backlog: int = 40               # queue length counted as "in backlog"
    cap: int = 120                  # truncation K of the CTMC
    review: int = 15                # minutes between decisions
    horizon: float = 30.0           # forecast window, minutes
    p_in: float = 0.70
    p_out: float = 0.30
    reactive_queue: int = 30
    shift_minutes: int = 480
    base_rates: tuple = (2.0, 2.0, 2.0, 2.0)   # arrivals per minute per zone
    seed: int = 0
    surge: list = field(default_factory=lambda: [(1, 120, 210, 1.9),
                                                  (2, 300, 360, 1.7)])


def arrival_rates(cfg: Config) -> np.ndarray:
    """Minute-by-minute arrival rate for each zone.

    Default: a flat base rate with a late-truck pattern, where a zone sees
    little volume and then a surge (zone, start, end, multiplier). Replace
    with rates estimated from data via `rates_from_profile`.
    """
    T = cfg.shift_minutes
    lam = np.tile(np.asarray(cfg.base_rates, float)[:, None], (1, T))
    for z, a, b, m in cfg.surge:
        lam[z, max(a - 30, 0):a] *= 0.3      # late truck: idle period first
        lam[z, a:b] *= m                     # then the volume arrives at once
    return lam


def rates_from_profile(hourly: np.ndarray, zone_share: np.ndarray,
                       scale: float = 1.0) -> np.ndarray:
    """Turn an hourly arrival profile (e.g. from a CSV) into per-minute,
    per-zone rates. `hourly` has one value per hour of the shift."""
    per_min = np.repeat(np.asarray(hourly, float) / 60.0, 60) * scale
    share = np.asarray(zone_share, float)
    return np.outer(share / share.sum(), per_min)


def _ilp_moves(donors: list, receivers: list, p: np.ndarray,
               staff: np.ndarray, min_staff: int) -> list:
    """Choose donor->receiver moves (at most one per receiver) maximising
    receiver risk minus donor risk, keeping every donor above min_staff."""
    pairs = [(i, j) for i in donors for j in receivers if i != j]
    if not pairs:
        return []
    gain = np.array([p[j] - p[i] for i, j in pairs])
    c = -gain                                   # milp minimises
    rows, ub = [], []
    for j in receivers:                         # each receiver gets <= 1
        rows.append([1.0 if pj == j else 0.0 for _, pj in pairs]); ub.append(1)
    for i in donors:                            # donor keeps min_staff
        rows.append([1.0 if pi == i else 0.0 for pi, _ in pairs])
        ub.append(max(int(staff[i]) - min_staff, 0))
    res = milp(c, constraints=LinearConstraint(np.array(rows), -np.inf, ub),
               integrality=np.ones(len(pairs)), bounds=Bounds(0, 1))
    if res.x is None:
        return []
    return [pairs[k] for k, v in enumerate(res.x) if v > 0.5 and gain[k] > 0]


def run(policy: str, cfg: Config, lam: np.ndarray | None = None) -> dict:
    rng = np.random.default_rng(cfg.seed)
    lam = arrival_rates(cfg) if lam is None else lam
    Z, T = lam.shape
    staff = np.array(cfg.staff, int)
    home = staff.copy()
    q = np.zeros(Z)
    queue_log = np.zeros((Z, T)); staff_log = np.zeros((Z, T), int)
    moves = 0
    for t in range(T):
        if t % cfg.review == 0 and t > 0 and policy != "fixed":
            if policy == "reactive":
                over = [z for z in range(Z) if q[z] > cfg.reactive_queue]
                for z in over:
                    donor = min((d for d in range(Z) if d != z and
                                 staff[d] > cfg.min_staff), key=lambda d: q[d],
                                default=None)
                    if donor is not None and q[donor] < cfg.reactive_queue:
                        staff[donor] -= 1; staff[z] += 1; moves += 1
            elif policy == "probability":
                p = np.array([prob_backlog_within(int(q[z]), cfg.horizon,
                              lam[z, t] , cfg.mu, int(staff[z]),
                              cfg.backlog, cfg.cap) for z in range(Z)])
                # send borrowed workers home once their zone is safe again
                for z in range(Z):
                    if staff[z] > home[z] and p[z] < cfg.p_out:
                        lenders = [d for d in range(Z) if staff[d] < home[d]]
                        if lenders:
                            d = max(lenders, key=lambda d: p[d])
                            staff[z] -= 1; staff[d] += 1; moves += 1
                receivers = [z for z in range(Z) if p[z] > cfg.p_in]
                donors = [z for z in range(Z) if p[z] < cfg.p_out
                          and staff[z] > cfg.min_staff]
                for i, j in _ilp_moves(donors, receivers, p, staff,
                                       cfg.min_staff):
                    staff[i] -= 1; staff[j] += 1; moves += 1
        q += rng.poisson(lam[:, t])
        q = np.maximum(q - rng.poisson(cfg.mu * staff), 0)
        queue_log[:, t] = q; staff_log[:, t] = staff
    return {"queue": queue_log, "staff": staff_log, "moves": moves,
            "backlog_minutes": int((queue_log >= cfg.backlog).sum()),
            "mean_queue": float(queue_log.mean()),
            "max_queue": float(queue_log.max())}
