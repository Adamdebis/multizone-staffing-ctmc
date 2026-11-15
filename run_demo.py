"""Reproduce the figures and summary table in results/."""
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent / "src"))
from staffing.ctmc import prob_backlog_at, prob_backlog_within  # noqa: E402
from staffing.simulate import Config, run  # noqa: E402

out = Path("results"); fig = out / "figures"; fig.mkdir(parents=True, exist_ok=True)

# 1. Why "within 30 minutes" is not the same as "at minute 30"
lam, mu, c, B, K, n0 = 2.8, 0.5, 5, 40, 120, 32
ts = np.linspace(0, 60, 61)
at = [prob_backlog_at(n0, t, lam, mu, c, B, K) for t in ts]
within = [prob_backlog_within(n0, t, lam, mu, c, B, K) for t in ts]
plt.figure(figsize=(7, 4))
plt.plot(ts, within, label="P(backlog reached within t)  [first passage]")
plt.plot(ts, at, "--", label="P(in backlog at t)  [state at time t]")
plt.axhline(0.7, color="grey", lw=0.8); plt.axvline(30, color="grey", lw=0.8)
plt.xlabel("minutes ahead"); plt.ylabel("probability")
plt.title(f"One zone: queue {n0}, backlog at {B}, {c} workers, arrivals {lam}/min")
plt.legend(); plt.tight_layout(); plt.savefig(fig / "forward_vs_first_passage.png", dpi=150)
print(f"t=30: at={at[30]:.3f} within={within[30]:.3f}")

# 2. One shift under each policy
cfg = Config()
res = {p: run(p, cfg) for p in ["fixed", "reactive", "probability"]}
f, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
for p, r in res.items():
    ax[0].plot(r["queue"][1], label=p)
    ax[1].plot(r["staff"][1], label=p)
ax[0].axhline(cfg.backlog, color="grey", lw=0.8)
ax[0].set_ylabel("zone 2 queue"); ax[1].set_ylabel("zone 2 workers")
ax[1].set_xlabel("minute of shift"); ax[0].legend()
ax[0].set_title("Zone 2 during a late-truck surge (synthetic arrivals)")
plt.tight_layout(); plt.savefig(fig / "shift_zone2.png", dpi=150)

# 3. Many shifts
rows = []
for p in ["fixed", "reactive", "probability"]:
    bm, mq, mv = [], [], []
    for s in range(200):
        r = run(p, Config(seed=s))
        bm.append(r["backlog_minutes"]); mq.append(r["max_queue"]); mv.append(r["moves"])
    rows.append((p, np.mean(bm), np.mean(mq), np.mean(mv)))
lines = ["policy,mean_zone_minutes_in_backlog,mean_max_queue,mean_staff_moves"]
lines += [f"{p},{a:.1f},{b:.1f},{c:.1f}" for p, a, b, c in rows]
(out / "policy_comparison.csv").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
