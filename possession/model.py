"""Possession as a two-state decode, not a per-frame vote.

Per frame the only evidence is which team's nearest player is clearly closer to the ball, and
only when the ball is seen and somebody is within reach of it. Frames with no ball, or a ball in
open space, say nothing. A Viterbi pass with a fixed cost per change of team then finds the
labelling that explains the evidence with as few hand-overs as it can justify. Three properties
fall out of that, each of which the majority-vote baseline lacks:

  * possession persists through ball drop-outs and loose balls (no evidence = no change)
  * a change needs sustained control by the other side, not one touch or a deflection
  * the change is placed where the new team's control starts, not after a confirmation delay

Dead ball is an input here (a boolean mask); each live stretch is decoded on its own, so whoever
restarts is decided by what happens after the restart.
"""
import numpy as np

REACH_M = 2.0      # a player further than this from the ball is not evidence of control
SCALE_M = 1.0      # distance advantage (m) at which the evidence saturates
SWITCH_COST = 20.0  # in units of saturated frames: 0.8 s of clear control at 25 fps


def nearest_by_team(ball, players, team):
    """(dH, dA): distance from the ball to the nearest home / away player, inf where unknown."""
    d = np.linalg.norm(players - ball[:, None, :], axis=2)
    d = np.where(np.isnan(d), np.inf, d)
    return d[:, team == 0].min(axis=1), d[:, team == 1].min(axis=1)


def evidence(ball, players, team, reach=REACH_M, scale=SCALE_M):
    """Per-frame score in [-1, 1]; positive favours home, 0 is 'this frame says nothing'."""
    dh, da = nearest_by_team(ball, players, team)
    near = np.minimum(dh, da) < reach
    with np.errstate(invalid="ignore"):
        e = np.clip((np.minimum(da, 1e3) - np.minimum(dh, 1e3)) / scale, -1.0, 1.0)
    return np.where(near, e, 0.0)


def decode(e, cost=SWITCH_COST):
    """Viterbi over {home, away} for one live stretch. Returns 0/1 per frame (0 = home)."""
    n = len(e)
    if n == 0:
        return np.zeros(0, dtype=np.int8)
    back = np.zeros((n, 2), dtype=np.int8)
    sh, sa = e[0], -e[0]
    for t in range(1, n):
        if sa - cost > sh:
            nh, back[t, 0] = sa - cost, 1
        else:
            nh = sh
        if sh - cost > sa:
            na, back[t, 1] = sh - cost, 0
        else:
            na, back[t, 1] = sa, 1
        sh, sa = nh + e[t], na - e[t]
    out = np.empty(n, dtype=np.int8)
    k = 0 if sh >= sa else 1
    for t in range(n - 1, -1, -1):
        out[t] = k
        k = back[t, k]
    return out


def label(ball, players, team, dead, runs, reach=REACH_M, scale=SCALE_M, cost=SWITCH_COST,
          prior=None):
    """H/A/D per frame inside `runs` (list of [lo, hi) graded stretches), X outside.
    prior: optional per-frame team evidence from elsewhere (tagging.py), positive = home."""
    e = evidence(ball, players, team, reach, scale)
    if prior is not None:
        e = np.clip(e + prior, -1.0, 1.0)
    out = np.full(len(ball), "X", dtype="<U1")
    for lo, hi in runs:
        out[lo:hi] = "D"
        live = ~dead[lo:hi]
        edges = np.diff(np.concatenate(([0], live.astype(np.int8), [0])))
        for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            k = decode(e[lo + a:lo + b], cost)
            out[lo + a:lo + b] = np.where(k == 0, "H", "A")
    return out


def baseline_vote(ball, players, team, dead, runs, window=50):
    """The organisers' baseline team logic (nearest player, centred majority vote), given the
    same dead mask, so the comparison isolates the team decision."""
    dh, da = nearest_by_team(ball, players, team)
    seen = ~np.isnan(ball[:, 0])
    h = np.concatenate(([0], np.cumsum(seen & (dh <= da))))
    a = np.concatenate(([0], np.cumsum(seen & (da < dh))))
    i = np.arange(len(ball)); l = np.maximum(0, i - window); u = np.minimum(len(ball), i + window + 1)
    hh, aa = h[u] - h[l], a[u] - a[l]
    vote = np.where(hh + aa == 0, -1, np.where(hh >= aa, 0, 1))
    idx = np.where(vote >= 0, i, -1); c = np.maximum.accumulate(idx); c[c < 0] = np.argmax(vote >= 0)
    vote = vote[c]
    out = np.full(len(ball), "X", dtype="<U1")
    for lo, hi in runs:
        out[lo:hi] = np.where(dead[lo:hi], "D", np.where(vote[lo:hi] == 0, "H", "A"))
    return out
