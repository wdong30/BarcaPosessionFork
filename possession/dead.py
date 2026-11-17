"""Dead-ball detection: a second two-state decode (live / dead), fed by three kinds of evidence.

  1. How many players are running. Learned from Metrica's public sample games, where the
     corrected tracking marks dead ball exactly (see fit_dead.py for the fit). Needs no ball,
     so it works through the long stretches where the automatic tracking has none.
  2. The ball, when it is seen: moving fast says live, sitting still says dead, off the pitch
     says dead.
  3. Nothing, when the ball is not seen. A missing ball is NOT treated as dead ball -- that is
     the baseline's rule, and the players decide instead.

The per-frame scores are log-likelihood ratios scaled by 1/FPS, i.e. one second of agreeing
frames counts as one observation; frames 40 ms apart are nowhere near independent.
"""
import numpy as np
from model import decode

FPS = 25
LAG = 12                 # frames over which speed is measured (0.48 s): robust to position noise
RUN_MPS = 3.0            # a player faster than this is "running"
RUN_WIN = 75             # +/- frames the running share is averaged over
# logit P(dead) = RUN_B0 + RUN_B1 * share_of_players_running  -- from fit_dead.py on sample game 1
RUN_B0, RUN_B1, PRIOR_DEAD = 1.21, -14.8, 0.39
BALL_FAST, BALL_SLOW = 4.0, 1.0      # m/s
# P(observation | state) for a visible ball. The 'live' column is measured on the sample games
# (fit_dead.py prints it); the 'dead' column is an assumption -- the corrected sample tracking
# has no ball at all while play is dead, so there is nothing to measure it on.
P_FAST = (0.05, 0.61)    # (dead, live)
P_SLOW = (0.70, 0.12)
OUT_M = 1.0              # ball this far outside the lines is out
OUT_LLR = 4.0
SWITCH_COST = 3.0        # nats; with the 1/FPS scaling this is seconds of unanimous evidence


def _moving_mean(x, k):
    c = np.concatenate(([0.0], np.cumsum(np.nan_to_num(x))))
    n = np.concatenate(([0], np.cumsum(~np.isnan(x))))
    i = np.arange(len(x)); l = np.maximum(0, i - k); u = np.minimum(len(x), i + k + 1)
    return (c[u] - c[l]) / np.maximum(1, n[u] - n[l])


def running_share(players):
    """Share of tracked players moving faster than RUN_MPS, averaged over +/- RUN_WIN frames."""
    h = LAG // 2
    v = np.full(players.shape[:2], np.nan)
    v[h:len(v) - (LAG - h)] = np.linalg.norm(players[LAG:] - players[:-LAG], axis=2) / (LAG / float(FPS))
    v[v > 11.0] = np.nan                       # identity switches and teleports, not sprints
    seen = ~np.isnan(v)
    share = np.where(seen.any(axis=1), (np.nan_to_num(v) > RUN_MPS).sum(axis=1) / np.maximum(1, seen.sum(axis=1)), np.nan)
    return _moving_mean(share, RUN_WIN)


def ball_speed(ball):
    h = LAG // 2
    v = np.full(len(ball), np.nan)
    v[h:len(v) - (LAG - h)] = np.linalg.norm(ball[LAG:] - ball[:-LAG], axis=1) / (LAG / float(FPS))
    return v


def dead_llr(ball, players, pitch):
    """Per-frame log-likelihood ratio, positive = dead. Not yet scaled by 1/FPS."""
    run = running_share(players)
    llr = RUN_B0 + RUN_B1 * run - np.log(PRIOR_DEAD / (1 - PRIOR_DEAD))
    v = ball_speed(ball)
    with np.errstate(invalid="ignore"):
        llr = llr + np.where(v > BALL_FAST, np.log(P_FAST[0] / P_FAST[1]), 0.0)
        llr = llr + np.where(v < BALL_SLOW, np.log(P_SLOW[0] / P_SLOW[1]), 0.0)
        out = (ball[:, 0] < -OUT_M) | (ball[:, 0] > pitch[0] + OUT_M) | \
              (ball[:, 1] < -OUT_M) | (ball[:, 1] > pitch[1] + OUT_M)
    return llr + np.where(out, OUT_LLR, 0.0)


def dead_mask(ball, players, pitch, runs, cost=SWITCH_COST, extra_llr=None):
    """Boolean per frame, True = dead ball. Decoded separately inside each graded stretch.
    extra_llr: optional per-frame dead evidence from elsewhere (tagging.py), same units."""
    llr = dead_llr(ball, players, pitch)
    if extra_llr is not None:
        llr = llr + extra_llr
    llr = llr / FPS
    dead = np.zeros(len(ball), dtype=bool)
    for lo, hi in runs:
        dead[lo:hi] = decode(-llr[lo:hi], cost) == 1     # decode(): state 1 wins on negative input
    return dead
