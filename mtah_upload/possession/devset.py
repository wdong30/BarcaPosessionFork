"""Dev set from Metrica's public sample games 1 and 2 (github.com/metrica-sports/sample-data).

Per-frame labels in the hackathon's vocabulary, built mechanically:
  D  where the corrected tracking has no ball (in these files the ball is NaN exactly when dead)
  H/A otherwise: the team of the most recent on-ball event (pass, shot, recovery, set piece,
      ball lost, ball out). Challenges, fouls and cards do not move possession, so a duel won
      without a following controlled action changes nothing -- the rule in CHALLENGE.md.
No frame of the competition match is involved anywhere in this file.
"""
import os
import numpy as np
import pandas as pd

PITCH = (105.0, 68.0)
ON_BALL = {"PASS", "SHOT", "RECOVERY", "SET PIECE", "BALL LOST", "BALL OUT"}


def _tracking(path):
    t = pd.read_csv(path, skiprows=2)
    period = t["Period"].to_numpy()
    xy = t.iloc[:, 3:].to_numpy(dtype=float)
    n_players = (xy.shape[1] - 2) // 2
    players = xy[:, :2 * n_players].reshape(len(t), n_players, 2)
    ball = xy[:, 2 * n_players:2 * n_players + 2]
    return period, players, ball


def load(game, data_dir):
    """-> dict(ball (N,2) m, players (N,P,2) m, team (P,) 0=home 1=away, labels (N,) str, period)"""
    base = os.path.join(data_dir, "Sample_Game_{}_".format(game))
    period, hp, ball = _tracking(base + "RawTrackingData_Home_Team.csv")
    _, ap, _ = _tracking(base + "RawTrackingData_Away_Team.csv")
    players = np.concatenate([hp, ap], axis=1) * np.array(PITCH)
    team = np.array([0] * hp.shape[1] + [1] * ap.shape[1], dtype=np.int8)
    ball = ball * np.array(PITCH)

    ev = pd.read_csv(base + "RawEventsData.csv")
    ev = ev[ev["Type"].isin(ON_BALL)].sort_values("Start Frame", kind="stable")
    n = len(period)
    owner = np.full(n, -1, dtype=np.int8)
    frames = np.clip(ev["Start Frame"].to_numpy() - 1, 0, n - 1)
    teams = (ev["Team"].to_numpy() == "Away").astype(np.int8)
    owner[frames] = teams                       # later events at the same frame win
    idx = np.where(owner >= 0, np.arange(n), -1)
    carry = np.maximum.accumulate(idx)
    carry[carry < 0] = frames[0]
    owner = owner[carry]
    labels = np.where(np.isnan(ball[:, 0]), "D", np.where(owner == 0, "H", "A"))
    return dict(ball=ball, players=players, team=team, labels=labels, period=period)


def degrade(g, seed, ball_gap_rate=0.10, ball_gap_mean_s=1.0, player_gap_rate=0.07,
            player_gap_mean_s=4.0, sigma_player=0.2, sigma_ball=0.3, fps=25):
    """Make clean corrected tracking look more like automatic tracking: noise, ball drop-outs
    during live play, and players vanishing for a few seconds at a time. Rates are assumptions,
    stated here so nobody mistakes them for measurements of the competition data."""
    rng = np.random.default_rng(seed)
    ball = g["ball"] + rng.normal(0, sigma_ball, g["ball"].shape)
    players = g["players"] + rng.normal(0, sigma_player, g["players"].shape)

    def gaps(n, rate, mean_s):
        mask = np.zeros(n, dtype=bool)
        mean = mean_s * fps
        p_start = rate / (mean * (1 - rate))
        starts = np.flatnonzero(rng.random(n) < p_start)
        for s, length in zip(starts, rng.geometric(1.0 / mean, len(starts))):
            mask[s:s + length] = True
        return mask

    ball[gaps(len(ball), ball_gap_rate, ball_gap_mean_s)] = np.nan
    for p in range(players.shape[1]):
        players[gaps(len(ball), player_gap_rate, player_gap_mean_s), p] = np.nan
    out = dict(g)
    out["ball"], out["players"] = ball, players
    return out


def simulate_atd(g, seed, p_dead_hidden=0.6, live_gaps_per_half=20, sigma_player=0.3,
                 sigma_ball=0.3, player_gap_rate=0.07, fps=25):
    """Dress a sample game up as automatic tracking, including what the corrected data lacks:
    a ball that is sometimes visible while play is dead (parked at the restart spot) and long
    drop-outs during live play. Modelled on the SHAPE of the gaps in the organisers' published
    baseline_raw.csv (almost none of 1-3 s, most 3-60 s). Every rate here is an assumption."""
    rng = np.random.default_rng(seed)
    x = degrade(g, seed, ball_gap_rate=0.0, player_gap_rate=player_gap_rate,
                sigma_player=sigma_player, sigma_ball=sigma_ball)
    ball, dead, n = x["ball"], g["labels"] == "D", len(g["labels"])
    edges = np.diff(np.concatenate(([0], dead.astype(np.int8), [0])))
    for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if rng.random() < p_dead_hidden or b >= n:
            lo = a + int(rng.uniform(-1, 3) * fps); hi = b - int(rng.uniform(0, 1) * fps)
            ball[max(0, lo):max(a, hi)] = np.nan          # hidden from a little after it dies
            ball[a:max(a, lo)] = g["ball"][a - 1] if a > 0 else np.nan
        else:
            spot = g["ball"][b]                           # parked where it will be restarted
            ball[a:b] = spot + rng.normal(0, 0.1, (b - a, 2))
    live = np.flatnonzero(~dead)
    for period in (1, 2):
        cand = live[g["period"][live] == period]
        for s in rng.choice(cand, live_gaps_per_half, replace=False):
            ball[s:s + int(rng.uniform(3, 12) * fps)] = np.nan
    x["ball"] = ball
    return x
