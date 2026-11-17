"""Tracking in, one label per frame out. Everything else in this folder feeds this function."""
import numpy as np
import dead as dead_ball
import model
import tagging


def possession(ball, players, team, pitch, runs, kickoffs=None, tags=None):
    """ball (N,2) and players (N,P,2) in metres with NaN where unseen; team (P,) 0 home / 1 away;
    runs = graded [lo, hi) stretches; kickoffs = optional {lo: first live frame of that stretch};
    tags = optional tagging.read() output, used as extra evidence in both decodes.
    Returns (N,) array of 'H' 'A' 'D' 'X' ('X' only outside the graded stretches)."""
    n = len(ball)
    extra, found = None, []
    if tags:
        extra, found = tagging.dead_llr(n, tags, dead_ball.ball_speed(ball))
    dead = dead_ball.dead_mask(ball, players, pitch, runs, extra_llr=extra)
    for lo, hi in runs:
        if kickoffs and lo in kickoffs:
            dead[lo:min(hi, max(lo, kickoffs[lo]))] = True      # nothing is live before kick-off
    if not tags:
        return model.label(ball, players, team, dead, runs)
    # pass 1: tagged teams only; pass 2: add throw-ins, credited against whoever had it before
    first = model.label(ball, players, team, dead, runs, prior=tagging.team_prior(n, tags, found))
    return model.label(ball, players, team, dead, runs,
                       prior=tagging.team_prior(n, tags, found, labels=first))
