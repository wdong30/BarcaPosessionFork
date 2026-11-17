"""Tracking in, one label per frame out. Everything else in this folder feeds this function."""
import numpy as np
import dead as dead_ball
import model


def possession(ball, players, team, pitch, runs, kickoffs=None):
    """ball (N,2) and players (N,P,2) in metres with NaN where unseen; team (P,) 0 home / 1 away;
    runs = graded [lo, hi) stretches; kickoffs = optional {lo: first live frame of that stretch}.
    Returns (N,) array of 'H' 'A' 'D' 'X' ('X' only outside the graded stretches)."""
    dead = dead_ball.dead_mask(ball, players, pitch, runs)
    for lo, hi in runs:
        if kickoffs and lo in kickoffs:
            dead[lo:min(hi, max(lo, kickoffs[lo]))] = True      # nothing is live before kick-off
    return model.label(ball, players, team, dead, runs)
