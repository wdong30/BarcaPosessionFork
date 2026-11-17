"""Where the constants in dead.py come from. Uses Metrica's public sample games only.
    python3 fit_dead.py ../data
"""
import sys
import numpy as np
import devset, dead

d = devset.load(1, sys.argv[1])
y = (d["labels"] == "D").astype(float)
x = dead.running_share(d["players"])
ok = ~np.isnan(x); x, y = x[ok], y[ok]
b0, b1 = 0.0, 0.0
for _ in range(50):                                   # Newton steps, 1-D logistic regression
    p = 1 / (1 + np.exp(-(b0 + b1 * x))); w = p * (1 - p)
    g = np.array([np.sum(y - p), np.sum((y - p) * x)])
    h = np.array([[np.sum(w), np.sum(w * x)], [np.sum(w * x), np.sum(w * x * x)]])
    b0, b1 = np.array([b0, b1]) + np.linalg.solve(h, g)
print("logit P(dead) = {:.2f} + {:.2f} * running_share   prior dead {:.2f}".format(b0, b1, y.mean()))
v = dead.ball_speed(d["ball"]); v = v[~np.isnan(v)]
print("live ball: P(speed > {}) = {:.2f}   P(speed < {}) = {:.2f}".format(
    dead.BALL_FAST, np.mean(v > dead.BALL_FAST), dead.BALL_SLOW, np.mean(v < dead.BALL_SLOW)))
