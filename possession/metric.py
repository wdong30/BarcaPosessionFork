"""spellF1 and macroF1, re-implemented from evaluation/possession_metrics.py for use on the dev
set. Checked against the organisers' published public-window numbers (0.0378/0.2868/0.3256/0.4242).
The official scorer remains the authority for anything you submit."""
import numpy as np

IOUS = (0.3, 0.4, 0.5, 0.6, 0.7)


def segments(a):
    b = np.flatnonzero(a[1:] != a[:-1]) + 1
    bd = np.concatenate(([0], b, [len(a)]))
    return [(int(bd[i]), int(bd[i + 1]), a[bd[i]]) for i in range(len(bd) - 1)]


def spells(labels, runs):
    out = []
    for lo, hi in runs:
        out += [(lo + a, lo + b, k) for a, b, k in segments(labels[lo:hi]) if k in ("H", "A")]
    return out


def spell_f1(gt, pred, runs):
    gs, ps = spells(gt, runs), spells(pred, runs)
    ov, j0 = [], 0
    for i, (a, b, k) in enumerate(gs):
        while j0 < len(ps) and ps[j0][1] <= a:
            j0 += 1
        j = j0
        while j < len(ps) and ps[j][0] < b:
            c, d, m = ps[j]
            if m == k and d > a:
                ov.append(((min(b, d) - max(a, c)) / float(max(b, d) - min(a, c)), i, j))
            j += 1
    ov.sort(key=lambda t: (-t[0], t[1], t[2]))
    f1s = []
    for th in IOUS:
        mg, mp = set(), set()
        for iou, i, j in ov:
            if iou < th:
                break
            if i in mg or j in mp:
                continue
            mg.add(i); mp.add(j)
        tp = len(mg)
        pr = tp / len(ps) if ps else 0.0
        rc = tp / len(gs) if gs else 0.0
        f1s.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    return float(np.mean(f1s)), len(gs), len(ps)


def macro_f1(gt, pred, runs):
    m = np.zeros(len(gt), dtype=bool)
    for lo, hi in runs:
        m[lo:hi] = True
    g, p, out = gt[m], pred[m], []
    for k in "HAD":
        tp = np.sum((g == k) & (p == k)); fp = np.sum((g != k) & (p == k)); fn = np.sum((g == k) & (p != k))
        out.append(2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0)
    return float(np.mean(out))
