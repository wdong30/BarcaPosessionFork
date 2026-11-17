"""How well does the real Smart Tagging agree with the public ground truth? Run from the repo root:

    python possession/check_tags.py

Reads only the Smart Tagging XML and evaluation/gt/*_possession_gt_public.csv (5 minutes, frames
11185-18684), so no tracking is needed. Prints:
  1. every tag overlapping the public window, with the ground-truth label mix inside it;
  2. for set pieces: where the ground truth's dead -> live change falls relative to the tag window
     (tagging.py assumes it is inside, or up to 1 s after);
  3. per phase code: how often the tagged team is the team in possession (ATTACK_CODES should be
     well above 50%, DEFEND_CODES well below). Only switch --phase-weight on in run.py if they are.
5 minutes is a small sample: read these as a sanity check of the assumptions, not as a fit.
"""
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import tagging  # noqa: E402


def main(tag_path=None, gt_path=None, fps=25):
    if tag_path is None:
        import games
        tag_path = games.asset_paths()["smart_tagging"]
    if gt_path is None:
        root = os.path.dirname(HERE)
        gt_path = glob.glob(os.path.join(root, "evaluation", "gt", "*_possession_gt_public.csv"))[0]
    gt = np.loadtxt(gt_path, delimiter=",", skiprows=1, dtype=str)[:, 1]
    graded = np.flatnonzero(gt != "X")
    w0, w1 = int(graded[0]), int(graded[-1]) + 1
    tags = [t for t in tagging.read(tag_path, fps) if t["hi"] > w0 and t["lo"] < w1]
    print("public window {}-{}: {} tags overlap it\n".format(w0, w1 - 1, len(tags)))

    print("{:<30} {:<22} {:>4} {:>7} {:>7}   H    A    D".format("code", "type", "side", "lo", "hi"))
    for t in tags:
        seg = gt[max(w0, t["lo"]):min(w1, t["hi"])]
        mix = [np.mean(seg == k) if len(seg) else 0 for k in "HAD"]
        print("{:<30} {:<22} {:>4} {:>7} {:>7}  {:.2f} {:.2f} {:.2f}".format(
            t["code"][:30], t["type"][:22], t["side"] or "-", t["lo"], t["hi"], *mix))

    print("\nset pieces: ground-truth restart (first live frame after a D) vs the tag window")
    for t in tagging.restarts(tags):
        lo, hi = max(w0, t["lo"] - 10 * fps), min(w1, t["hi"] + 5 * fps)
        seg = gt[lo:hi]
        ch = [lo + i + 1 for i in range(len(seg) - 1) if seg[i] == "D" and seg[i + 1] in "HA"]
        if not ch:
            print("  {} at {}-{}: no dead->live change within -10 s/+5 s".format(t["type"], t["lo"], t["hi"]))
            continue
        r = min(ch, key=lambda f: abs(f - (t["lo"] + t["hi"]) / 2))
        print("  {:<22} tag {}-{}  restart {}  = {:+.1f} s from tag start, {:+.1f} s from tag end,"
              " restarting team {} (tag says {})".format(t["type"], t["lo"], t["hi"], r,
                                                         (r - t["lo"]) / fps, (r - t["hi"]) / fps,
                                                         gt[r], t["side"] or "N/A"))

    print("\nphase codes: share of live frames where the tagged team has the ball")
    stats = {}
    for t in tags:
        if t["side"] not in ("H", "A") or t["code"] == "set pieces":
            continue
        seg = gt[max(w0, t["lo"]):min(w1, t["hi"])]
        live = seg[(seg == "H") | (seg == "A")]
        s = stats.setdefault(t["code"], [0, 0, 0])
        s[0] += int(np.sum(live == t["side"])); s[1] += len(live); s[2] += 1
    for code, (same, n, k) in sorted(stats.items()):
        kind = "attack" if code in tagging.ATTACK_CODES else "defend" if code in tagging.DEFEND_CODES else "?"
        print("  {:<32} {:<7} {:>3} tags  {:>6} frames  tagged team has it {:.0%}".format(
            code, kind, k, n, same / n if n else float("nan")))


if __name__ == "__main__":
    main(*sys.argv[1:3])
