"""Competition tracking -> submission CSV. Run from the root of the hackathon repository:

    python possession/run.py --out evaluation/results/<team_name>.csv
    python evaluation/validate_submission.py evaluation/results/<team_name>.csv
    python evaluation/run_evaluation.py

Reads the tactical ATD (metadata XML + raw text) and, if present, the Smart Tagging XML. The
'Kick Off (Start)' tags say where each half's play begins; the other set pieces and the goals are
extra dead-ball and restart-team evidence (tagging.py). --no-tag-evidence turns the latter off,
for an A/B against the public ground truth.
Deterministic: no randomness, no model files, numpy only.
The ATD parsing follows starter/baseline.py from the organisers' repository (MIT licence).
"""
import argparse
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from pipeline import possession  # noqa: E402
import tagging  # noqa: E402


def read_teams(meta_path):
    root = ET.parse(meta_path).getroot()
    team_ids = [t.get("id") for t in root.iter("Team")]
    if len(team_ids) != 2:
        raise ValueError("{}: expected 2 teams, found {}".format(meta_path, len(team_ids)))
    slots = [p.get("teamId") for p in root.iter("Player")]
    return np.array([0 if t == team_ids[0] else 1 for t in slots], dtype=np.int8)


def read_atd(raw_path, n_frames, n_slots, pitch):
    """ball (n_frames, 2) and players (n_frames, n_slots, 2) in metres, NaN where unseen."""
    ball = np.full((n_frames, 2), np.nan)
    players = np.full((n_frames, n_slots, 2), np.nan)
    scale = np.array(pitch, dtype=float)
    with open(raw_path) as handle:
        for line in handle:
            parts = line.strip().split(":")
            if len(parts) != 3:
                continue
            i = int(parts[0]) - 1                     # frame_id counts from 1, submissions from 0
            if not 0 <= i < n_frames:
                continue
            groups = parts[1].split(";")
            for slot in range(min(n_slots, len(groups))):
                f = groups[slot].split(",")
                if len(f) >= 2 and f[0][:1] not in ("N", "n", ""):
                    players[i, slot] = (float(f[0]), float(f[1]))
            f = parts[2].split(",")
            if len(f) >= 2 and f[0][:1] not in ("N", "n", ""):
                ball[i] = (float(f[0]), float(f[1]))
    return ball * scale, players * scale


def read_kickoffs(tagging_path, runs, fps):
    """{first frame of a graded stretch: frame its 'Kick Off (Start)' tag begins}."""
    starts = []
    for inst in ET.parse(tagging_path).iter("instance"):
        tags = {l.findtext("group"): (l.findtext("text") or "") for l in inst.findall("label")}
        if (inst.findtext("code") or "").strip().lower() == "set pieces" and \
                tags.get("Type", "").strip().lower() == "kick off (start)":
            starts.append(int(round(float(inst.findtext("start")) * fps)))
    out = {}
    for lo, hi in runs:
        inside = [s for s in starts if lo - 15 * fps <= s < hi]
        if inside:
            out[lo] = max(lo, min(inside))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--meta"); ap.add_argument("--raw"); ap.add_argument("--tagging")
    ap.add_argument("--frames", type=int); ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--pitch", type=float, nargs=2)
    ap.add_argument("--runs", type=int, nargs="+", help="lo hi lo hi ... (half-open)")
    ap.add_argument("--no-tag-evidence", action="store_true",
                    help="use the tagging for the kick-offs only, as before")
    ap.add_argument("--phase-weight", type=float, default=tagging.PHASE_TEAM,
                    help="team prior from phase tags (default off; see check_tags.py)")
    ap.add_argument("--sp-dead", type=float, default=tagging.SP_DEAD_LLR,
                    help="dead evidence before each tagged restart (default %(default)s)")
    a = ap.parse_args()

    if not (a.meta and a.raw and a.frames and a.pitch and a.runs):
        import games                                   # the organisers' constants and paths
        paths = games.asset_paths()
        a.meta = a.meta or paths["atd_meta"]; a.raw = a.raw or paths["atd_raw"]
        if a.tagging is None and os.path.exists(paths.get("smart_tagging", "")):
            a.tagging = paths["smart_tagging"]
        a.frames = a.frames or games.source_frames("tactical")
        a.pitch = a.pitch or games.GAMES[games.GAME]["pitch"]
        a.runs = a.runs or [v for _, lo, hi in games.half_video_blocks(games.GAME) for v in (lo, hi + 1)]
    runs = [(a.runs[i], a.runs[i + 1]) for i in range(0, len(a.runs), 2)]

    team = read_teams(a.meta)
    print("slots: {} home, {} away".format(int((team == 0).sum()), int((team == 1).sum())))
    ball, players = read_atd(a.raw, a.frames, len(team), a.pitch)
    print("ball seen on {:.1%} of frames".format(np.mean(~np.isnan(ball[:, 0]))))
    kickoffs = read_kickoffs(a.tagging, runs, a.fps) if a.tagging else None
    print("graded stretches {}  kick-offs {}".format(runs, kickoffs))
    tags = None
    if a.tagging and not a.no_tag_evidence:
        tagging.PHASE_TEAM, tagging.SP_DEAD_LLR = a.phase_weight, a.sp_dead
        tags = tagging.read(a.tagging, a.fps)
        print("tag evidence: {} restarts, {} goals, phase weight {}, sp-dead {}".format(
            len(tagging.restarts(tags)), sum(t["code"] == "goals" for t in tags), a.phase_weight,
            a.sp_dead))

    labels = possession(ball, players, team, tuple(a.pitch), runs, kickoffs, tags)
    folder = os.path.dirname(a.out)
    if folder and not os.path.isdir(folder):
        os.makedirs(folder)
    with open(a.out, "w") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, l) for i, l in enumerate(labels))
    for lo, hi in runs:
        seg = labels[lo:hi]
        n = int(np.sum((seg[1:] != seg[:-1]) & (seg[1:] != "D"))) + int(seg[0] != "D")
        print("  {}-{}: H {:.1%}  A {:.1%}  D {:.1%}  possessions {}".format(
            lo, hi, np.mean(seg == "H"), np.mean(seg == "A"), np.mean(seg == "D"), n))
    print("wrote {}".format(a.out))
