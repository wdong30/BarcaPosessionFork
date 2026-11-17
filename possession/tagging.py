"""Smart Tagging as evidence for both decodes. Never copied across as labels.

Three things the tagging says that the tracking struggles with:

  1. Every set piece is a restart, and every restart ends a dead-ball stretch. Dead-ball detection
     is the pipeline's largest loss, so each tagged set piece adds dead evidence up to the restart
     and live evidence just after it. The restart frame is found from the ball where possible
     (first fast ball inside the tag window), because the tag window is a few seconds wide and the
     metric scores boundaries.
  2. The team on a set-piece tag is the team that restarts, i.e. the team in possession right
     after it. Throw-ins are tagged N/A; for those the restarting team is taken as the opposite of
     the team that had the ball before it went out (a second pass, see pipeline.py).
  3. A goal is followed by a long stoppage until the 'Kick Off (After Goal)'.

Optionally (off by default, PHASE_TEAM = 0), the team on phase tags such as BUILD UP or
DEFENDING IN MIDDLE THIRD is a weak team prior over the phase. Its sign per code is an assumption;
check it against the public ground truth with check_tags.py before switching it on.

Weights were chosen on sample game 1 with dev_tags.py and checked on held-out game 2.
All weights are evidence, in the units the decodes already use, so the tracking can still overrule
a wrong tag. The tagging is automatic and has false positives and false negatives (DATA.md).
"""
import xml.etree.ElementTree as ET

import numpy as np

FPS = 25
RESTART_TYPES = {"throw-in", "corner kick", "goal kick", "free kick",
                 "kick off (after goal)", "kick off (start)"}

# -- dead / live, in the per-frame units of dead.dead_llr (positive = dead) ----------------------
SP_DEAD_LLR = 0.75       # tag start -> restart: the ball is waiting to be played (weak: see README)
SP_LIVE_LLR = 3.0        # restart -> restart + SP_LIVE_S: play is on
SP_LIVE_S = 3.0
SP_DEFAULT_SHARE = 0.5   # restart placed this far into the window when the ball does not show it
SP_SEARCH_PAD_S = 1.0    # also look this far past the tag end for the kick
SP_MIN_DEAD_S = 0.4      # a restart is never placed in the first 0.4 s of the window
BALL_FAST = 4.0          # m/s, same threshold as dead.py
GOAL_DEAD_LLR = 2.0      # from shortly after a GOALS tag to the next kick-off after a goal
GOAL_GAP_S = 2.0
GOAL_LOOKBACK_S = 180.0

# -- team, in the units of model.evidence (one saturated frame of clear control = 1) -------------
SP_TEAM = 1.0            # restart -> restart + SP_TEAM_S, for the team on the tag
SP_TEAM_S = 3.0
THROW_TEAM = 0.0         # throw-ins, team inferred from who put the ball out (hurt on dev: off)
THROW_LOOKBACK_S = 30.0
PHASE_TEAM = 0.0         # off by default -- see check_tags.py
ATTACK_CODES = {"build up", "progression", "ball in final third", "ball in the box",
                "creating chances", "long ball", "attacking transition", "players in the box"}
DEFEND_CODES = {"defending in defensive third", "defending in middle third",
                "defending in attacking third", "defensive transition"}


def side_of(team_name, home=("barcelona",), away=("atletic", "athletic")):
    """'H', 'A' or None for a Team tag value ('N/A' and anything unrecognised -> None)."""
    t = (team_name or "").strip().lower()
    if any(h in t for h in home):
        return "H"
    if any(a in t for a in away):
        return "A"
    return None


def read(path, fps=FPS):
    """[{code, type, side, lo, hi}] with lo/hi video frames (hi exclusive), sorted by lo."""
    out = []
    for inst in ET.parse(path).iter("instance"):
        tags = {l.findtext("group"): (l.findtext("text") or "") for l in inst.findall("label")}
        out.append(dict(
            code=(inst.findtext("code") or "").strip().lower(),
            type=tags.get("Type", "").strip().lower(),
            side=side_of(tags.get("Team")),
            lo=int(round(float(inst.findtext("start")) * fps)),
            hi=int(round(float(inst.findtext("end")) * fps))))
    return sorted(out, key=lambda t: t["lo"])


def restarts(tags):
    return [t for t in tags if t["code"] == "set pieces" and t["type"] in RESTART_TYPES]


def restart_frame(tag, ball_speed, fps=FPS):
    """Frame where the set piece is taken: first fast ball in the window, else a fixed share."""
    n = len(ball_speed)
    lo = min(n, tag["lo"] + int(SP_MIN_DEAD_S * fps))
    hi = min(n, max(lo, tag["hi"] + int(SP_SEARCH_PAD_S * fps)))
    with np.errstate(invalid="ignore"):
        fast = np.flatnonzero(ball_speed[lo:hi] > BALL_FAST)
    if len(fast):
        return lo + int(fast[0])
    return min(n - 1, tag["lo"] + int(SP_DEFAULT_SHARE * (tag["hi"] - tag["lo"])))


def dead_llr(n, tags, ball_speed, fps=FPS):
    """Extra per-frame dead evidence (positive = dead) and the restart frames found.
    Returns (llr (n,), [(restart_frame, tag), ...])."""
    llr = np.zeros(n)
    found = []
    for t in restarts(tags):
        if t["lo"] >= n:
            continue
        r = restart_frame(t, ball_speed, fps)
        llr[max(0, t["lo"]):r] += SP_DEAD_LLR
        llr[r:min(n, r + int(SP_LIVE_S * fps))] -= SP_LIVE_LLR
        found.append((r, t))
    goals = [t for t in tags if t["code"] == "goals"]
    for r, t in found:
        if t["type"] != "kick off (after goal)":
            continue
        before = [g for g in goals if r - GOAL_LOOKBACK_S * fps <= g["hi"] < r]
        if before:
            g0 = before[-1]["hi"] + int(GOAL_GAP_S * fps)
            if g0 < t["lo"]:
                llr[g0:t["lo"]] += GOAL_DEAD_LLR
    return llr, found


def team_prior(n, tags, found, labels=None, fps=FPS):
    """Extra per-frame team evidence (positive = home), added to model.evidence.

    `found` is dead_llr's list of restarts. `labels` is a first-pass labelling; when given,
    throw-ins (tagged N/A) are credited to the team that did NOT have the ball before it went out."""
    prior = np.zeros(n)
    sign = {"H": 1.0, "A": -1.0}
    w = int(SP_TEAM_S * fps)
    for r, t in found:
        if t["side"] in sign:
            prior[r:min(n, r + w)] += SP_TEAM * sign[t["side"]]
        elif t["type"] == "throw-in" and labels is not None and THROW_TEAM:
            before = labels[max(0, r - int(THROW_LOOKBACK_S * fps)):r]
            live = before[(before == "H") | (before == "A")]
            if len(live):
                other = "A" if live[-1] == "H" else "H"
                prior[r:min(n, r + w)] += THROW_TEAM * sign[other]
    if PHASE_TEAM:
        for t in tags:
            if t["side"] not in sign:
                continue
            s = sign[t["side"]] if t["code"] in ATTACK_CODES else \
                -sign[t["side"]] if t["code"] in DEFEND_CODES else 0.0
            prior[max(0, t["lo"]):min(n, t["hi"])] += PHASE_TEAM * s
    return prior
