"""Does the Smart Tagging evidence help? A/B on Metrica's public sample games.

    python3 dev_tags.py <folder with Sample_Game_1_* and Sample_Game_2_* files> [game] [seeds]

The sample games have no Smart Tagging, so one is simulated from their set-piece events and
written as SportsCode XML (so tagging.read() is exercised too). Everything about the simulation
is an assumption, chosen to look like DATA.md's description of the real file:
  - recall per type: throw-ins 0.8, corners 0.9, free kicks 0.3 (the real file has only 5),
    kick-offs 1.0; a few false set pieces at random live frames; 5% of team tags wrong;
  - every throw-in is tagged Team N/A;
  - the window is 4-7 s long and the restart falls anywhere inside it;
  - a GOALS tag ends 1 s after each goal.
No frame of the competition match is involved anywhere in this file.
"""
import os
import sys
import tempfile

import numpy as np
import pandas as pd

import devset
import metric
import pipeline
import tagging

FPS = 25
RECALL = {"throw-in": 0.8, "corner kick": 0.9, "free kick": 0.3,
          "kick off (start)": 1.0, "kick off (after goal)": 1.0}
TYPE_OF = {"THROW IN": "throw-in", "CORNER KICK": "corner kick", "FREE KICK": "free kick",
           "FREE KICK-RETAKEN": "free kick", "KICK OFF": "kick off", "KICK OFF-RETAKEN": "kick off"}
NAME = {"H": "Barcelona", "A": "Atletic Club", None: "N/A"}


def simulate_tags(events, labels, seed, n_false=4, p_wrong_team=0.05):
    rng = np.random.default_rng(seed)
    ev = events.sort_values("Start Frame")
    goals = ev[(ev["Type"] == "SHOT") & ev["Subtype"].str.contains("GOAL", na=False)]
    goal_frames = (goals["Start Frame"].to_numpy() - 1).tolist()
    out = []
    for g in goal_frames:
        out.append(("GOALS", None, None, (g - 8 * FPS) / FPS, (g + FPS) / FPS))
    for _, e in ev[ev["Type"] == "SET PIECE"].iterrows():
        kind = TYPE_OF.get(e["Subtype"])
        if kind is None:
            continue
        f = int(e["Start Frame"]) - 1
        if kind == "kick off":
            kind = "kick off (after goal)" if any(f - 180 * FPS < g < f for g in goal_frames) \
                else "kick off (start)"
        if rng.random() > RECALL[kind]:
            continue
        side = "H" if e["Team"] == "Home" else "A"
        if kind == "throw-in":
            side = None
        elif rng.random() < p_wrong_team:
            side = "A" if side == "H" else "H"
        length = rng.uniform(4, 7)
        start = f / FPS - rng.uniform(0.3, length - 0.3)
        out.append(("SET PIECES", kind, side, start, start + length))
    live = np.flatnonzero(labels != "D")
    for f in rng.choice(live, n_false, replace=False):
        kind = rng.choice(["free kick", "corner kick", "throw-in"])
        side = None if kind == "throw-in" else rng.choice(["H", "A"])
        out.append(("SET PIECES", kind, side, f / FPS, f / FPS + 5))
    return out


def write_xml(tags, path):
    with open(path, "w") as h:
        h.write("<file><ALL_INSTANCES>\n")
        for i, (code, kind, side, s, e) in enumerate(tags):
            h.write("<instance><ID>{}</ID><start>{:.2f}</start><end>{:.2f}</end><code>{}</code>"
                    "<label><group>Team</group><text>{}</text></label>".format(i, s, e, code, NAME[side]))
            if kind:
                h.write("<label><group>Type</group><text>{}</text></label>".format(kind.title()))
            h.write("</instance>\n")
        h.write("</ALL_INSTANCES></file>\n")


def runs_of(period):
    return [(int(np.flatnonzero(period == p)[0]), int(np.flatnonzero(period == p)[-1]) + 1)
            for p in (1, 2)]


def main(data_dir, game=2, seeds=(0, 1, 2), p_hidden=(0.3, 0.6, 0.9)):
    g = devset.load(game, data_dir)
    events = pd.read_csv(os.path.join(data_dir, "Sample_Game_{}_RawEventsData.csv".format(game)))
    runs = runs_of(g["period"])
    pitch = devset.PITCH
    print("game {}: spellF1 / macroF1, mean over seeds {}".format(game, list(seeds)))
    print("{:>10} {:>16} {:>16} {:>16}".format("hidden", "no tags", "tags", "tags + throw-in"))
    tmp = tempfile.mkdtemp()
    for ph in p_hidden:
        res = {k: [] for k in ("off", "on", "nothrow")}
        for seed in seeds:
            x = devset.simulate_atd(g, seed, p_dead_hidden=ph)
            path = os.path.join(tmp, "tags_{}.xml".format(seed))
            write_xml(simulate_tags(events, g["labels"], seed), path)
            tags = tagging.read(path, FPS)
            for key in res:
                saved = tagging.THROW_TEAM
                if key == "on":
                    tagging.THROW_TEAM = 0.3
                pred = pipeline.possession(x["ball"], x["players"], g["team"], pitch, runs,
                                           tags=None if key == "off" else tags)
                tagging.THROW_TEAM = saved
                res[key].append((metric.spell_f1(g["labels"], pred, runs)[0],
                                 metric.macro_f1(g["labels"], pred, runs)))
        cell = {k: "{:.3f} / {:.3f}".format(*np.mean(v, axis=0)) for k, v in res.items()}
        print("{:>10} {:>16} {:>16} {:>16}".format("{:.0%}".format(ph), cell["off"],
                                                   cell["nothrow"], cell["on"]))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 2,
         tuple(range(int(sys.argv[3]))) if len(sys.argv) > 3 else (0, 1, 2))
