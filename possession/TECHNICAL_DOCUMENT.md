# Technical document: possession as two decodes

More than a Hack 2027, Phase 1 (ball possession). Team: **[TEAM NAME]**

## Approach

The baseline decides each frame on its own (nearest player to the ball) and then smooths. We
treat possession as a segmentation problem instead, and solve it with two small Viterbi decodes.
There is no trained network and no model file: numpy only, deterministic, about 300 lines.

**1. Dead or live.** A two-state decode over per-frame evidence, scaled so that one second of
agreeing frames counts as one observation:

- the share of tracked players moving faster than 3 m/s, averaged over +/- 3 s. Its relation to
  dead ball is a two-parameter logistic fit on a public match (see Training data). This signal
  needs no ball, so it works through the long stretches where the tracking has none;
- the ball, when it is seen: faster than 4 m/s counts towards live, slower than 1 m/s towards
  dead, more than 1 m outside the lines towards dead;
- a missing ball counts as nothing. The baseline's rule "no ball = dead ball" is the thing we
  most wanted to remove, because the ball also goes missing for seconds at a time in live play.

**2. Which team.** Inside each live stretch, a second two-state decode. The evidence on a frame
is how much closer one team's nearest player is to the ball than the other's, capped at 1 m of
advantage, and only when some player is within 2 m of the ball. A change of team costs the
equivalent of 0.8 s of clear control. Each live stretch is decoded independently, so the team
that restarts is decided by what happens after the restart.

**Why this rather than something else.** The ranking metric scores whole possessions by temporal
overlap, so the costly errors are extra possessions and misplaced boundaries, not wrong frames.
A decode with a fixed price per change attacks exactly that: it will not open a new possession
unless the evidence pays for it, and because it is not causal it places the change where the new
team's control starts rather than after a confirmation delay.

## Possession rules implemented

- Possession stays with the last team through loose balls, balls in the air and tracking
  drop-outs: frames without evidence cannot change the state.
- A single touch, a deflection or a duel without follow-up does not change possession: it does
  not accumulate 0.8 s of control.
- Dead ball is a stoppage, never a possession, and no team is assigned to it.
- Frames of a half before its Smart Tagging "Kick Off (Start)" tag are dead.
- Possessions shorter than about a second are mostly not reported. This is deliberate.

## Assets used and dropped

| asset | use |
|---|---|
| tactical ATD (positions, team of each slot, ball) | the whole input |
| Smart Tagging | only the two "Kick Off (Start)" tags |
| ATD speed column | dropped; speeds are recomputed from positions over 0.48 s so that they are measured the same way as on the fitting data |
| rest of Smart Tagging | dropped: phases are not possessions, and throw-ins carry no team |
| tactical and broadcast video, broadcast ATD, homography | not used |

## Training data

No frame of the provided match was labelled, corrected or used for fitting.

- **Source:** Metrica Sports' public sample data, github.com/metrica-sports/sample-data, sample
  games 1 and 2 (corrected tracking at 25 fps plus event data, both produced by Metrica).
- **Labels:** derived mechanically by `devset.py`: dead where the corrected tracking has no ball;
  otherwise the team of the most recent on-ball event. Nobody on the team labelled anything.
- **Amount:** game 1, 145,006 frames and 385 possessions, used to fit the logistic relation
  between running share and dead ball, to measure how often a live ball is fast or slow, and to
  choose the reach and the two change costs. Game 2, 141,156 frames and 374 possessions, held out.
- **Simulation:** because the sample tracking is clean, `devset.simulate_atd` degrades it
  (position noise, players vanishing, long ball drop-outs, a ball left in view during some
  stoppages). Its rates are assumptions. Two constants (how often a visible ball is fast or slow
  while play is dead) are assumed too, since the corrected sample tracking has no ball then.
- **The public 5 minutes** of the provided match were used to check the pipeline and to
  understand how stoppages look in the automatic tracking. No constant was fitted to them.

On held-out game 2 under that simulation, spellF1 was 0.43 to 0.50 against 0.31 to 0.34 for the
organisers' baseline logic, depending on how many stoppages hide the ball.

## Limitations

- Dead-ball detection is the largest remaining loss: with a perfect dead-ball mask the same team
  decode reaches about 0.64 on the dev set.
- The simulation is a guess at how the automatic tracking behaves, and the dev labels are more
  fragmented than the competition's ground truth, so dev numbers are relative, not predictions.
- Short possessions are missed, and a boundary can sit up to about a second from the truth when
  the ball is unseen at the hand-over.
- With more time: restart detection from the Smart Tagging set pieces, the ball's height from
  the video, and a dead-ball model trained on more public matches.

## Reproduction

    pip install -r requirements.txt
    # unpack the match data into ./data as described in DATA.md
    python possession/run.py --out evaluation/results/[TEAM NAME].csv
    python evaluation/validate_submission.py evaluation/results/[TEAM NAME].csv

`python possession/fit_dead.py <folder with the sample games>` reprints the fitted constants.
The code was written with an AI assistant (Claude). No competition data was shared with it.
