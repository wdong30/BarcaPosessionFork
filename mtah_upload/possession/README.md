# Possession from automatic tracking: two decodes instead of a vote

Built 16 Nov 2026 for More than a Hack 2027, Phase 1. numpy only, deterministic, no model files.
Nothing here has been run on the competition match yet.

## Method

1. **Dead or live** (`dead.py`). A two-state Viterbi decode over per-frame evidence:
   the share of players running (fitted on Metrica's public sample game 1, where the corrected
   tracking marks dead ball exactly), plus the ball when it is seen (fast = live, still = dead,
   off the pitch = dead). A missing ball is not evidence of anything.
2. **Which team** (`model.py`). Inside each live stretch, a second two-state decode. Evidence is
   which team's nearest player is clearly closer to the ball, only when someone is within 2 m.
   A change of team costs 20 saturated frames (0.8 s of clear control), so one touch or a
   deflection does not flip possession, drop-outs carry the last team, and the change is placed
   where the new team's control starts.
3. Frames of a half before its Smart Tagging `Kick Off (Start)` tag are dead (`run.py`).

## Run on the competition data

Copy this folder into the hackathon repository as `possession/`, then from the repository root:

    python possession/run.py --out evaluation/results/<team_name>.csv
    python evaluation/validate_submission.py evaluation/results/<team_name>.csv
    python evaluation/run_evaluation.py

## Development data and results

Dev set: Metrica sample games 1 and 2 (github.com/metrica-sports/sample-data), labels built
mechanically in `devset.py` (D = no ball in the corrected tracking; team = most recent on-ball
event). Constants were chosen on game 1; game 2 is held out. `simulate_atd` degrades the clean
tracking (noise, vanishing players, long ball drop-outs, ball parked in view during some
stoppages); its rates are assumptions, not measurements of the competition data.

spellF1 on held-out game 2, simulated automatic tracking:

| share of stoppages with the ball hidden | organisers' baseline (+/-50) | this pipeline | perfect dead mask + team decode |
|---|---|---|---|
| 30% | 0.305 | 0.497 | 0.638 |
| 60% | 0.318 | 0.464 | 0.637 |
| 90% | 0.343 | 0.428 | 0.626 |

Reproduce the fitted constants: `python3 fit_dead.py <folder with the sample games>`.

## Known limits

- The dev labels are more fragmented than the competition's public ground truth (about 33
  possessions per 5 live minutes against about 21), so absolute numbers will not transfer.
- `P_FAST` / `P_SLOW` for a visible ball during dead play are assumed: the corrected sample
  tracking has no ball while play is dead, so there was nothing to fit them on.
- Dead-ball detection is the larger remaining loss (0.46 against 0.64 with a perfect mask).
- Possessions shorter than about a second are mostly missed by design.
- `run.py`'s parser is tested on a file written in the documented ATD layout, not on the real one.
