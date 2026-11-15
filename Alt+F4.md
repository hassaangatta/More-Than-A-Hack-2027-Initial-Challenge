# Team Alt+F4 — technical document

**Task:** for every frame of the Barcelona–Athletic match, say who has the ball (`H` home, `A` away),
or that the ball is dead (`D`).

**Our answer in one line:** a short chain of simple, explainable rules applied to the tracking data.
No machine learning, no training data, no video. The same input always produces the same file.

---

## 1. The idea, in plain words

Football possession is a *long-lasting* thing: a team has the ball for ten or twenty seconds, not for
a fraction of a second. The scoring metric (`spellF1`) counts whole possessions, not single frames. So
the main job is not "who is nearest the ball right now" (that flickers constantly) but **"who really
has it, and for how long"**.

```
tracking file ─▶ 1. who is nearest the ball? ─▶ 2. is the ball dead? ─▶ 3. make possessions last ─▶ 4. fill gaps ─▶ CSV
```

| step | what it does | in one sentence |
|---|---|---|
| 1 | `starter/nearest_ball.py` | The team of the player closest to the ball is the candidate. |
| 2 | `starter/dead_ball.py` | If the ball is out, or everyone is standing still, label it `D`. |
| 3 | `starter/possession_hold.py` | A change of possession only counts if it lasts 3 seconds. |
| 4 | `starter/gap_filler.py` | Where the ball is not tracked, carry the nearby answer across. |

## 2. What we used (and did not)

| asset | used? |
|---|---|
| tactical tracking data (`atd.txt` + `atd.xml`) | **Yes — the only input** |
| tactical video, homography, Smart Tagging, broadcast files | No |

*Tried and dropped:*

- **A learned model** (gradient-boosted trees plus a smoothing step, trained on Metrica's three public
  sample games). It was better at who-has-the-ball (about 90% correct on live frames against 84%), but
  it missed stoppages on the real tracking data (found 37% of dead-ball frames against 64% for our
  rules). The reason is that in the sample games the ball is almost never tracked during a stoppage,
  so the model learned a cue that does not exist in the competition data. We dropped it for the
  simpler, deterministic approach. None of its code or data is used in the submission.
- **Smart Tagging** as a team signal: its team tag matched the truth on only about 70% of live frames
  in the public window, worse than nearest-player.
- **Video:** not used. The tracking already contains positions, and the machine we had has no GPU.

## 3. Training data

**None.** There is nothing learned, so there is no training set and no labels of any kind, by hand or
by algorithm. In particular, no hand labels exist for the provided game.

One disclosure: we looked at Metrica's public Sample Game 3 and at the public 5-minute window to
sanity-check how often our stoppage rule fires. We did not fit anything to them, but we did adjust a
cap on the rule after seeing that the first version flagged too much (see limitations).

## 4. The possession rules we implemented

This is where we drew the lines.

- **Who has the ball.** The team of the *closest tracked player* to the ball (distance measured in
  metres, not in 0–1 units, because the pitch is not square).
- **Loose balls, deflections and duels.** There is no "contested" state. The ball stays with the team
  that had it until the other team has been nearest for **3 seconds** (75 tracked frames). Anything
  shorter is treated as a touch or a deflection and ignored. We took "3 seconds" from the rules'
  "a player keeps the ball for a few seconds".
- **When a change happens.** Once a change is accepted, it is dated to the *first* frame the new team
  became nearest, not to the moment the 3 seconds ran out. Late changes are expensive on the metric.
- **Dead ball (`D`).** A stretch is dead if the ball is outside the lines, or if the players are
  barely moving (median under 3 m in 2 s) while the ball is slow (under 6 m in 2 s) or not tracked.
  It has to last at least 4 seconds, and we join stretches less than 2 seconds apart. A stoppage
  where the ball is not tracked is trusted for at most 30 seconds.
- **Restarts.** A stoppage ends a possession. `H, D, H` is two possessions, so each half is cut at
  every stoppage and the live stretches on either side are handled separately. After a restart, the
  first team to hold the ball for 3 seconds owns it. We do not guess who "owns" a dead ball (the task
  does not ask for it).
- **Ball not tracked (41% of frames).** A gap of up to 2 seconds is bridged by drawing a straight line
  between the last and next tracked ball position. For longer gaps: if the same team is nearest on
  both sides, we keep that team. If the two sides differ, we put the change at the **middle** of the
  gap.
- **Outside play.** Before kickoff, halftime and after the final whistle we write `X`; those frames are
  not graded.

**A tiny example** (one letter per second, ball tracked throughout, so "3 seconds" is 3 letters):

```
nearest player:   H H H H H A H H H A A A A A A
after step 3:     H H H H H H H H H A A A A A A
```

The single `A` lasts 1 second, so it is ignored. The final run of `A` lasts 6 seconds, so the change
counts, and it is dated to the first `A` of that run.

## 5. Results (public 5-minute window — a pipeline check, not a ranking)

Compared with the organisers' best baseline on the ranking metric (`baseline_smooth125`):

| | Alt+F4 | baseline_smooth125 |
|---|---|---|
| spellF1 (ranking metric) | **0.482** | 0.424 |
| macroF1 | **0.781** | 0.581 |
| possessions predicted (true: 17) | 17 | 16 |
| dead-ball frames found | **64%** | 8% |
| right team on live frames | 84% | **89%** |

The scorer itself says differences below about 0.34 are noise on a window with only 17 possessions,
so this shows the pipeline works and is in a sensible range. **It does not prove we beat the
baselines.** Over the whole match the file contains 208 possessions averaging 19 seconds (counted
with `X` treated as a gap).

## 6. Limitations (be honest)

- **The dead-ball thresholds were set by us after looking at the public window's statistics**, and
  the 30-second cap was added after the first version flagged 44% of the match as dead. So the
  public-window score is optimistic for that part. On the full match the rule flags 35% of graded
  frames, with a longest stoppage of 94 seconds, and its accuracy there is unknown.
- **The weakest part is the exact moment possession changes:** we find 7 of the 20 true changes within
  one second, and 18 of our 25 predicted changes are spurious (transition F1 0.31).
- **Long ball gaps are a guess.** Gaps of 15 seconds or more hold about 75% of all missing-ball time;
  the "middle of the gap" rule and "same team on both sides" are assumptions.
- **Nearest-player is not control.** A long pass in the air can briefly make the opponent "nearest".
  The 3-second rule absorbs most of this, but not all.

*What we would do with more time:* find out how the hidden dead-ball label is defined, learn the
hand-over point inside long gaps from other games' data, and use the video to find the ball where
the tracker lost it.

## 7. Reproduce it

```bash
python -m pip install -r requirements-final.txt
# put the data bundle in data/  (see DATA.md), or:  set MTAH_DATA_DIR=<folder that holds tactical/>
python make_submission.py                       # writes  submission/Alt+F4.csv   (about a minute)
python evaluation/validate_submission.py "submission/Alt+F4.csv"
python evaluation/run_evaluation.py --results submission
```

Only `data/tactical/BAR-ATH_tactical_atd.txt` and `.xml` are read. There is no randomness, so running
it twice gives byte-identical files (we checked: SHA-256 `14b561ce…516009`).
