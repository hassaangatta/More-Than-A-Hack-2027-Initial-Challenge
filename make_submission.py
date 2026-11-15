"""Build the submission file from the tracking data.

    python make_submission.py                      # writes submission/Alt+F4.csv
    python make_submission.py --team "Other Name"  # writes submission/Other Name.csv
    python make_submission.py --out some/path.csv  # writes to an exact path

Steps (see the modules in starter/ and Alt+F4.md):
    1. nearest_ball      closest team to the ball
    2. dead_ball         stretches where the ball is dead -> D
    3. possession_hold   a change of possession must last 3 s
    4. gap_filler        fill the frames with no answer
There is no training and no randomness, so the same data gives the same file.
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(ROOT, "starter"), os.path.join(ROOT, "evaluation"), ROOT]

import possession_metrics as pm  # noqa: E402
from games import GAME, GAMES, require, source_frames  # noqa: E402
from baseline import team_of_each_track, write_submission  # noqa: E402
from nearest_ball import load_positions, two_pass  # noqa: E402
from dead_ball import dead_mask  # noqa: E402
from pipeline import make_labels, count_spells  # noqa: E402

TEAM_NAME = "Alt+F4"
OUT_DIR = os.path.join(ROOT, "submission")


def build_labels(game=GAME, data_dir=None):
    """Labels for every frame of the match."""
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir)
    n_frames = source_frames("tactical", game)
    pitch = GAMES[game]["pitch"]
    players, ball = load_positions(raw_path, n_frames)
    codes, _, _ = two_pass(players, ball, team_of_each_track(meta_path), pitch)
    return make_labels(codes, dead_mask(players, ball, pitch), game=game)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--team", default=TEAM_NAME, help="team name, used as the file name")
    parser.add_argument("--out", default=None, help="exact output path (overrides --team)")
    args = parser.parse_args()

    out = args.out or os.path.join(OUT_DIR, args.team + ".csv")
    labels = build_labels()
    path = write_submission(out, labels)

    # same check the official scorer runs
    problem = pm.validate_submission(pm.read_label_csv(path), source_frames("tactical", GAME))
    if problem:
        sys.exit("INVALID: {}".format(problem))

    print("wrote {}".format(os.path.relpath(path, ROOT)))
    print("  " + "  ".join("{} {}".format(c, int((labels == c).sum())) for c in "HADX"))
    print("  {} possessions".format(count_spells(labels)))
    print("score it:  python evaluation/run_evaluation.py --results "
          + os.path.relpath(os.path.dirname(path) or ".", ROOT))
