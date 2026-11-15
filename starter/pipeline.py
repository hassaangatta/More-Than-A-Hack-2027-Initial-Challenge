"""Put the steps together into H/A/D/X labels (used by make_submission.py).

Each half is cut at the dead-ball stretches. The live stretches in between go through the hold step
and the gap filler one by one, and the dead stretches are labelled D. Frames outside the two halves
are X, because they are not graded.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]
from games import GAME, half_video_blocks  # noqa: E402
from baseline import HOME, AWAY  # noqa: E402
from dead_ball import runs_of  # noqa: E402
from possession_hold import hold_smooth, HOLD  # noqa: E402
from gap_filler import fill_gaps, SPLIT  # noqa: E402


def label_half(codes, dead, hold=HOLD, split=SPLIT):
    """H/A/D letters for one half."""
    labels = np.full(len(codes), "D", dtype="<U1")
    for first, last in runs_of(~dead):
        span = fill_gaps(hold_smooth(codes[first:last + 1], hold), split)
        labels[first:last + 1] = np.where(span == HOME, "H", np.where(span == AWAY, "A", "X"))
    return labels


def make_labels(codes, dead, hold=HOLD, split=SPLIT, game=GAME):
    """Labels for every frame of the match."""
    labels = np.full(len(codes), "X", dtype="<U1")
    for _, first, last in half_video_blocks(game):
        labels[first:last + 1] = label_half(codes[first:last + 1], dead[first:last + 1], hold, split)
    return labels


def count_spells(labels, game=GAME):
    """Number of possessions in the graded halves (a new one starts at each change of team or after a D)."""
    total = 0
    for _, first, last in half_video_blocks(game):
        block = labels[first:last + 1]
        block = block[block != "X"]
        previous = np.concatenate(([""], block[:-1]))
        total += int(((block != "D") & (block != previous)).sum())
    return total
