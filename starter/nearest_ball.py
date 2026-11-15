"""Step 1: which team is closest to the ball.

Pass 1 uses the frames where the ball is tracked. Pass 2 covers short gaps (up to 2 s): the ball is
drawn as a straight line between its last and next tracked position, then the same rule is used.
Frames that are still unanswered stay NOBODY.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]
from baseline import NOBODY  # noqa: E402

MAX_INTERP_GAP = 50   # frames, 2 s at 25 fps
CHUNK = 20000         # frames handled at a time

FIELD_SEP, GROUP_SEP = ":", ";"


def load_positions(raw_path, n_frames):
    """Read the ATD text file into arrays (0-1 coordinates, NaN where missing).

    Returns players (n_frames, n_slots, 2) and ball (n_frames, 2). Row i is ATD frame i + 1.
    """
    players = ball = None
    with open(raw_path) as handle:
        for line in handle:
            parts = line.rstrip("\n").split(FIELD_SEP)
            if len(parts) != 3:
                continue
            row = int(parts[0]) - 1
            if not 0 <= row < n_frames:
                continue
            groups = parts[1].split(GROUP_SEP)
            if players is None:
                players = np.full((n_frames, len(groups), 2), np.nan, dtype=np.float32)
                ball = np.full((n_frames, 2), np.nan, dtype=np.float32)
            for slot, group in enumerate(groups):
                fields = group.split(",")
                if len(fields) >= 2:
                    players[row, slot] = (float(fields[0]), float(fields[1]))
            fields = parts[2].split(",")
            if len(fields) >= 2:
                ball[row] = (float(fields[0]), float(fields[1]))
    if players is None:
        raise ValueError("{}: no tracking lines".format(raw_path))
    return players, ball


def nearest_team(players, ball, team_of_slot, pitch):
    """Team of the player closest to the ball for each frame (NOBODY if there is no ball)."""
    length, width = pitch
    scale = np.array([length, width], dtype=np.float32)   # work in metres, the pitch is not square
    out = np.full(len(ball), NOBODY, dtype=np.int8)
    for start in range(0, len(ball), CHUNK):
        stop = min(start + CHUNK, len(ball))
        offset = (players[start:stop] - ball[start:stop, None, :]) * scale
        distance = np.einsum("fsk,fsk->fs", offset, offset)
        usable = ~np.isnan(distance).all(axis=1) & ~np.isnan(ball[start:stop]).any(axis=1)
        nearest = np.full(stop - start, NOBODY, dtype=np.int8)
        if usable.any():
            closest = np.nanargmin(distance[usable], axis=1)
            nearest[usable] = team_of_slot[closest]
        out[start:stop] = nearest
    return out


def gap_runs(missing):
    """First and last index of every run of True, as a list of (first, last)."""
    padded = np.concatenate(([False], missing, [False])).astype(np.int8)
    edges = np.diff(padded)
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1))


def interpolate_ball(ball, max_gap=MAX_INTERP_GAP):
    """Fill short ball gaps with a straight line. Returns the new ball array and a mask of filled frames."""
    estimate = ball.copy()
    filled = np.zeros(len(ball), dtype=bool)
    for first, last in gap_runs(np.isnan(ball).any(axis=1)):
        if first == 0 or last == len(ball) - 1 or last - first + 1 > max_gap:
            continue
        before, after = ball[first - 1], ball[last + 1]
        weight = (np.arange(first, last + 1) - (first - 1)) / float(last + 1 - (first - 1))
        estimate[first:last + 1] = before + (after - before) * weight[:, None]
        filled[first:last + 1] = True
    return estimate, filled


def two_pass(players, ball, team_of_slot, pitch, max_gap=MAX_INTERP_GAP):
    """Run both passes. Returns the team codes, the pass-1-only codes and the mask pass 2 filled."""
    pass1 = nearest_team(players, ball, team_of_slot, pitch)

    estimate, filled = interpolate_ball(ball, max_gap)
    pass2 = nearest_team(players, estimate, team_of_slot, pitch)

    codes = pass1.copy()
    codes[filled] = pass2[filled]
    return codes, pass1, filled
