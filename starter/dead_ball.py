"""Step 2: find the stretches where the ball is dead (out of play).

A frame looks dead if the ball is outside the pitch lines, or if the players are barely moving and
the ball is slow or not tracked. Movement is measured over a 2 s window. A stretch only counts as a
stoppage if it lasts at least 4 s, and a stretch that rests only on "ball not tracked" is trusted
for at most 30 s.

The thresholds were picked by looking at the tracking, not by a search. See Alt+F4.md (limitations).
"""

import warnings

import numpy as np

HALF_WINDOW = 25        # frames either side of the frame, so a 2 s window
SLOW_BALL_M = 6.0       # ball moved less than this in the window: slow
SLOW_PLAYERS_M = 3.0    # median player moved less than this in the window: slow
MIN_PLAYERS = 10        # tracked players needed to judge the team
MERGE_S = 2.0           # join stretches closer together than this
MAX_HIDDEN_S = 30.0     # longest stoppage trusted without seeing the ball
MIN_DEAD_S = 4.0        # drop stoppages shorter than this
FPS = 25


def _shifted_distance(positions, scale, half):
    """Metres moved between frame i - half and frame i + half (NaN if either is missing)."""
    n = len(positions)
    ahead = np.full(positions.shape, np.nan, dtype=np.float32)
    behind = np.full(positions.shape, np.nan, dtype=np.float32)
    ahead[:n - half] = positions[half:]
    behind[half:] = positions[:n - half]
    return np.linalg.norm((ahead - behind) * scale, axis=-1)


def runs_of(mask):
    """First and last index of every run of True, as a list of (first, last)."""
    padded = np.concatenate(([False], mask, [False])).astype(np.int8)
    edges = np.diff(padded)
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1))


def dead_evidence(players, ball, pitch):
    """Per frame: is the ball out, is the ball slow, are the players slow."""
    scale = np.array(pitch, dtype=np.float32)
    ball_missing = np.isnan(ball).any(axis=1)

    ball_out = ~ball_missing & ((ball < 0) | (ball > 1)).any(axis=1)

    ball_moved = _shifted_distance(ball, scale, HALF_WINDOW)
    ball_slow = ~ball_missing & (ball_moved < SLOW_BALL_M)

    moved = _shifted_distance(players, scale, HALF_WINDOW)
    counted = (~np.isnan(moved)).sum(axis=1)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        median = np.nanmedian(np.where(np.isnan(moved), np.nan, moved), axis=1)
    players_slow = (counted >= MIN_PLAYERS) & (median < SLOW_PLAYERS_M)
    return ball_out, ball_slow, players_slow


def dead_mask(players, ball, pitch, merge_s=MERGE_S, min_dead_s=MIN_DEAD_S,
              max_hidden_s=MAX_HIDDEN_S):
    """True for every frame judged to be dead ball."""
    ball_out, ball_slow, players_slow = dead_evidence(players, ball, pitch)
    seen = ball_out | (players_slow & ball_slow)
    loose = seen | (players_slow & np.isnan(ball).any(axis=1))

    # a long stretch with the ball hidden is not trusted, keep only the frames where it is seen
    candidate = loose.copy()
    for first, last in runs_of(loose):
        if last - first + 1 > max_hidden_s * FPS:
            candidate[first:last + 1] = seen[first:last + 1]

    # join stretches that are close together
    mask = candidate.copy()
    runs = runs_of(candidate)
    for (_, last), (first, _) in zip(runs, runs[1:]):
        if first - last - 1 <= merge_s * FPS:
            mask[last + 1:first] = True

    # drop stretches that are too short
    keep = np.zeros_like(mask)
    for first, last in runs_of(mask):
        if last - first + 1 >= min_dead_s * FPS:
            keep[first:last + 1] = True
    return keep
