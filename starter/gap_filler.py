"""Step 4: fill the frames that still have no answer.

A gap with the same team on both sides takes that team. A gap between two different teams is split:
the first part stays with the earlier team and the rest goes to the later one (the middle, by default).
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]
from baseline import NOBODY  # noqa: E402
from nearest_ball import gap_runs  # noqa: E402

SPLIT = 0.5  # where in a hand-over gap the change is placed


def fill_gaps(codes, split=SPLIT):
    """Return a copy of the team codes with every NOBODY gap filled."""
    out = codes.copy()
    n = len(codes)
    for first, last in gap_runs(codes == NOBODY):
        left = codes[first - 1] if first > 0 else NOBODY
        right = codes[last + 1] if last < n - 1 else NOBODY
        if left == NOBODY and right == NOBODY:
            continue
        if left == NOBODY:
            out[first:last + 1] = right
        elif right == NOBODY or left == right:
            out[first:last + 1] = left
        else:
            cut = first + int(round(split * (last - first + 1)))
            out[first:cut] = left
            out[cut:last + 1] = right
    return out
