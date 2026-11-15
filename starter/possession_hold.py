"""Step 3: a change of possession only counts if it lasts.

The closest player flips between the two teams all the time in a duel, so the raw labels break one
possession into many pieces. Here a team only takes the ball once it has been the closest team for
HOLD frames (3 s). Shorter runs are ignored, and a change that does count is dated to the first
frame of the new run.

Example with HOLD = 3 and one letter per frame:

    closest   H H H H H A H H H A A A A A A
    result    H H H H H H H H H A A A A A A
"""

import numpy as np

HOME, AWAY, NOBODY = 0, 1, -1

HOLD = 75  # frames, 3 s at 25 fps


def hold_smooth(codes, hold=HOLD):
    """Remove short interruptions from the team codes. Frames with no answer stay NOBODY."""
    out = codes.copy()
    observed = np.flatnonzero(codes != NOBODY)
    if len(observed) == 0:
        return out

    # split the answered frames into runs of the same team
    teams = codes[observed]
    starts = np.concatenate(([0], np.flatnonzero(teams[1:] != teams[:-1]) + 1))
    lengths = np.diff(np.concatenate((starts, [len(teams)])))
    run_team = teams[starts]

    long_runs = np.flatnonzero(lengths >= hold)
    held = np.empty(len(starts), dtype=np.int8)
    if len(long_runs) == 0:
        # no run is long enough: use the longest one for the whole stretch
        held[:] = run_team[np.argmax(lengths)]
    else:
        current = run_team[long_runs[0]]
        for r in range(len(starts)):
            if r >= long_runs[0] and lengths[r] >= hold:
                current = run_team[r]
            held[r] = current
    out[observed] = np.repeat(held, lengths)
    return out
