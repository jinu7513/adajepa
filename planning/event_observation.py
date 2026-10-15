"""Pair rendered observations with the immediately preceding RGB frame."""

import numpy as np


def with_previous_visual(obs):
    """Return a new observation dict with the true t-1 frame at each time.

    The first frame is paired with itself, as in the Event training dataset.
    Apply any visual corruption *before* calling this function so both members
    of a pair come from the same rendered observation stream.
    """
    visual = obs["visual"]
    if not isinstance(visual, np.ndarray) or visual.ndim != 5 or visual.shape[1] == 0:
        raise ValueError("Event observations require visual [batch, time, height, width, channels]")
    previous = np.empty_like(visual)
    previous[:, :1] = visual[:, :1]
    previous[:, 1:] = visual[:, :-1]
    result = dict(obs)
    result["visual_previous"] = previous
    return result
