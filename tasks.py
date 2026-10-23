import sys

import numpy as np

# The only seven tasks the env accepts; anything else raises at gym.make.
# travel is how far past the success threshold the object has to move, which
# roughly ranks how long a demo takes.
TASKS = {
    "slide cabinet": "Slide the cabinet door open",              # travel 0.07
    "kettle":        "Move the kettle to the top left burner",   # travel 0.11
    "light switch":  "Turn on the overhead light switch",        # travel 0.39
    "microwave":     "Open the microwave door",                  # travel 0.45
    "bottom burner": "Turn the oven knob for the bottom left burner",  # 0.58
    "top burner":    "Turn the oven knob for the top left burner",     # 0.62
    "hinge cabinet": "Open the cabinet second from the left",    # travel 1.15
}

# Dict order is what task_index()'s integer ids mean, so reordering TASKS
# silently invalidates every checkpoint trained before the change.
TASK_DESCRIPTIONS = list(TASKS.values())
_TASK_INDEX = {description: i for i, description in enumerate(TASK_DESCRIPTIONS)}


def task_index(task_descriptions):
    """task_description string, or array of them, -> int64 array of class ids."""
    task_descriptions = np.atleast_1d(task_descriptions)
    return np.array([_TASK_INDEX[d] for d in task_descriptions], dtype=np.int64)


def task_from_argv(argv):
    """Task name from the command line, or exit listing the valid ones."""
    name = argv[1].replace("_", " ") if len(argv) == 2 else ""
    if name not in TASKS:
        sys.exit(f"usage: {argv[0]} <task>\n\ntasks:\n  " + "\n  ".join(TASKS))
    return name
