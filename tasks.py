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

# Index 0 is always the description stored in the shards; order is fixed so eval rollouts stay comparable.
TASK_PHRASES = {
    "slide cabinet": [
        TASKS["slide cabinet"],
        "Open the sliding cabinet door",
        "Push the sliding cabinet door open",
        "Slide open the cabinet door",
        "Move the sliding door of the cabinet to open it",
        "Open the cabinet by sliding its door",
        "Slide the door of the cabinet open",
        "Open the cabinet with the sliding door",
    ],
    "kettle": [
        TASKS["kettle"],
        "Put the kettle on the top left burner",
        "Place the kettle on the top left burner",
        "Set the kettle on the upper left burner",
        "Move the kettle onto the upper left burner",
        "Lift the kettle onto the top left burner",
        "Take the kettle to the upper left burner",
        "Carry the kettle over to the top left burner",
    ],
    "light switch": [
        TASKS["light switch"],
        "Switch on the overhead light",
        "Turn the overhead light on",
        "Flip the overhead light switch on",
        "Activate the overhead light switch",
        "Move the overhead light switch to on",
        "Use the switch to turn on the overhead light",
        "Turn on the light overhead",
    ],
    "microwave": [
        TASKS["microwave"],
        "Open the microwave",
        "Pull the microwave door open",
        "Swing the microwave door open",
        "Open the door of the microwave",
        "Pull open the microwave door",
        "Open up the microwave",
        "Use the handle to open the microwave door",
    ],
    "bottom burner": [
        TASKS["bottom burner"],
        "Turn on the bottom left burner",
        "Twist the oven knob for the bottom left burner",
        "Rotate the knob for the lower left burner",
        "Turn the knob that controls the bottom left burner",
        "Switch on the lower left burner with its knob",
        "Twist the knob for the lower left burner",
        "Rotate the oven knob for the bottom left burner",
    ],
    "top burner": [
        TASKS["top burner"],
        "Turn on the top left burner",
        "Twist the oven knob for the top left burner",
        "Rotate the knob for the upper left burner",
        "Turn the knob that controls the top left burner",
        "Switch on the upper left burner with its knob",
        "Twist the knob for the upper left burner",
        "Rotate the oven knob for the top left burner",
    ],
    "hinge cabinet": [
        TASKS["hinge cabinet"],
        "Open the second cabinet from the left",
        "Swing open the cabinet second from the left",
        "Open the hinged cabinet second from the left",
        "Pull open the second cabinet from the left",
        "Open the door of the cabinet second from the left",
        "Swing the door of the second cabinet from the left open",
        "Pull the hinged door of the second cabinet from the left open",
    ],
}
_PHRASES_BY_DESCRIPTION = {TASKS[task]: phrases for task, phrases in TASK_PHRASES.items()}


def sample_phrases(task_descriptions):
    """Swap each stored task description for a random synonymous phrase."""
    return [str(np.random.choice(_PHRASES_BY_DESCRIPTION[d])) for d in task_descriptions]


def eval_phrase(task, rollout):
    """Phrase for an eval rollout, cycling in the fixed TASK_PHRASES order."""
    phrases = TASK_PHRASES[task]
    return phrases[rollout % len(phrases)]


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
