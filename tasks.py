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
# Training-only, in styles TASK_PHRASES lacks (requests, goal states, fragments, casual, long-winded).
# None use the held-out templates ("Get X ...", "X should end up ...", "Could you ...?", "X: verb").
STYLE_PHRASES = {
    "slide cabinet": [
        "Please slide the cabinet door open",
        "Can you slide the cabinet open?",
        "I want the sliding cabinet door open",
        "Leave the sliding door of the cabinet open",
        "slide cabinet open",
        "open sliding door",
        "Shove the sliding cabinet door over",
        "Reach over to the cabinet with the sliding door and slide it open",
    ],
    "kettle": [
        "Please put the kettle on the top left burner",
        "Can you move the kettle to the upper left burner?",
        "I want the kettle sitting on the top left burner",
        "Leave the kettle on the upper left burner",
        "kettle to top left burner",
        "kettle on upper left burner",
        "Plop the kettle on the top left burner",
        "Pick up the kettle and set it down on the burner at the top left",
    ],
    "light switch": [
        "Please turn on the overhead light",
        "Can you switch the overhead light on?",
        "I want the overhead light on",
        "Leave the overhead light switched on",
        "overhead light on",
        "light switch on",
        "Hit the overhead light",
        "Reach up to the overhead light switch and flip it to on",
    ],
    "microwave": [
        "Please open the microwave door",
        "Can you open the microwave?",
        "I want the microwave door open",
        "Leave the microwave open",
        "microwave open",
        "open microwave door",
        "Pop the microwave open",
        "Grab the handle of the microwave and pull its door open",
    ],
    "bottom burner": [
        "Please turn on the bottom left burner",
        "Can you turn the knob for the lower left burner?",
        "I want the bottom left burner on",
        "Leave the lower left stove burner on",
        "bottom left burner on",
        "turn lower left knob",
        "Crank the bottom left burner knob",
        "Reach for the knob that runs the bottom left burner and turn it",
    ],
    "top burner": [
        "Please turn on the top left burner",
        "Can you turn the knob for the upper left burner?",
        "I want the top left burner on",
        "Leave the upper left stove burner on",
        "top left burner on",
        "turn upper left knob",
        "Crank the top left burner knob",
        "Reach for the knob that runs the top left burner and turn it",
    ],
    "hinge cabinet": [
        "Please open the cabinet second from the left",
        "Can you open the second cabinet from the left?",
        "I want the second cabinet from the left open",
        "Leave the hinged cabinet second from the left open",
        "second cabinet from left open",
        "open hinged cabinet",
        "Yank open the cabinet second from the left",
        "Take hold of the second cabinet from the left and swing its door open",
    ],
}
_PHRASES_BY_DESCRIPTION = {TASKS[task]: TASK_PHRASES[task] + STYLE_PHRASES[task] for task in TASKS}

# Never trained on: eval checks the policy follows wording it hasn't seen.
HELDOUT_PHRASES = {
    "microwave": ["Get the microwave open", "The microwave door should end up open",
                  "Could you open the microwave for me?", "microwave door: open"],
    "hinge cabinet": ["Get the cabinet that is second from the left open",
                      "The cabinet second from the left should end up open",
                      "Could you open the cabinet next to the leftmost one?",
                      "cabinet, second from left: open"],
    "top burner": ["Get the top left burner going", "The knob for the top left burner should end up turned",
                   "Could you turn on the upper left stove burner?", "top left burner knob: turn"],
}


def sample_phrases(task_descriptions):
    """Swap each stored task description for a random synonymous phrase."""
    return [str(np.random.choice(_PHRASES_BY_DESCRIPTION[d])) for d in task_descriptions]


def task_from_argv(argv):
    """Task name from the command line, or exit listing the valid ones."""
    name = argv[1].replace("_", " ") if len(argv) == 2 else ""
    if name not in TASKS:
        sys.exit(f"usage: {argv[0]} <task>\n\ntasks:\n  " + "\n  ".join(TASKS))
    return name
