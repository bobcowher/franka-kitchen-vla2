"""Experiment: deployment success when the joint readings carry a constant offset, as after an encoder swap or re-zero.

Env: CHECKPOINT_URL (a policy checkpoint to fetch). Train wording only, ROLLOUTS per task per condition.
Actions are joint velocities, so only the observed joint_pos is offset; the arm itself is untouched.
"""
import os
import sys
import urllib.request
from pathlib import Path

import numpy as np
from gymnasium import ObservationWrapper

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import agent as agent_module
from agent import DEPLOYED_SUCCESS, EVAL_TASKS, Agent
from gym_robotics_custom import make_test_env
from tasks import TASKS
from vlm import JOINT_MAX, JOINT_MIN

ROLLOUTS = 20
ARM = slice(0, 7)
FINGERS = slice(7, 9)
# (name, joints, offset): arm offsets in degrees, finger offsets in millimetres; "blank" pins every joint mid-range.
CONDITIONS = [("base", None, 0), ("arm+0.5deg", ARM, 0.5), ("arm+1deg", ARM, 1), ("arm-1deg", ARM, -1),
              ("arm+2deg", ARM, 2), ("arm-2deg", ARM, -2), ("arm+5deg", ARM, 5), ("arm-5deg", ARM, -5),
              ("j2+1deg", slice(1, 2), 1), ("j2-1deg", slice(1, 2), -1), ("fingers+1mm", FINGERS, 1),
              ("fingers-1mm", FINGERS, -1), ("blank", None, None)]
offset = np.zeros(9, dtype=np.float32)
blank = [False]


def set_condition(joints, amount):
    """Point the module-level offset at one condition."""
    offset[:] = 0
    blank[0] = amount is None
    if joints is not None:
        offset[joints] = np.radians(amount) if joints != FINGERS else amount / 1000


class JointOffsetWrapper(ObservationWrapper):
    """Adds the current offset to observed joint_pos, or replaces it with mid-range when blanked."""

    def observation(self, observation):
        if blank[0]:
            joint_pos = ((JOINT_MIN + JOINT_MAX) / 2).numpy().astype(np.float32)
        else:
            joint_pos = observation["joint_pos"] + offset
        return {**observation, "joint_pos": joint_pos}


def make_offset_test_env(*args, **kwargs):
    """make_test_env with the agent's view of the joints offset."""
    return JointOffsetWrapper(make_test_env(*args, **kwargs))


agent_module.make_test_env = make_offset_test_env  # Agent.test builds its env through this name
os.makedirs("checkpoints", exist_ok=True)
urllib.request.urlretrieve(os.environ["CHECKPOINT_URL"], "checkpoints/policy.pth")
agent = Agent(eval=True, max_buffer_size=1)
agent.vlm.load_the_model("checkpoints/policy.pth")

table = []
for name, joints, amount in CONDITIONS:
    set_condition(joints, amount)
    row = [name]
    for task in EVAL_TASKS:
        cell = [agent.test(task, TASKS[task]) for _ in range(ROLLOUTS)]
        row.append(f"{sum(r for r, _ in cell)}/{sum(s in DEPLOYED_SUCCESS for _, s in cell)}")
    print(f"  condition {name}: " + "  ".join(f"{t} {c}" for t, c in zip(EVAL_TASKS, row[1:])), flush=True)
    table.append(row)

print("\nreached/deployed per task, out of", ROLLOUTS)
print("| condition | " + " | ".join(EVAL_TASKS) + " |")
for row in table:
    print("| " + " | ".join(row) + " |")
