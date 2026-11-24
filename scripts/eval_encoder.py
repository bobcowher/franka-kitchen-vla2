"""Experiment: deployment success when each joint reading carries its own constant offset, as after zeroing the arm by eye.

Each rollout draws a fresh per-joint offset, uniform within the condition's half-width, so a condition is many hand calibrations.

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
# (name, arm half-width in degrees, finger half-width in millimetres); "blank" pins every joint mid-range.
CONDITIONS = [("base", 0, 0), ("rand3deg", 3, 1), ("rand5deg", 5, 1), ("rand10deg", 10, 1), ("rand15deg", 15, 1),
              ("rand20deg", 20, 1), ("blank", None, None)]
condition = [0, 0]  # arm, finger half-widths for the running condition; None = blank


class JointOffsetWrapper(ObservationWrapper):
    """Adds a per-rollout random offset to observed joint_pos, or replaces it with mid-range when blanked."""

    def __init__(self, env):
        super().__init__(env)
        self.rng = np.random.default_rng()
        self.offset = np.zeros(9, dtype=np.float32)

    def reset(self, **kwargs):
        arm, finger = condition
        self.offset[:] = 0
        if arm is not None:
            self.offset[:7] = np.radians(self.rng.uniform(-arm, arm, 7))
            self.offset[7:] = self.rng.uniform(-finger, finger, 2) / 1000
        return super().reset(**kwargs)

    def observation(self, observation):
        if condition[0] is None:
            joint_pos = ((JOINT_MIN + JOINT_MAX) / 2).numpy().astype(np.float32)
        else:
            joint_pos = observation["joint_pos"] + self.offset
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
for name, arm, finger in CONDITIONS:
    condition[:] = [arm, finger]
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
