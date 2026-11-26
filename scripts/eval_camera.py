"""Experiment: stricter stop rules on hinge cabinet at the poses where seed 2 (run 140) stopped early (run 143).

Env: CHECKPOINT_URL (a policy checkpoint to fetch). Train wording only, ROLLOUTS per task per pose.
"""
import os
from collections import Counter
import sys
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import agent as agent_module
from agent import STOPS, Agent
from gym_robotics_custom import make_test_env
from gymnasium_robotics.envs.franka_kitchen.franka_env import DEFAULT_CAMERA_CONFIG
from tasks import TASKS

ROLLOUTS = 20
# (name, field, offset): azimuth/elevation in degrees, distance in metres, lookat in metres along world x.
POSES = [("base", None, 0), ("az+5", "azimuth", 5), ("el-5", "elevation", -5), ("look_x-0.05", "lookat", -0.05)]
TASK = "hinge cabinet"
# (done_threshold, done_patience); run 143 was the default (0.5, 2).
RULES = [(0.5, 4), (0.5, 8), (0.8, 2)]
camera = dict(DEFAULT_CAMERA_CONFIG)


def set_pose(field, offset):
    """Point the module-level camera config at one perturbed pose."""
    camera.clear()
    camera.update({k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in DEFAULT_CAMERA_CONFIG.items()})
    if field == "lookat":
        camera["lookat"][0] += offset
    elif field is not None:
        camera[field] += offset


def make_moved_test_env(*args, **kwargs):
    """make_test_env with the free camera at the current pose; must be set before the first render."""
    env = make_test_env(*args, **kwargs)
    env.unwrapped.robot_env.mujoco_renderer.default_cam_config = dict(camera)
    return env


agent_module.make_test_env = make_moved_test_env  # Agent.test builds its env through this name
os.makedirs("checkpoints", exist_ok=True)
urllib.request.urlretrieve(os.environ["CHECKPOINT_URL"], "checkpoints/policy.pth")
agent = Agent(eval=True, max_buffer_size=1)
agent.vlm.load_the_model("checkpoints/policy.pth")

for threshold, patience in RULES:
    agent.done_threshold, agent.done_patience = threshold, patience
    for name, field, offset in POSES:
        set_pose(field, offset)
        cell = [agent.test(TASK, TASKS[TASK]) for _ in range(ROLLOUTS)]
        stops = Counter(s for _, s in cell)
        print(f"  rule {threshold}/{patience} pose {name}: reached {sum(r for r, _ in cell)}/{ROLLOUTS}  "
              + "  ".join(f"{s} {stops[s]}" for s in STOPS), flush=True)
