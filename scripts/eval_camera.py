"""Experiment: deployment success when the scene camera is moved off the pose every demo was recorded from.

Env: CHECKPOINT_URL (a policy checkpoint to fetch). Train wording only, ROLLOUTS per task per pose.
"""
import os
import sys
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import agent as agent_module
from agent import DEPLOYED_SUCCESS, EVAL_TASKS, Agent
from gym_robotics_custom import make_test_env
from gymnasium_robotics.envs.franka_kitchen.franka_env import DEFAULT_CAMERA_CONFIG
from tasks import TASKS

ROLLOUTS = 20
# (name, field, offset): azimuth/elevation in degrees, distance in metres, lookat in metres along world x.
POSES = [("base", None, 0), ("az+5", "azimuth", 5), ("az-5", "azimuth", -5), ("az+15", "azimuth", 15),
         ("az-15", "azimuth", -15), ("el+5", "elevation", 5), ("el-5", "elevation", -5),
         ("dist+0.22", "distance", 0.22), ("dist-0.22", "distance", -0.22),
         ("look_x+0.05", "lookat", 0.05), ("look_x-0.05", "lookat", -0.05)]
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

table = []
for name, field, offset in POSES:
    set_pose(field, offset)
    row = [name]
    for task in EVAL_TASKS:
        cell = [agent.test(task, TASKS[task]) for _ in range(ROLLOUTS)]
        row.append(f"{sum(r for r, _ in cell)}/{sum(s in DEPLOYED_SUCCESS for _, s in cell)}")
    print(f"  pose {name}: " + "  ".join(f"{t} {c}" for t, c in zip(EVAL_TASKS, row[1:])), flush=True)
    table.append(row)

print("\nreached/deployed per task, out of", ROLLOUTS)
print("| pose | " + " | ".join(EVAL_TASKS) + " |")
for row in table:
    print("| " + " | ".join(row) + " |")
