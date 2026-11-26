"""Experiment: run 118's offset joints (proprio C, no dropout) on run 103's demos plus copies re-rendered from random scene-camera poses.

Replay from stored qpos is exact, so each copy is the same demo seen from a moved camera. The pose is
drawn once per episode, the way a mounted camera would sit. Env: TASK_DIRS (colon-separated v2 folders).
"""
import glob
import os
import sys
import time
from pathlib import Path

import functools

import gymnasium as gym
import gymnasium_robotics  # noqa: F401  registers FrankaKitchen-v1
import mujoco
import numpy as np
from gymnasium_robotics.envs.franka_kitchen.franka_env import DEFAULT_CAMERA_CONFIG

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import frames
import vlm as vlm_module
from agent import Agent
from dataset import _open

ARM_OFFSET = np.radians(3.0)  # uniform half-width per arm joint, as in run 119
FINGER_OFFSET = 0.001  # metres
DROPOUT = 0.0  # proprio C: encoders hold zero across power cycles, so joints are never missing


class JitterRenderer:
    """Renders a demo's stored qpos from a random free-camera pose, reduced the way load_data reduces shards."""

    def __init__(self, image_size=frames.VLM_SIZE, render_size=frames.RENDER_SIZE, azimuth=20.0,
                 elevation=8.0, distance=0.3, lookat=0.06, seed=0):
        self.image_size = image_size
        # Uniform half-widths: degrees, degrees, metres, metres per lookat axis; wider than eval_camera's poses.
        self.azimuth, self.elevation, self.distance, self.lookat = azimuth, elevation, distance, lookat
        self.rng = np.random.default_rng(seed)
        self.env = gym.make("FrankaKitchen-v1", tasks_to_complete=["kettle"], render_mode="rgb_array")
        robot = self.env.unwrapped.robot_env
        self.renderer = robot.mujoco_renderer
        self.model, self.data = robot.model, robot.data
        robot.width = robot.height = self.renderer.width = self.renderer.height = render_size
        self.model.vis.global_.offwidth = self.model.vis.global_.offheight = render_size
        self.env.reset(seed=0)

    def sample_pose(self):
        """One camera config drawn around the recorded pose."""
        pose = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in DEFAULT_CAMERA_CONFIG.items()}
        pose["azimuth"] += self.rng.uniform(-self.azimuth, self.azimuth)
        pose["elevation"] += self.rng.uniform(-self.elevation, self.elevation)
        pose["distance"] += self.rng.uniform(-self.distance, self.distance)
        pose["lookat"] = pose["lookat"] + self.rng.uniform(-self.lookat, self.lookat, 3)
        return pose

    def render_episode(self, qpos, pose):
        """Frames for every step of qpos (T, nq) from pose, at image_size."""
        self.renderer.default_cam_config = pose
        self.renderer.camera_id = -1
        out = np.empty((len(qpos), self.image_size, self.image_size, 3), dtype=np.uint8)
        for t, q in enumerate(qpos):
            self.data.qpos[:] = q[:self.model.nq]
            self.data.qvel[:] = 0
            mujoco.mj_forward(self.model, self.data)
            if t == 0:
                self.renderer._get_viewer("rgb_array")
                self.renderer._set_cam_config()  # the viewer only reads default_cam_config when it is created
            out[t] = frames.resize(self.renderer.render("rgb_array"), self.image_size)
        return out


def augmented_sample_batch(sample_batch, batch_size, chunk_size=1):
    """sample_batch with a random constant offset on every sample's joints, and half pinned mid-range."""
    state, actions, mask, tasks = sample_batch(batch_size, chunk_size)
    joints = state["joint_pos"].copy()
    joints[:, :7] += np.random.uniform(-ARM_OFFSET, ARM_OFFSET, (len(joints), 7))
    joints[:, 7:] += np.random.uniform(-FINGER_OFFSET, FINGER_OFFSET, (len(joints), joints.shape[1] - 7))
    drop = np.random.rand(len(joints)) < DROPOUT
    joints[drop] = ((vlm_module.JOINT_MIN + vlm_module.JOINT_MAX) / 2).numpy()
    state["joint_pos"] = joints.astype(np.float32)
    return state, actions, mask, tasks


if __name__ == "__main__":
    data_root = "/data/datasets/franka-kitchen-v2"
    task_dirs = os.environ["TASK_DIRS"].split(":")
    copies = 2  # jittered copies per demo, on top of the original
    shards = [f for d in task_dirs for f in sorted(glob.glob(f"{data_root}/{d}/**/*.npz", recursive=True))]
    steps = sum(len(_open(f)["action"]) for f in shards)
    print(f"{len(shards)} demos, {steps} steps; {copies} jittered copies -> {steps * (copies + 1)} rows", flush=True)

    agent = Agent(data_path=f"{data_root}/{task_dirs[0]}", max_buffer_size=steps * (copies + 1))
    for task_dir in task_dirs[1:]:
        agent.dataset.load_data(path=f"{data_root}/{task_dir}")

    jitter = JitterRenderer()
    start = time.time()
    for copy in range(copies):
        for shard in shards:
            data = _open(shard)
            agent.dataset.add_episode(jitter.render_episode(data["qpos"], jitter.sample_pose()), data)
        print(f"Jittered copy {copy + 1}/{copies} done, {agent.dataset.mem_ctr} rows, {time.time() - start:.0f}s", flush=True)

    agent.dataset.sample_batch = functools.partial(augmented_sample_batch, agent.dataset.sample_batch)
    agent.train(epochs=10000, batch_size=64)  # 3x the rows; at 5000 the done head was still at 0.1-0.2 loss
