"""Experiment: run 103's config with the joint input changed, for robustness to encoder offsets.

Env: TASK_DIRS (colon-separated v2 folders), PROPRIO_MODE:
  none            joint text dropped from the prompt (image + language only)
  vel             joint velocities in place of positions (offset-invariant)
  offset          positions, plus a random constant offset per training sample
  offset_dropout  offset, and half the samples pinned mid-range (what eval_encoder's blank shows)
"""
import functools
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import dataset as dataset_module
import gym_robotics_custom
import vlm as vlm_module
from agent import Agent

DATA = "/data/datasets/franka-kitchen-v2"
ARM_OFFSET = np.radians(3.0)  # uniform half-width per arm joint; eval_encoder sweeps 0.5-5 deg
FINGER_OFFSET = 0.001  # metres
DROPOUT = 0.5


class VelocityShard:
    """An npz shard whose joint_pos reads as joint_vel."""

    def __init__(self, data):
        self.data = data

    def __getitem__(self, key):
        return self.data["joint_vel" if key == "joint_pos" else key]

    def __contains__(self, key):
        return key in self.data


def open_velocity_shard(filename):
    return VelocityShard(np.load(filename))


def velocity_observation(self, observation):
    """VLAObservationWrapper.observation with joint velocities (robot_obs[9:18], as the collector stored them)."""
    self._renderer.camera_id = -1
    frame = self._renderer.render("rgb_array")
    return {"camera_scene": gym_robotics_custom.frames.resize(frame, self.image_size),
            "joint_pos": observation["observation"][9:9 + self.n_joints].astype(np.float32)}


def no_joint_state(self, prompts, joint_pos):
    """_add_joint_state with the joints left out."""
    return list(prompts)


def augmented_sample_batch(sample_batch, dropout, batch_size, chunk_size=1):
    """sample_batch with a random constant offset on every sample's joints, and optionally half pinned mid-range."""
    state, actions, mask, tasks = sample_batch(batch_size, chunk_size)
    joints = state["joint_pos"].copy()
    joints[:, :7] += np.random.uniform(-ARM_OFFSET, ARM_OFFSET, (len(joints), 7))
    joints[:, 7:] += np.random.uniform(-FINGER_OFFSET, FINGER_OFFSET, (len(joints), joints.shape[1] - 7))
    if dropout:
        drop = np.random.rand(len(joints)) < DROPOUT
        joints[drop] = ((vlm_module.JOINT_MIN + vlm_module.JOINT_MAX) / 2).numpy()
    state["joint_pos"] = joints.astype(np.float32)
    return state, actions, mask, tasks


mode = os.environ["PROPRIO_MODE"]
task_dirs = os.environ["TASK_DIRS"].split(":")
print(f"Proprio mode {mode} on {task_dirs}", flush=True)

if mode == "vel":
    dataset_module._open = open_velocity_shard
    gym_robotics_custom.VLAObservationWrapper.observation = velocity_observation
if mode == "none":
    vlm_module.VisionLanguageModel._add_joint_state = no_joint_state

agent = Agent(data_path=f"{DATA}/{task_dirs[0]}", max_buffer_size=30000)
for task_dir in task_dirs[1:]:
    agent.dataset.load_data(path=f"{DATA}/{task_dir}")

if mode == "vel":
    # Bin velocities over their own range: 1st-99th percentile per joint, so outliers don't squash the bins.
    vel = agent.dataset.joint_pos_memory[:agent.dataset.mem_ctr]
    vlm_module.JOINT_MIN = torch.tensor(np.percentile(vel, 1, axis=0), dtype=torch.float32)
    vlm_module.JOINT_MAX = torch.tensor(np.percentile(vel, 99, axis=0), dtype=torch.float32)
    print("velocity bins", vlm_module.JOINT_MIN.tolist(), vlm_module.JOINT_MAX.tolist(), flush=True)
if mode in ("offset", "offset_dropout"):
    agent.dataset.sample_batch = functools.partial(augmented_sample_batch, agent.dataset.sample_batch,
                                                   mode == "offset_dropout")

agent.train(epochs=5000, batch_size=64)
