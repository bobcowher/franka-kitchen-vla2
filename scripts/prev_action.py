"""Experiment: the previous action in the prompt where the joints went (image + language + last action).

The joint_pos slot carries the action taken one step before: in training from the demo, in rollouts from
what the policy last sent the env. Zeros at an episode's first step.
"""
import numpy as np
from gymnasium import Wrapper


def previous_action_state(self, prompts, prev_action):
    """_add_joint_state with the previous action, binned 0-99 over [-1, 1], in place of the joints."""
    bins = ((prev_action + 1) / 2 * 99).round().clamp(0, 99).int().tolist()
    return [f"{p}. Last action: {' '.join(map(str, b))}" for p, b in zip(prompts, bins)]


def previous_action_batch(dataset, sample_batch, batch_size, chunk_size=1):
    """sample_batch with joint_pos replaced by the demo's action one step earlier, zeros at an episode start."""
    state, actions, mask, tasks = sample_batch(batch_size, chunk_size)
    row = state["row"]
    prev = np.maximum(row - 1, 0)
    first = (row == 0) | (dataset.episode_end_memory[prev] != dataset.episode_end_memory[row])
    prev_action = dataset.action_memory[prev].astype(np.float32)
    prev_action[first] = 0
    state["joint_pos"] = prev_action
    return state, actions, mask, tasks


class PreviousActionWrapper(Wrapper):
    """Puts the last action sent to the env in obs['joint_pos'], zeros after reset."""

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.with_action(obs, np.zeros(self.env.action_space.shape, dtype=np.float32)), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self.with_action(obs, np.asarray(action, dtype=np.float32)), reward, terminated, truncated, info

    def with_action(self, obs, action):
        return {**obs, "joint_pos": action}
