import os

# MuJoCo picks its GL backend at import time and defaults to GLFW, which needs
# an X display. Headless training servers have none, so fall back to EGL there.
# setdefault, so the environment can still override.
if not os.environ.get("DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch
import torch.nn.functional as F
import itertools
import math
import datetime
import subprocess
import gymnasium as gym 
import gymnasium_robotics  # registers FrankaKitchen-v1; no longer automatic in gymnasium 1.x
from torch.optim.adam import Adam
from gym_robotics_custom import ArmHomeWrapper, CloseTaskWrapper, HeldSetpointWrapper, RealTimeWrapper, VLAObservationWrapper


from torch.utils.tensorboard import SummaryWriter

import frames
from dataset import Dataset
from vlm import VisionLanguageModel
from tasks import CLOSE_PHRASES, CLOSE_TASKS, HELDOUT_PHRASES, TASK_PHRASES, TASKS, sample_phrases

from config import SMOLVLM2_CONFIG_500M

EVAL_TASKS = ["microwave", "hinge cabinet", "top burner"]
# How a rollout's first stop went. Deployment succeeds on on_time and late: the task was complete when the policy stopped.
STOPS = (
    "on_time",     # complete at the stop, within done_window steps of the first reward
    "late",        # complete at the stop, but later than that
    "undone",      # stopped after the task completed, by which time it no longer was
    "early",       # stopped before the task completed; it completed afterwards
    "never",       # completed, never stopped
    "false_stop",  # stopped, never completed
    "none",        # never completed, never stopped
)
DEPLOYED_SUCCESS = ("on_time", "late")


def git(*args):
    """A git command's stdout, or "" if git fails."""
    return subprocess.run(["git", *args], capture_output=True, text=True).stdout.strip()


def git_label():
    """branch_shorthash for run names; the hash pins the exact code even if the branch name is off."""
    return f"{git('branch', '--show-current') or 'detached'}_{git('rev-parse', '--short', 'HEAD') or 'nogit'}"


# Chunks near a demo's end run past it; mask zeroes those steps so the arm isn't taught to keep moving after the demo stops.
def masked_mse(pred, target, mask):
    per_step = ((pred - target) ** 2).mean(dim=-1)
    return (per_step * mask).sum() / mask.sum()


class Agent:

    def __init__(self, eval=False, data_path="dataset",
                 learning_rate=0.0001,
                 gripper_weight=0.125,
                 chunk_size=8,
                 execute_steps=4,
                 warmup_steps=200,
                 min_lr_fraction=0.1,
                 eval_rollouts=3,
                 final_eval_rollouts=20,
                 done_threshold=0.5,
                 done_patience=2,
                 done_window=20,
                 done_learning_rate=3e-4,
                 lora_rank=16,
                 close_band_fraction=0.2):
        # 0.26 (L1-equalized) left gripper error unchanged and drove the arm off target in run 4; revisit once the model sees joint state.
        self.gripper_weight = gripper_weight
        # Action chunking, as in ACT and diffusion policy: predict the next chunk_size actions, run the first execute_steps, then look again.
        self.chunk_size = chunk_size
        self.execute_steps = execute_steps
        # Linear warmup over warmup_steps, then cosine decay to min_lr_fraction of the peak LR by the last epoch.
        self.warmup_steps = warmup_steps
        self.min_lr_fraction = min_lr_fraction
        self.eval_rollouts = eval_rollouts  # per task and wording, every 1000 steps
        self.final_eval_rollouts = final_eval_rollouts
        self.learning_rate = learning_rate
        # Stop once done_prob reaches done_threshold on done_patience replans in a row.
        self.done_threshold = done_threshold
        self.done_patience = done_patience
        self.done_window = done_window  # env steps after the first reward in which a stop counts as on time
        self.lora_rank = lora_rank  # adapter rank on every attention and MLP projection
        self.close_band_fraction = close_band_fraction  # a close task is done within this fraction of the door's span from shut

        self.max_episode_steps = 400  # longest demo on file is 314; a policy still going at 400 has failed
        self.image_size = frames.VLM_SIZE  # matches the processor, so it skips resizing
        max_buffer_size = 100000

        env = self._make_env(EVAL_TASKS[0], render_mode='rgb_array')
        num_actions = env.action_space.shape[0]
        self.dataset = Dataset(max_size=max_buffer_size,
                               image_size=self.image_size,
                               n_actions=num_actions,
                               n_joints=env.observation_space["joint_pos"].shape[0])
        env.close()

        if not eval:
            self.dataset.load_data(path=data_path)

        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        
        self.vlm = VisionLanguageModel(vlm_config=SMOLVLM2_CONFIG_500M,
                                       num_actions=num_actions,
                                       chunk_size=self.chunk_size,
                                       lora_rank=lora_rank)
        self.reset_policy()

        # Group 0 is the policy, which follows the LR schedule; group 1 is the done head at a constant rate.
        self.optimizer = Adam([{"params": [p for n, p in self.vlm.model.named_parameters()
                                           if p.requires_grad and not n.startswith("done_head.")]},
                               {"params": self.vlm.model.done_head.parameters(), "lr": done_learning_rate}],
                              self.learning_rate)

    def _make_env(self, task, render_mode, max_episode_steps=None, **kitchen_args):
        tasks = task if isinstance(task, list) else [task]
        env = gym.make("FrankaKitchen-v1", max_episode_steps=max_episode_steps or self.max_episode_steps,
                       tasks_to_complete=tasks, render_mode=render_mode, **kitchen_args)
        if render_mode == "human":
            env = RealTimeWrapper(env)
        env = HeldSetpointWrapper(env)
        return VLAObservationWrapper(env, image_size=self.image_size)

    def process_observation(self,obs):
        images    = obs['camera_scene']
        joint_pos = torch.tensor(obs['joint_pos'], dtype=torch.float32).to(self.device)

        # Images stay uint8 HWC numpy for the processor, which moves them to
        # the GPU itself. A single frame from the env gets a batch dim here.
        if images.ndim == 3:
            images    = images[None]
            joint_pos = joint_pos.unsqueeze(0)

        return list(images), joint_pos



    def train(self, epochs, batch_size):
        summary_writer_name = f'runs/{git_label()}_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
        summary_writer_name = summary_writer_name + f"_bs={batch_size}"
        summary_writer = SummaryWriter(summary_writer_name)

        print(f"{self.dataset.terminal_memory[:self.dataset.mem_ctr].sum()} done frames of {self.dataset.mem_ctr}", flush=True)

        for epoch in range(epochs):
            lr = self.learning_rate * self._lr_factor(epoch, epochs)
            self.optimizer.param_groups[0]["lr"] = lr

            states, actions, action_mask, tasks = self.dataset.sample_batch(batch_size, self.chunk_size)
           
            images, joint_pos = self.process_observation(states)

            actions = torch.tensor(actions).to(self.device).float()          # (batch, chunk, 9)
            action_mask = torch.tensor(action_mask).to(self.device).float()  # (batch, chunk)

            pred_actions, done_logits = self.vlm.forward(images, sample_phrases(tasks), joint_pos)

            arm_loss     = masked_mse(pred_actions[..., :7], actions[..., :7], action_mask)
            gripper_loss = masked_mse(pred_actions[..., 7:], actions[..., 7:], action_mask)

            loss = arm_loss + self.gripper_weight * gripper_loss
            done_loss = self._done_loss(done_logits, states["row"])

            self.optimizer.zero_grad()

            # The done head reads detached features, so its loss only reaches its own weights.
            (loss + done_loss).backward()

            self.optimizer.step()
            done_loss = done_loss.item()

            if(epoch % 10 == 0):
                summary_writer.add_scalar("loss/action", loss, epoch)  # what the policy minimizes: arm + gripper_weight * gripper
                summary_writer.add_scalar("loss/arm_joints", arm_loss, epoch)
                summary_writer.add_scalar("loss/gripper", gripper_loss, epoch)
                summary_writer.add_scalar("loss/done", done_loss, epoch)
                summary_writer.add_scalar("optim/lr", lr, epoch)

            if(epoch % 100 == 0):
                print(f"Epoch: {epoch} Loss: {loss.item()} Done loss: {done_loss:.4f}", flush=True)
                self.vlm.save_the_model()

            if(epoch and epoch % 1000 == 0):
                self.eval(epoch, summary_writer)

        self.vlm.save_the_model()
        self.eval(epochs, summary_writer, rollouts=self.final_eval_rollouts)

    def _lr_factor(self, epoch, epochs):
        """Linear warmup over warmup_steps, then cosine decay to min_lr_fraction by the last epoch."""
        if epoch < self.warmup_steps:
            return (epoch + 1) / self.warmup_steps
        progress = (epoch - self.warmup_steps) / max(1, epochs - self.warmup_steps)
        return self.min_lr_fraction + (1 - self.min_lr_fraction) * 0.5 * (1 + math.cos(math.pi * progress))

    def _done_loss(self, logits, batch_rows):
        """BCE on the demo's done label, with done frames weighted to count as much as all the others together."""
        targets = torch.tensor(self.dataset.terminal_memory[batch_rows], dtype=logits.dtype, device=logits.device)
        positives = targets.sum()
        # About 4% of batches have no done frame; clamp keeps pos_weight finite there.
        return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=(len(targets) - positives) / positives.clamp(min=1))

    def eval(self, epoch, summary_writer, rollouts=None):
        # Kept per eval so a good point in training can't be overwritten by later saves.
        self.vlm.save_the_model(f"checkpoints/smolvlm2-step{epoch}.pth")
        rollouts = rollouts or self.eval_rollouts
        rows = []  # (wording, task, reached, stop), one per rollout
        for wording, phrases in (("train", TASK_PHRASES), ("heldout", HELDOUT_PHRASES)):
            for task in EVAL_TASKS:
                cell = [self.test(task, phrases[task][i % len(phrases[task])]) for i in range(rollouts)]
                counts = {"reached": sum(reached for reached, _ in cell)}
                counts.update({s: sum(stop == s for _, stop in cell) for s in STOPS})
                print(f"  eval {wording} {task}: {counts}", flush=True)
                rows += [(wording, task, reached, stop) for reached, stop in cell]

        # TensorBoard gets totals; per-cell detail (early, late, ...) is in the printout above.
        groups = [("all", rows), ("heldout_wording", [r for r in rows if r[0] == "heldout"])]
        groups += [(task.replace(" ", "_"), [r for r in rows if r[1] == task]) for task in EVAL_TASKS]
        for name, picked in groups:
            summary_writer.add_scalar(f"eval_success/{name}", np.mean([r[2] for r in picked]), epoch)
            summary_writer.add_scalar(f"eval_stop/{name}", np.mean([r[3] in DEPLOYED_SUCCESS for r in picked]), epoch)

    def reset_policy(self):
        """Forget any queued actions; call at the start of every episode."""
        self.action_queue = []
        self.done_streak = 0

    def get_action(self, obs, prompt):
        """Next action for one env. The model only runs when the queue is empty, i.e. every execute_steps steps."""
        if not self.action_queue:
            images, joint_pos = self.process_observation(obs)
            chunks, done_prob = self.vlm.predict(images, [prompt], joint_pos)
            self.action_queue = list(chunks[0][:self.execute_steps].cpu().numpy())
            self.done_streak = self.done_streak + 1 if done_prob.item() >= self.done_threshold else 0
        return self.action_queue.pop(0)

    def stopping(self):
        """The done rule: done_prob at or above done_threshold on done_patience replans in a row."""
        return self.done_streak >= self.done_patience

    def make_session_env(self, render_mode="human"):
        """An env for many commands in a row: it never ends on success and tracks every task."""
        env = self._make_env(list(TASKS), render_mode, terminate_on_tasks_completed=False, remove_task_when_completed=False)
        return ArmHomeWrapper(env)

    def make_test_env(self, task, render_mode="rgb_array"):
        """A session env for one test rollout; a CLOSE_TASKS task starts with its cabinet open."""
        env = self.make_session_env(render_mode)
        if task in CLOSE_TASKS:
            env = CloseTaskWrapper(env, CLOSE_TASKS[task], self.close_band_fraction)
        return env

    def rollout(self, env, obs, prompt, watch=None):
        """Runs the policy from obs, the one loop both play.py and the eval use.

        watch=None ends at the first stop, as a deployed robot would. watch=task keeps
        going past an early stop until that task completes, so the eval also learns
        whether the arm would have finished; every step up to the stop is the same.
        Returns (obs, step of the first stop or None, tasks complete at the stop or
        at the end, {task: step it first completed}). Steps count env steps taken."""
        self.reset_policy()
        self.vlm.model.eval()
        stop_at, at_stop, first_done = None, None, {}
        completions = []  # tasks complete after the last step taken
        for t in range(self.max_episode_steps):
            action = self.get_action(obs, prompt)
            if stop_at is None and self.stopping():
                stop_at, at_stop = t, completions
            if stop_at is not None and (watch is None or watch in first_done):
                break
            obs, _, _, _, info = env.step(action)
            completions = info["step_task_completions"]
            for task in completions:
                first_done.setdefault(task, t + 1)
        self.vlm.model.train()
        return obs, stop_at, completions if at_stop is None else at_stop, first_done

    def run_command(self, env, obs, prompt):
        """Act until the done rule fires or time runs out, then go home.

        Returns (obs, done fired, tasks complete when the policy stopped, tasks complete after homing)."""
        obs, stop_at, at_stop, _ = self.rollout(env, obs, prompt)
        for action in itertools.islice(env.home_actions(), env.max_steps):
            obs, _, _, _, info = env.step(action)
        worst = np.abs(env.home - env.unwrapped.data.qpos[:9]).max()
        if worst >= env.tolerance:
            print(f"Homing gave up after {env.max_steps} steps, worst joint still {worst:.3f} rad off", flush=True)
        obs, _, _, _, info = env.step(np.zeros(env.action_space.shape[0]))
        return obs, stop_at is not None, at_stop, info["step_task_completions"]

    def test(self, task, prompt=None, render_mode="rgb_array"):
        """One fresh-scene rollout in play.py's env and loop; returns (reached, one of STOPS).

        reached: the task completed at some step, whether or not the policy had already
        stopped. No homing: it comes after the stop, and has its own paired test."""
        env = self.make_test_env(task, render_mode)
        obs, _ = env.reset()
        default_prompt = CLOSE_PHRASES[task][0] if task in CLOSE_TASKS else TASKS[task]
        _, stop_at, at_stop, first_done = self.rollout(env, obs, prompt or default_prompt, watch=task)
        env.close()
        reached_at = first_done.get(task)
        if stop_at is None:
            return reached_at is not None, "none" if reached_at is None else "never"
        if reached_at is None:
            return False, "false_stop"
        if stop_at < reached_at:
            return True, "early"
        if task not in at_stop:
            return True, "undone"
        return True, "on_time" if stop_at - reached_at < self.done_window else "late"
