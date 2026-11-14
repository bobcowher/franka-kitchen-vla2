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
import time
import datetime
import subprocess
import gymnasium as gym 
import gymnasium_robotics  # registers FrankaKitchen-v1; no longer automatic in gymnasium 1.x
from torch.optim.adam import Adam
from gym_robotics_custom import ArmHomeWrapper, HeldSetpointWrapper, VLAObservationWrapper


from torch.utils.tensorboard import SummaryWriter

import frames
from dataset import Dataset
from vlm import VisionLanguageModel
from tasks import HELDOUT_PHRASES, TASK_PHRASES, TASKS, sample_phrases

from config import SMOLVLM2_CONFIG_500M, SMOLVLM2_CONFIG_2_2B

EVAL_TASKS = ["microwave", "hinge cabinet", "top burner"]
# When the policy first stopped: within done_window of the first reward, before it, not by the window's end, or (no reward) at all or never.
STOPS = ("on_time", "early", "late", "false_stop", "none")


def git_label():
    """branch_shorthash for run names; the hash pins the exact code even if the branch name is off."""
    def git(*args):
        return subprocess.run(["git", *args], capture_output=True, text=True).stdout.strip()
    return f"{git('branch', '--show-current') or 'detached'}_{git('rev-parse', '--short', 'HEAD') or 'nogit'}"


# Chunks near a demo's end run past it; mask zeroes those steps so the arm isn't taught to keep moving after the demo stops.
# Divided by the whole batch's mask count, so micro-batch pieces add up to the batch's masked mean.
def masked_mse(pred, target, mask, mask_total):
    per_step = ((pred - target) ** 2).mean(dim=-1)
    return (per_step * mask).sum() / mask_total


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
                 vlm_config=SMOLVLM2_CONFIG_500M,
                 lora_rank=16,
                 micro_batch_size=None):
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
        self.lora_rank = lora_rank  # 0 unfreezes the last four layers in full; >0 trains adapters on every layer
        self.micro_batch_size = micro_batch_size  # each batch runs in pieces this size with gradients summed; None is one piece
        self.done_window = done_window  # env steps after the first reward in which a stop counts as on time

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

        
        self.vlm = VisionLanguageModel(vlm_config=vlm_config,
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

        def lr_factor(epoch):
            if epoch < self.warmup_steps:
                return (epoch + 1) / self.warmup_steps
            progress = (epoch - self.warmup_steps) / max(1, epochs - self.warmup_steps)
            return self.min_lr_fraction + (1 - self.min_lr_fraction) * 0.5 * (1 + math.cos(math.pi * progress))
        print(f"{self.dataset.terminal_memory[:self.dataset.mem_ctr].sum()} done frames of {self.dataset.mem_ctr}", flush=True)

        for epoch in range(epochs):
            lr = self.learning_rate * lr_factor(epoch)
            self.optimizer.param_groups[0]["lr"] = lr

            states, actions, action_mask, tasks = self.dataset.sample_batch(batch_size, self.chunk_size)
           
            images, joint_pos = self.process_observation(states)

            actions = torch.tensor(actions).to(self.device).float()          # (batch, chunk, 9)
            action_mask = torch.tensor(action_mask).to(self.device).float()  # (batch, chunk)

            phrases = sample_phrases(tasks)
            done_targets = torch.tensor(self.dataset.terminal_memory[states["row"]], dtype=torch.float32, device=self.device)
            positives = done_targets.sum()
            # About 4% of batches have no done frame; clamp keeps pos_weight finite there.
            pos_weight = (batch_size - positives) / positives.clamp(min=1)
            mask_total = action_mask.sum()

            self.optimizer.zero_grad()
            arm_loss = gripper_loss = done_loss = 0.0
            piece = self.micro_batch_size or batch_size
            for start in range(0, batch_size, piece):
                part = slice(start, start + piece)
                pred_actions, done_logits = self.vlm.forward(images[part], phrases[part], joint_pos[part])
                arm     = masked_mse(pred_actions[..., :7], actions[part, :, :7], action_mask[part], mask_total)
                gripper = masked_mse(pred_actions[..., 7:], actions[part, :, 7:], action_mask[part], mask_total)
                done    = self._done_loss(done_logits, done_targets[part], pos_weight, batch_size)
                # The done head reads detached features, so its loss only reaches its own weights.
                (arm + self.gripper_weight * gripper + done).backward()
                arm_loss, gripper_loss, done_loss = arm_loss + arm.detach(), gripper_loss + gripper.detach(), done_loss + done.detach()

            loss = arm_loss + self.gripper_weight * gripper_loss
            self.optimizer.step()
            done_loss = done_loss.item()

            if(epoch % 10 == 0):
                summary_writer.add_scalar("done/loss", done_loss, epoch)
                summary_writer.add_scalar("train/loss", loss, epoch)
                summary_writer.add_scalar("train/arm", arm_loss, epoch)
                summary_writer.add_scalar("train/gripper", gripper_loss, epoch)
                summary_writer.add_scalar("train/lr", lr, epoch)

            if(epoch % 100 == 0):
                print(f"Epoch: {epoch} Loss: {loss.item()} Done loss: {done_loss:.4f}", flush=True)
                self.vlm.save_the_model()

            if(epoch and epoch % 1000 == 0):
                self.eval(epoch, summary_writer)

        self.vlm.save_the_model()
        self.eval(epochs, summary_writer, rollouts=self.final_eval_rollouts)

    def _done_loss(self, logits, targets, pos_weight, batch_size):
        """BCE on the demo's done label, done frames weighted to count as much as all the others together.

        Summed and divided by the whole batch size, so micro-batch pieces add up to the batch mean."""
        return F.binary_cross_entropy_with_logits(logits.float(), targets, pos_weight=pos_weight, reduction="sum") / batch_size

    def eval(self, epoch, summary_writer, rollouts=None):
        # Kept per eval so a good point in training can't be overwritten by later saves.
        self.vlm.save_the_model(f"checkpoints/smolvlm2-step{epoch}.pth")
        rollouts = rollouts or self.eval_rollouts
        for wording, phrases in (("train", TASK_PHRASES), ("heldout", HELDOUT_PHRASES)):
            for task in EVAL_TASKS:
                results = [self.test(task, phrases[task][i % len(phrases[task])]) for i in range(rollouts)]
                counts = {"reached": sum(reached for reached, _ in results)}
                counts.update({s: sum(stop == s for _, stop in results) for s in STOPS})
                tag = f"eval_{wording}/{task.replace(' ', '_')}"
                for name in ("reached", "on_time", "early", "false_stop"):
                    summary_writer.add_scalar(f"{tag}/{name}", counts[name] / rollouts, epoch)
                print(f"  eval {wording} {task}: {counts}", flush=True)

    def reset_policy(self):
        """Forget any queued actions; call at the start of every episode."""
        self.action_queue = []
        self.done_prob = 0.0
        self.done_streak = 0

    def get_action(self, images, prompt, joint_pos):
        """Next action for one env. The model only runs when the queue is empty, i.e. every execute_steps steps."""
        if not self.action_queue:
            chunks, done_prob = self.vlm.predict(images, [prompt], joint_pos)
            self.action_queue = list(chunks[0][:self.execute_steps].cpu().numpy())
            self.done_prob = done_prob.item()
            self.done_streak = self.done_streak + 1 if self.done_prob >= self.done_threshold else 0
        return self.action_queue.pop(0)

    def make_session_env(self, render_mode="human"):
        """An env for many commands in a row: it never ends on success and tracks every task."""
        env = self._make_env(list(TASKS), render_mode, terminate_on_tasks_completed=False, remove_task_when_completed=False)
        return ArmHomeWrapper(env)

    def run_command(self, env, obs, prompt, slowdown=1.0):
        """Act until the done rule fires or time runs out, then go home.

        Returns (obs, done fired, tasks complete when the policy stopped, tasks complete after homing)."""
        self.reset_policy()
        self.vlm.model.eval()
        last_step = time.time()

        def step(action):
            nonlocal obs, last_step
            obs, _, _, _, step_info = env.step(action)
            # Sleep for (slowdown - 1) times what the step took, so slowdown=2 runs at half speed.
            time.sleep((slowdown - 1) * (time.time() - last_step))
            last_step = time.time()
            return step_info

        fired = False
        info = {"step_task_completions": []}
        for _ in range(self.max_episode_steps):
            images, joint_pos = self.process_observation(obs)
            action = self.get_action(images, prompt, joint_pos)
            if self.done_streak >= self.done_patience:
                fired = True
                break
            info = step(action)
        at_stop = info["step_task_completions"]

        for action in itertools.islice(env.home_actions(), env.max_steps):
            step(action)
        worst = np.abs(env.home - env.unwrapped.data.qpos[:9]).max()
        if worst >= env.tolerance:
            print(f"Homing gave up after {env.max_steps} steps, worst joint still {worst:.3f} rad off", flush=True)

        info = step(np.zeros(env.action_space.shape[0]))
        self.vlm.model.train()
        return obs, fired, at_stop, info["step_task_completions"]

    def test(self, task, prompt=None, render_mode="rgb_array"):
        """One fresh-scene rollout scored two ways; returns (reached, one of STOPS).

        reached: the task gave a reward at any step. The stop is judged on its own:
        a stop does not end the rollout, so an early stop can't hide a reach.
        It runs until done_window steps after the first reward, or the step limit
        if the task is never reached. There is no homing; it has its own test."""
        env = self._make_env(task, render_mode, max_episode_steps=self.max_episode_steps + self.done_window,
                             terminate_on_tasks_completed=False, remove_task_when_completed=False)
        obs, _ = env.reset()
        self.reset_policy()
        self.vlm.model.eval()
        reward_at = stop_at = None
        for t in range(self.max_episode_steps + self.done_window):
            if reward_at is None and t >= self.max_episode_steps:
                break
            images, joint_pos = self.process_observation(obs)
            action = self.get_action(images, prompt or TASKS[task], joint_pos)
            if stop_at is None and self.done_streak >= self.done_patience:
                stop_at = t
            # Past the first reward, the first stop or the window's end settles the outcome.
            if reward_at is not None and (stop_at is not None or t >= reward_at + self.done_window):
                break
            obs, _, _, _, info = env.step(action)
            if reward_at is None and task in info["step_task_completions"]:
                reward_at = t + 1
        env.close()
        self.vlm.model.train()
        if reward_at is None:
            return False, "none" if stop_at is None else "false_stop"
        if stop_at is None:
            return True, "late"
        return True, "early" if stop_at < reward_at else "on_time"
