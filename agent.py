import os

# MuJoCo picks its GL backend at import time and defaults to GLFW, which needs
# an X display. Headless training servers have none, so fall back to EGL there.
# setdefault, so the environment can still override.
if not os.environ.get("DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch
import torch.nn.functional as F
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

from config import SMOLVLM2_CONFIG_500M

EVAL_TASKS = ["microwave", "hinge cabinet", "top burner"]
# How a rollout ended under the stop rule, judged when the policy stopped; success is correct or miss.
OUTCOMES = ("correct", "false_stop", "miss", "quiet_fail")


def git_label():
    """branch_shorthash for run names; the hash pins the exact code even if the branch name is off."""
    def git(*args):
        return subprocess.run(["git", *args], capture_output=True, text=True).stdout.strip()
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
                 done_learning_rate=3e-4):
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
                                       chunk_size=self.chunk_size)
        self.reset_policy()

        # Group 0 is the policy, which follows the LR schedule; group 1 is the done output layer at a constant rate.
        self.optimizer = Adam([{"params": [p for n, p in self.vlm.model.named_parameters()
                                           if p.requires_grad and not n.startswith("lm_head.done_output.")]},
                               {"params": self.vlm.model.lm_head.done_output.parameters(), "lr": done_learning_rate}],
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

            pred_actions, done_logits = self.vlm.forward(images, sample_phrases(tasks), joint_pos)

            arm_loss     = masked_mse(pred_actions[..., :7], actions[..., :7], action_mask)
            gripper_loss = masked_mse(pred_actions[..., 7:], actions[..., 7:], action_mask)

            loss = arm_loss + self.gripper_weight * gripper_loss
            done_loss = self._done_loss(done_logits, states["row"])

            self.optimizer.zero_grad()

            # The done loss also trains the shared head layers and the unfrozen backbone.
            (loss + done_loss).backward()

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
        for wording, phrases in (("train", TASK_PHRASES), ("heldout", HELDOUT_PHRASES)):
            for task in EVAL_TASKS:
                results = [self.test(task, phrases[task][i % len(phrases[task])]) for i in range(rollouts)]
                outcomes = [outcome for outcome, _ in results]
                counts = {o: outcomes.count(o) for o in OUTCOMES}
                counts["undone_by_homing"] = sum(undone for _, undone in results)
                tag = f"eval_{wording}/{task.replace(' ', '_')}"
                summary_writer.add_scalar(f"{tag}/success", (counts["correct"] + counts["miss"]) / rollouts, epoch)
                summary_writer.add_scalar(f"{tag}/false_stop", counts["false_stop"] / rollouts, epoch)
                summary_writer.add_scalar(f"{tag}/undone_by_homing", counts["undone_by_homing"] / rollouts, epoch)
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

        for _ in range(env.max_steps):
            action = env.home_action()
            if action is None:
                break
            step(action)
        else:
            print(f"Homing gave up after {env.max_steps} steps, worst joint still "
                  f"{np.abs(env.home - env.unwrapped.data.qpos[:9]).max():.3f} rad off", flush=True)

        info = step(np.zeros(env.action_space.shape[0]))
        self.vlm.model.train()
        return obs, fired, at_stop, info["step_task_completions"]

    def test(self, task, prompt=None, render_mode="rgb_array", slowdown=1.0):
        """One fresh-scene rollout run as play.py runs a command; returns (one of OUTCOMES, undone by homing)."""
        env = ArmHomeWrapper(self._make_env(task, render_mode, terminate_on_tasks_completed=False,
                                            remove_task_when_completed=False))
        obs, _ = env.reset()
        _, fired, at_stop, after_home = self.run_command(env, obs, prompt or TASKS[task], slowdown)
        env.close()
        done = task in at_stop
        outcome = ("correct" if done else "false_stop") if fired else ("miss" if done else "quiet_fail")
        return outcome, done and task not in after_home
