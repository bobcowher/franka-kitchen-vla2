import os

# MuJoCo picks its GL backend at import time and defaults to GLFW, which needs
# an X display. Headless training servers have none, so fall back to EGL there.
# setdefault, so the environment can still override.
if not os.environ.get("DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "egl")

import torch
import torch.nn.functional as F
import math
import time
import datetime
import subprocess
import gymnasium as gym 
import gymnasium_robotics  # registers FrankaKitchen-v1; no longer automatic in gymnasium 1.x
from torch.optim.adam import Adam
from gym_robotics_custom import HeldSetpointWrapper, VLAObservationWrapper, ObsReshapeWrapper 


from torch.utils.tensorboard import SummaryWriter

import frames
from dataset import Dataset
from vlm import VisionLanguageModel
from tasks import eval_phrase, sample_phrases

from config import SMOLVLM2_CONFIG_500M

EVAL_TASKS = ["microwave", "hinge cabinet", "top burner"]


def git_label():
    """branch_shorthash for run names; the hash pins the exact code even if the branch name is off."""
    def git(*args):
        return subprocess.run(["git", *args], capture_output=True, text=True).stdout.strip()
    return f"{git('branch', '--show-current') or 'detached'}_{git('rev-parse', '--short', 'HEAD') or 'nogit'}"


# Per-joint min/max over the dataset; fingers (last two) span only 0-0.049 m, so each joint is binned to its own range.
JOINT_MIN = torch.tensor([-2.9014, -1.7744, -2.3526, -3.0786, -0.7136, -0.0261, -0.0030, -0.0026, -0.0027])
JOINT_MAX = torch.tensor([ 2.2670, -0.9490,  2.5636, -0.3810,  1.5339,  2.1493,  2.5794,  0.0489,  0.0428])


def with_joint_state(prompts, joint_pos):
    """Append joint positions as 0-99 integer bins: ~25 tokens, equal resolution per joint."""
    lo, hi = JOINT_MIN.to(joint_pos.device), JOINT_MAX.to(joint_pos.device)
    bins = ((joint_pos - lo) / (hi - lo) * 99).round().clamp(0, 99).int().tolist()
    return [f"{p}. Joints: {' '.join(map(str, b))}" for p, b in zip(prompts, bins)]

def masked_mse(pred, target, mask):
    """MSE over the chunk steps that exist; mask is 0 for steps past the end of the episode."""
    per_step = ((pred - target) ** 2).mean(dim=-1)
    return (per_step * mask).sum() / mask.sum()


class Agent:

    def __init__(self, eval=False, data_path="dataset", name='bc_network',
                 learning_rate=0.0001,
                 gripper_weight=0.125,
                 chunk_size=8,
                 execute_steps=4,
                 warmup_steps=200,
                 min_lr_fraction=0.1,
                 grad_clip=10.0,
                 eval_rollouts=3):
        # 0.26 (L1-equalized) left gripper error unchanged and drove the arm off target in run 4; revisit once the model sees joint state.
        self.gripper_weight = gripper_weight
        # Action chunking, as in ACT and diffusion policy: predict the next chunk_size actions, run the first execute_steps, then look again.
        self.chunk_size = chunk_size
        self.execute_steps = execute_steps
        # Linear warmup over warmup_steps, then cosine decay to min_lr_fraction of the peak LR by the last epoch.
        self.warmup_steps = warmup_steps
        self.min_lr_fraction = min_lr_fraction
        self.grad_clip = grad_clip  # SmolVLA's value
        self.eval_rollouts = eval_rollouts
        self.learning_rate = learning_rate

        self.max_episode_steps = 400  # longest demo on file is 314; a policy still going at 400 has failed
        self.image_size = frames.VLM_SIZE  # matches the processor, so it skips resizing
        self.native_image_size = 896
        max_buffer_size = 100000

        env = self._make_env(EVAL_TASKS[0], render_mode='rgb_array')
        obs, _ = env.reset()

        self.dataset = Dataset(max_size=max_buffer_size,
                               image_size=self.image_size,
                               n_actions=env.action_space.shape[0],
                               n_joints=9)

        if not eval:
            self.dataset.load_data(path=data_path)

        num_actions = env.action_space.shape[0]

        env.close()

        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        
        self.vlm = VisionLanguageModel(vlm_config=SMOLVLM2_CONFIG_500M,
                                       num_actions=num_actions,
                                       chunk_size=self.chunk_size)
        self.reset_policy()

        self.optimizer = Adam([p for p in self.vlm.model.parameters() if p.requires_grad], self.learning_rate)

    def _make_env(self, task, render_mode):
        env = gym.make("FrankaKitchen-v1", max_episode_steps=self.max_episode_steps,
                       tasks_to_complete=[task], render_mode=render_mode)
        env = HeldSetpointWrapper(env)
        env = VLAObservationWrapper(env, image_size=self.native_image_size)
        return ObsReshapeWrapper(env, image_size=self.image_size)

    def process_observation(self,obs):
        images    = obs['camera_scene']
        joint_pos = obs['joint_pos']
        joint_vel = obs['joint_vel']

        joint_pos = torch.tensor(joint_pos, dtype=torch.float32).to(self.device)
        joint_vel = torch.tensor(joint_vel, dtype=torch.float32).to(self.device)

        # Images stay uint8 HWC numpy for the processor, which moves them to
        # the GPU itself. A single frame from the env gets a batch dim here.
        if images.ndim == 3:
            images    = images[None]
            joint_pos = joint_pos.unsqueeze(0)
            joint_vel = joint_vel.unsqueeze(0)

        return list(images), joint_pos, joint_vel



    def train(self, epochs, batch_size):
        summary_writer_name = f'runs/{git_label()}_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
        summary_writer_name = summary_writer_name + f"_bs={batch_size}"
        summary_writer = SummaryWriter(summary_writer_name)

        def lr_factor(epoch):
            if epoch < self.warmup_steps:
                return (epoch + 1) / self.warmup_steps
            progress = (epoch - self.warmup_steps) / max(1, epochs - self.warmup_steps)
            return self.min_lr_fraction + (1 - self.min_lr_fraction) * 0.5 * (1 + math.cos(math.pi * progress))
        scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_factor)
        trainable = [p for g in self.optimizer.param_groups for p in g["params"]]

        for epoch in range(epochs):
            states, actions, action_mask, tasks = self.dataset.sample_batch(batch_size, self.chunk_size)
           
            images, joint_pos, joint_vel = self.process_observation(states)

            actions = torch.tensor(actions).to(self.device).float()          # (batch, chunk, 9)
            action_mask = torch.tensor(action_mask).to(self.device).float()  # (batch, chunk)

            pred_actions = self.vlm.get_action_logits(images=images,
                                                      prompts=with_joint_state(sample_phrases(tasks), joint_pos))
    
            arm_loss     = masked_mse(pred_actions[..., :7], actions[..., :7], action_mask)
            gripper_loss = masked_mse(pred_actions[..., 7:], actions[..., 7:], action_mask)

            loss = arm_loss + self.gripper_weight * gripper_loss

            self.optimizer.zero_grad()

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, self.grad_clip)

            self.optimizer.step()
            scheduler.step()
            
            if(epoch % 10 == 0):
                summary_writer.add_scalar("train/loss", loss, epoch)
                summary_writer.add_scalar("train/arm", arm_loss, epoch)
                summary_writer.add_scalar("train/gripper", gripper_loss, epoch)
                summary_writer.add_scalar("train/grad_norm", grad_norm, epoch)
                summary_writer.add_scalar("train/lr", scheduler.get_last_lr()[0], epoch)

            if(epoch % 100 == 0):
                print(f"Epoch: {epoch} Loss: {loss.item()}")
                self.vlm.save_the_model()

            if(epoch and epoch % 1000 == 0):
                self.eval(epoch, summary_writer)

        self.vlm.save_the_model()
        self.eval(epochs, summary_writer)

    def eval(self, epoch, summary_writer):
        # Kept per eval so a good point in training can't be overwritten by later saves.
        self.vlm.save_the_model(f"checkpoints/smolvlm2-step{epoch}.pth")
        for task in EVAL_TASKS:
            rate = sum(self.test(task, rollout=i) for i in range(self.eval_rollouts)) / self.eval_rollouts
            summary_writer.add_scalar(f"eval/{task.replace(' ', '_')}", rate, epoch)
            print(f"  eval {task}: {rate:.0%}")

    def reset_policy(self):
        """Forget any queued actions; call at the start of every episode."""
        self.action_queue = []

    def act(self, images, prompt, joint_pos):
        """Next action for one env. The model only runs when the queue is empty, i.e. every execute_steps steps."""
        if not self.action_queue:
            chunk = self.vlm.get_action(images, with_joint_state([prompt], joint_pos))[0]  # (chunk, 9)
            self.action_queue = list(chunk[:self.execute_steps].cpu().numpy())
        return self.action_queue.pop(0)

    def test(self, task, render_mode="rgb_array", delay=0, rollout=0):
        env = self._make_env(task, render_mode)
        prompt = eval_phrase(task, rollout)
        obs, _ = env.reset()
        self.reset_policy()
        done = trunc = False
        total_reward = 0.0

        self.vlm.model.eval()
        while not (done or trunc):
            images, joint_pos, _ = self.process_observation(obs)
            action = self.act(images, prompt, joint_pos)
            obs, reward, done, trunc, _ = env.step(action)
            total_reward += reward
            time.sleep(delay)
        self.vlm.model.train()

        env.close()
        return total_reward > 0
