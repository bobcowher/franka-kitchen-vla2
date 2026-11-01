import os

# MuJoCo picks its GL backend at import time and defaults to GLFW, which needs
# an X display. Headless training servers have none, so fall back to EGL there.
# setdefault, so the environment can still override.
if not os.environ.get("DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "egl")

import torch
import torch.nn.functional as F
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
from tasks import TASKS

from config import SMOLVLM2_CONFIG_500M

# L1 scales with mean absolute deviation, 3.8x higher on the gripper than an arm joint (variance is 7.8x, which suited MSE).
GRIPPER_WEIGHT = 0.26

EVAL_TASKS = ["microwave", "hinge cabinet", "top burner"]
EVAL_ROLLOUTS = 3

class Agent:

    def __init__(self, eval=False, data_path="dataset", name='bc_network'):
        self.max_episode_steps = 400  # longest demo on file is 314; a policy still going at 400 has failed
        self.image_size = frames.VLM_SIZE  # matches the processor, so it skips resizing
        self.native_image_size = 896
        max_buffer_size = 100000
        learning_rate = 0.0001

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
                                       num_actions=num_actions)

        self.optimizer = Adam([p for p in self.vlm.model.parameters() if p.requires_grad], learning_rate)

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
        branch = subprocess.run(["git", "branch", "--show-current"],
                                capture_output=True, text=True).stdout.strip() or "detached"
        summary_writer_name = f'runs/{branch}_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
        summary_writer_name = summary_writer_name + f"_bs={batch_size}"
        summary_writer = SummaryWriter(summary_writer_name)

        for epoch in range(epochs):
            states, actions, _, _, tasks = self.dataset.sample_batch(batch_size)
           
            images, joint_pos, joint_vel = self.process_observation(states)

            actions = torch.tensor(actions).to(self.device).float()

            pred_actions = self.vlm.get_action_logits(images=images,
                                                      prompts=tasks)
    
            arm_loss     = F.l1_loss(actions[:, :7], pred_actions[:, :7])
            gripper_loss = F.l1_loss(actions[:, 7:], pred_actions[:, 7:])

            loss = arm_loss + GRIPPER_WEIGHT * gripper_loss

            self.optimizer.zero_grad()

            loss.backward()

            self.optimizer.step()
            
            if(epoch % 10 == 0):
                summary_writer.add_scalar("train/loss", loss, epoch)
                summary_writer.add_scalar("train/arm", arm_loss, epoch)
                summary_writer.add_scalar("train/gripper", gripper_loss, epoch)

            if(epoch % 100 == 0):
                print(f"Epoch: {epoch} Loss: {loss.item()}")
                self.vlm.save_the_model()

            if(epoch and epoch % 1000 == 0):
                self.eval(epoch, summary_writer)

    def eval(self, epoch, summary_writer):
        for task in EVAL_TASKS:
            rate = sum(self.test(task) for _ in range(EVAL_ROLLOUTS)) / EVAL_ROLLOUTS
            summary_writer.add_scalar(f"eval/{task.replace(' ', '_')}", rate, epoch)
            print(f"  eval {task}: {rate:.0%}")

    def test(self, task, render_mode="rgb_array", delay=0):
        env = self._make_env(task, render_mode)
        prompt = TASKS[task]
        obs, _ = env.reset()
        done = trunc = False
        total_reward = 0.0

        self.vlm.model.eval()
        while not (done or trunc):
            images, _, _ = self.process_observation(obs)
            action = self.vlm.get_action(images, [prompt])
            obs, reward, done, trunc, _ = env.step(action.cpu().numpy().squeeze())
            total_reward += reward
            time.sleep(delay)
        self.vlm.model.train()

        env.close()
        return total_reward > 0
