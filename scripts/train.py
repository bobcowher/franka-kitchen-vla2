import os
import sys
from pathlib import Path

# Make the project root importable when run as scripts/<name>.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import frames
from agent import Agent

# Experiment branch only: TASK_DIRS (colon-separated) picks lab's v2 task folders; IMAGE_SIZE 896 tiles the frame, 512 is the control.
DATA = "/data/datasets/franka-kitchen-v2"
task_dirs = os.environ["TASK_DIRS"].split(":")
image_size = int(os.environ["IMAGE_SIZE"])
hires = image_size > frames.VLM_SIZE
print(f"Training on {task_dirs}, image_size={image_size}, splitting={hires}", flush=True)

# 4 x 16 keeps the effective batch at 64 (896 at bs 64 needs ~40 GB); 30000 holds the 21810 open-task steps.
agent = Agent(data_path=f"{DATA}/{task_dirs[0]}", image_size=image_size, image_splitting=hires,
              grad_accum_steps=4 if hires else 1, max_buffer_size=30000)
for task_dir in task_dirs[1:]:
    agent.dataset.load_data(path=f"{DATA}/{task_dir}")

agent.train(epochs=5000, batch_size=64)
