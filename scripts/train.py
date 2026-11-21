import os
import sys
from pathlib import Path

# Make the project root importable when run as scripts/<name>.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent import Agent

# Experiment branch only: TASK_DIRS (colon-separated) picks lab's v2 task folders; DONE_GRAD_SCALE sets done_grad_scale.
DATA = "/data/datasets/franka-kitchen-v2"
task_dirs = os.environ["TASK_DIRS"].split(":")
done_grad_scale = float(os.environ["DONE_GRAD_SCALE"])
print(f"Training on {task_dirs}, done_grad_scale={done_grad_scale}", flush=True)

agent = Agent(data_path=f"{DATA}/{task_dirs[0]}", done_grad_scale=done_grad_scale)
for task_dir in task_dirs[1:]:
    agent.dataset.load_data(path=f"{DATA}/{task_dir}")

agent.train(epochs=5000, batch_size=64)
