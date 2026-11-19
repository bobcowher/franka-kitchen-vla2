import os
import sys
from pathlib import Path

# Make the project root importable when run as scripts/<name>.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent import Agent

# Experiment branch only: TASK_DIRS (colon-separated) picks which task folders of lab's v2 to train on.
DATA = "/data/datasets/franka-kitchen-v2"
task_dirs = os.environ["TASK_DIRS"].split(":")
print(f"Training on {task_dirs}", flush=True)

agent = Agent(data_path=f"{DATA}/{task_dirs[0]}")
for task_dir in task_dirs[1:]:
    agent.dataset.load_data(path=f"{DATA}/{task_dir}")

agent.train(epochs=8000, batch_size=64)
