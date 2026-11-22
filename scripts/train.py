import glob
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

# Make the project root importable when run as scripts/<name>.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent import Agent

# Experiment branch only: run 103's 512 config trained on DEMO_FRACTION of each task's demos.
DATA = "/data/datasets/franka-kitchen-v2"
task_dirs = os.environ["TASK_DIRS"].split(":")
fraction = float(os.environ["DEMO_FRACTION"])

# Fixed-seed permutation per task, so each smaller fraction is a subset of every larger one.
subset_root = tempfile.mkdtemp(prefix="demo_subset_")
for task_dir in task_dirs:
    shards = sorted(glob.glob(f"{DATA}/{task_dir}/**/*.npz", recursive=True))
    keep = max(1, round(len(shards) * fraction))
    order = np.random.default_rng(0).permutation(len(shards))[:keep]
    os.makedirs(f"{subset_root}/{task_dir}")
    for i in order:
        os.symlink(shards[i], f"{subset_root}/{task_dir}/{i:05d}.npz")
    print(f"{task_dir}: {keep}/{len(shards)} demos", flush=True)

agent = Agent(data_path=f"{subset_root}/{task_dirs[0]}", max_buffer_size=30000)
for task_dir in task_dirs[1:]:
    agent.dataset.load_data(path=f"{subset_root}/{task_dir}")

agent.train(epochs=5000, batch_size=64)
