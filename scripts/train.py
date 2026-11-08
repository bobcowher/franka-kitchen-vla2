import sys
from pathlib import Path

# Make the project root importable when run as scripts/<name>.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from PIL import Image, ImageDraw
from config import SMOLVLM2_CONFIG_500M
from agent import Agent

agent = Agent(data_path="dataset", name="bc_network")

agent.train(epochs=5000, batch_size=64)

# The same stop eval as run 26, so the two compare directly.
from train_done import EVAL_POST_STEPS, ROLLOUT_LIMIT, run_eval

agent.vlm.model.eval()
agent.max_episode_steps = ROLLOUT_LIMIT + EVAL_POST_STEPS + 300
run_eval(agent)
