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
