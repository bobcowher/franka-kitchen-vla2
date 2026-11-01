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
vlm = agent.vlm

n_params = sum(p.numel() for p in vlm.model.parameters())
print(f"Loaded {SMOLVLM2_CONFIG_500M['model_id']} ({n_params/1e6:.0f}M params) on {vlm.device}")

image = Image.new("RGB", (384, 384), "white")
ImageDraw.Draw(image).rectangle([96, 96, 288, 288], fill="red")

# Batch test: two real dataset frames plus two synthetic ones, all uint8 HWC
# like process_observation hands out. Prompt lengths differ so padding is
# actually exercised.
states, _, _, _, tasks = agent.dataset.sample_batch(2)
images, _, _ = agent.process_observation(states)

size = agent.image_size
green = Image.new("RGB", (size, size), "white")
ImageDraw.Draw(green).rectangle([size // 4, size // 4, 3 * size // 4, 3 * size // 4], fill="green")
images += [np.array(image.resize((size, size))), np.array(green)]
tasks  += ["What color is the square in this image?", "Square?"]

batched = vlm.get_action(images, tasks)
print(batched)

# 1. One row of 9 actions per sample.
assert batched.shape == (len(images), 9), f"expected {(len(images), 9)}, got {tuple(batched.shape)}"

# 2. Clamped into the normalized action range.
assert batched.min() >= -1 and batched.max() <= 1, "get_action output escaped [-1, 1]"

# 3. Batch invariance: each sample alone must match its row in the batch. Right
#    padding, or reading the wrong position, shows up here as a large diff on
#    the shorter prompts. bf16 noise alone stays well under the tolerance.
for i, (img, task) in enumerate(zip(images, tasks)):
    alone = vlm.get_action([img], [task])
    diff = (alone[0].float() - batched[i].float()).abs().max().item()
    print(f"sample {i}: max |alone - batched| = {diff:.4f}  ({task!r})")
    assert diff < 0.05, f"sample {i} changes when batched"

print("Batch test passed")
