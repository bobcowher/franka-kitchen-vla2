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

# Synthetic image with a known answer
image = Image.new("RGB", (384, 384), "white")
ImageDraw.Draw(image).rectangle([96, 96, 288, 288], fill="red")

print(vlm.generate(image, "What color is the square in this image?"))

vlm.configure_model(num_actions=9)

# Batch test: two real dataset frames plus two synthetic ones, all uint8 HWC
# like process_observation hands out. Prompt lengths differ so padding is
# actually exercised.
states, _, _, _, tasks = agent.dataset.sample_batch(2)
images, _, _ = agent.process_observation(states)

green = Image.new("RGB", (224, 224), "white")
ImageDraw.Draw(green).rectangle([56, 56, 168, 168], fill="green")
images += [np.array(image.resize((224, 224))), np.array(green)]
tasks  += ["What color is the square in this image?", "Square?"]

batched = vlm.get_action(images, tasks)
print(batched)

# 1. One row of 9 actions per sample.
assert batched.shape == (len(images), 9), f"expected {(len(images), 9)}, got {tuple(batched.shape)}"

# 2. Clamped into the normalized action range.
assert batched.min() >= -1 and batched.max() <= 1, "get_action output escaped [-1, 1]"

# 3. Batch invariance: each sample alone must match its row in the batch.
#    Compared before the clamp, which would hide a mismatch on saturated dims.
#    Measured against a random head: left padding differs by <= 0.055 (bf16
#    noise), right padding by > 2 on every padded sample. The longest prompt
#    gets no padding, so it passes either way.
with torch.no_grad():
    batched_logits = vlm.get_action_logits(images, tasks)
    for i, (img, task) in enumerate(zip(images, tasks)):
        alone = vlm.get_action_logits([img], [task])
        diff = (alone[0].float() - batched_logits[i].float()).abs().max().item()
        print(f"sample {i}: max |alone - batched| = {diff:.4f}  ({task!r})")
        assert diff < 0.25, f"sample {i} changes when batched"

print("Batch test passed")
