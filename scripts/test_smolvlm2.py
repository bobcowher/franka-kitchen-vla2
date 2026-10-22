import sys
from pathlib import Path

# Make the project root importable when run as scripts/<name>.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image, ImageDraw
from config import SMOLVLM2_CONFIG_500M
from vlm import VisionLanguageModel

vlm = VisionLanguageModel(vlm_config=SMOLVLM2_CONFIG_500M)

n_params = sum(p.numel() for p in vlm.model.parameters())
print(f"Loaded {SMOLVLM2_CONFIG_500M['model_id']} ({n_params/1e6:.0f}M params) on {vlm.device}")

# Synthetic image with a known answer
image = Image.new("RGB", (384, 384), "white")
ImageDraw.Draw(image).rectangle([96, 96, 288, 288], fill="red")

print(vlm.generate(image, "What color is the square in this image?"))

vlm.configure_model(num_actions=7)

print(vlm.get_action(image, "What color is the square in this image?"))


