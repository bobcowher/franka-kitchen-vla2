import sys
from pathlib import Path

# Make the project root importable when run as scripts/<name>.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from huggingface_hub import snapshot_download
from config import SMOLVLM2_CONFIG_500M

# Skip the onnx/ exports; we only need the PyTorch weights
path = snapshot_download(
        SMOLVLM2_CONFIG_500M["model_id"],
        allow_patterns=["*.json", "*.txt", "*.safetensors"]
        )
print(f"SmolVLM2 downloaded to {path}")
