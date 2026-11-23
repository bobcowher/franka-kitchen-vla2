import sys
from pathlib import Path

# Make the project root importable when run as scripts/<name>.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent import Agent

# 2 camera-jittered copies of every demo; 3x the rows needs ~2x the steps for the done head to catch up.
agent = Agent(camera_jitter_copies=2)

agent.train(epochs=10000, batch_size=64)

