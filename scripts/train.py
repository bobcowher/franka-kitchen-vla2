import os
import sys
from pathlib import Path

# Make the project root importable when run as scripts/<name>.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent import Agent
from stop_eval import StopEval

# DATA_PATH lets a queued run choose its data without changing the project's data folder.
agent = Agent(data_path=os.environ.get("DATA_PATH", "dataset"), name="bc_network")

agent.train(epochs=5000, batch_size=64)

StopEval(agent).run()
