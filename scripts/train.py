import sys
from pathlib import Path

# Make the project root importable when run as scripts/<name>.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent import Agent
from stop_eval import StopEval

agent = Agent()

agent.train(epochs=5000, batch_size=64)

StopEval(agent).run()
