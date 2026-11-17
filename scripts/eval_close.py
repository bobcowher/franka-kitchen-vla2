"""Close-cabinet eval: the scene starts with a cabinet open and the prompt asks to shut it. Usage: python scripts/eval_close.py --checkpoint PATH [--rollouts N]"""
import argparse
import sys
from pathlib import Path

# Make the project root importable when run as scripts/<name>.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent import STOPS, Agent
from tasks import CLOSE_PHRASES

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--rollouts", type=int, default=40)
args = parser.parse_args()

agent = Agent(eval=True)
agent.vlm.load_the_model(args.checkpoint)

for task, phrases in CLOSE_PHRASES.items():
    cell = [agent.test(task, phrases[i % len(phrases)]) for i in range(args.rollouts)]
    counts = {"reached": sum(reached for reached, _ in cell)}
    counts.update({s: sum(stop == s for _, stop in cell) for s in STOPS})
    print(f"{task}: {counts}", flush=True)
