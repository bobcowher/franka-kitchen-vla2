"""Type prompts and watch the policy try them. Usage: python scripts/play.py <task> [--checkpoint PATH]"""
import argparse
import sys
from pathlib import Path

# Make the project root importable when run as scripts/<name>.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent import Agent
from tasks import TASKS

parser = argparse.ArgumentParser()
# The task only decides which goal the env scores as success; the prompt is whatever you type.
parser.add_argument("task", choices=[t.replace(" ", "_") for t in TASKS])
parser.add_argument("--checkpoint", default="checkpoints/franka-kitchen-vla2-24-smolvlm2-step5000.pth")
parser.add_argument("--done-head", default="checkpoints/franka-kitchen-vla2-26-done_head.pth")
args = parser.parse_args()
task = args.task.replace("_", " ")

agent = Agent(eval=True)
agent.vlm.load_the_model(args.checkpoint)
agent.vlm.load_done_head(args.done_head)

prompt = TASKS[task]
while True:
    typed = input(f'\nPrompt (Enter repeats "{prompt}", q quits): ').strip()
    if typed == "q":
        break
    prompt = typed or prompt
    fired, complete = agent.run_until_done(task, prompt)
    print(f"done {'fired, went home' if fired else 'never fired'}; {task} {'still complete' if complete else 'not complete'}")
