"""Give the policy commands in a row in one live scene. Usage: python scripts/play.py [--checkpoint PATH] [--slowdown N]"""
import argparse
import select
import sys
from pathlib import Path

# Make the project root importable when run as scripts/<name>.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent import Agent


def wait_for_line(env):
    """Read a line of input, redrawing meanwhile so the window stays responsive. None at end of input."""
    while not select.select([sys.stdin], [], [], 0.05)[0]:
        env.render()
    line = sys.stdin.readline()
    return line.strip() if line else None


parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", default="checkpoints/franka-kitchen-vla2-42-smolvlm2-step5000.pth")
parser.add_argument("--slowdown", type=float, default=2.0, help="2 runs at half speed")
args = parser.parse_args()

agent = Agent(eval=True)
agent.vlm.load_the_model(args.checkpoint)

env = agent.make_session_env()
obs, _ = env.reset()

prompt = None
while True:
    print("\nCommand (Enter repeats the last one, r resets the scene, q quits): ", end="", flush=True)
    typed = wait_for_line(env)
    if typed is None or typed == "q":
        break
    if typed == "r":
        obs, _ = env.reset()
        continue
    prompt = typed or prompt
    if prompt is None:
        continue
    obs, stop_step, _, completed = agent.run_command(env, obs, prompt, slowdown=args.slowdown)
    print(f"done {'fired' if stop_step is not None else 'never fired'}, went home. Complete now: {', '.join(completed) or 'nothing'}")

env.close()
