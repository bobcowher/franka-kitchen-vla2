import sys
from pathlib import Path

# Make the project root importable when run as scripts/<name>.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent import Agent
from tasks import task_from_argv

# Checkpoints from before action chunking (run 23 and earlier) need their own branch; the head shape changed.
CHECKPOINT = "checkpoints/franka-kitchen-vla2-24-smolvlm2-step5000.pth"

task = task_from_argv(sys.argv)

agent = Agent(eval=True)
agent.vlm.load_the_model(CHECKPOINT)

success = agent.test(task, render_mode="human")
print(f"{task}: {'success' if success else 'fail'}")
