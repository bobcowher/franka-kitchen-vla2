import sys
from pathlib import Path

# Make the project root importable when run as scripts/<name>.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent import Agent
from tasks import task_from_argv

# Trained on 512 frames; the 448 checkpoint needs the first-training-run branch.
CHECKPOINT = "checkpoints/smolvlm2-512-step4900.pth"

task = task_from_argv(sys.argv)

agent = Agent(eval=True)
agent.vlm.load_the_model(CHECKPOINT)

success = agent.test(task, render_mode="human")
print(f"{task}: {'success' if success else 'fail'}")
