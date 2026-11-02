"""Re-evaluate one checkpoint with many rollouts. Run from a worktree of the commit that trained it."""
import argparse
import json
import os
import sys
import time

os.environ["MUJOCO_GL"] = "egl"
sys.path.insert(0, os.getcwd())

from agent import Agent

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--label", required=True)
parser.add_argument("--tasks", required=True, help="task:rollouts,task:rollouts")
parser.add_argument("--out", required=True)
args = parser.parse_args()

# Appending per rollout means a killed sweep resumes without repeating finished rollouts.
done = {}
if os.path.exists(args.out):
    with open(args.out) as f:
        for line in f:
            row = json.loads(line)
            key = (row["label"], row["task"])
            done[key] = done.get(key, 0) + 1

plan = [(t.split(":")[0], int(t.split(":")[1])) for t in args.tasks.split(",")]
if all(done.get((args.label, task), 0) >= n for task, n in plan):
    sys.exit(0)

agent = Agent(eval=True)
agent.vlm.load_the_model(args.checkpoint)

for task, n in plan:
    for _ in range(n - done.get((args.label, task), 0)):
        start = time.time()
        success = bool(agent.test(task))
        with open(args.out, "a") as f:
            f.write(json.dumps({"label": args.label, "task": task, "success": success,
                                "seconds": round(time.time() - start, 1)}) + "\n")
