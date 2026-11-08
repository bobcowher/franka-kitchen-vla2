"""Run the policy until the done rule fires, then try each way of stopping and see whether the task stays complete."""
import json
import os
import sys
import urllib.request
from pathlib import Path

# Make the project root importable when run as scripts/<name>.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from gymnasium_robotics.envs.franka_kitchen.kitchen_env import OBS_ELEMENT_INDICES

from agent import Agent, EVAL_TASKS
from gym_robotics_custom import ArmHomeWrapper
from tasks import TASKS


def goal_distance(env, task):
    kitchen = env.unwrapped
    return float(np.linalg.norm(kitchen.data.qpos[OBS_ELEMENT_INDICES[task]] - kitchen.goal[task]))


def fetch(run, name, path):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        url = f"http://localhost:5000/api/v1/projects/franka-kitchen-vla2/runs/{run}/files/checkpoints/{name}"
        urllib.request.urlretrieve(url, path)


def run_to_done(agent, env, task, threshold, patience, step_limit):
    """Returns (observation, fired step or None, first success step or None)."""
    obs, _ = env.reset()
    agent.reset_policy()
    streak = 0
    first_success = None
    for step in range(step_limit):
        replanned = not agent.action_queue
        images, joint_pos, _ = agent.process_observation(obs)
        action = agent.get_action(images, TASKS[task], joint_pos)
        if replanned:
            streak = streak + 1 if agent.done_prob >= threshold else 0
            if streak >= patience:
                return obs, step, first_success
        obs, reward, _, _, _ = env.step(action)
        if reward > 0 and first_success is None:
            first_success = step + 1
    return obs, None, first_success


def hold(env, steps=30):
    for _ in range(steps):
        env.step(np.zeros(9))
    return steps


def go_home(env):
    return env.return_to_home()[1]


def open_then_home(env, open_steps=10):
    # Fingers go to their home width while the arm stays put, then the normal home move runs.
    for _ in range(open_steps):
        action = env.home_action()
        if action is None:
            break
        action = action.copy()
        action[:7] = 0.0
        env.step(action)
    return open_steps + env.return_to_home()[1]


def main(rollouts=20, threshold=0.5, patience=2, step_limit=400):
    fetch(26, "run24-step5000.pth", "checkpoints/run24-step5000.pth")
    fetch(26, "done_head.pth", "checkpoints/done_head.pth")

    agent = Agent(eval=True)
    agent.max_episode_steps = step_limit + 300
    agent.vlm.load_the_model("checkpoints/run24-step5000.pth")
    agent.vlm.load_done_head("checkpoints/done_head.pth")
    agent.vlm.model.eval()

    for name, stop in [("hold", hold), ("home", go_home), ("open_then_home", open_then_home)]:
        for task in EVAL_TASKS:
            rows = []
            for _ in range(rollouts):
                env = ArmHomeWrapper(agent._make_env(task, "rgb_array", terminate_on_tasks_completed=False,
                                                     remove_task_when_completed=False))
                _, fired, first_success = run_to_done(agent, env, task, threshold, patience, step_limit)
                at_fire = goal_distance(env, task)
                steps = stop(env) if fired is not None else None
                _, reward, _, _, _ = env.step(np.zeros(9))
                rows.append({"stop": name, "task": task, "fired": fired, "first_success": first_success,
                             "distance_at_fire": round(at_fire, 3), "distance_after": round(goal_distance(env, task), 3),
                             "stop_steps": steps, "complete_after": bool(reward > 0)})
                env.close()
                with open("checkpoints/home_diag.jsonl", "a") as f:
                    f.write(json.dumps(rows[-1]) + "\n")
            stopped = [r for r in rows if r["fired"] is not None]
            before = np.median([r["distance_at_fire"] for r in stopped]) if stopped else None
            after = np.median([r["distance_after"] for r in stopped]) if stopped else None
            print(f"{name} {task}: fired {len(stopped)}/{rollouts}, complete after "
                  f"{sum(r['complete_after'] for r in stopped)}/{len(stopped)}, median goal distance {before} -> {after} "
                  f"(success below 0.3)", flush=True)


if __name__ == "__main__":
    main()
