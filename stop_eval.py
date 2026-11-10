"""Closed-loop eval of the policy and the done head's stop rule, on original and held-out wording."""
import json

import numpy as np

from agent import EVAL_TASKS
from gym_robotics_custom import ArmHomeWrapper
from tasks import TASKS

HELDOUT_PHRASES = {
    "microwave": ["Get the microwave open", "The microwave door should end up open",
                  "Could you open the microwave for me?", "microwave door: open"],
    "hinge cabinet": ["Get the cabinet that is second from the left open",
                      "The cabinet second from the left should end up open",
                      "Could you open the cabinet next to the leftmost one?",
                      "cabinet, second from left: open"],
    "top burner": ["Get the top left burner going", "The knob for the top left burner should end up turned",
                   "Could you turn on the upper left stove burner?", "top left burner knob: turn"],
}


class StopEval:

    def __init__(self, agent, rollouts=20, rollout_limit=400, post_steps=60, home_room=300,
                 thresholds=(0.5, 0.7, 0.9), patiences=(1, 2, 3), out_path="checkpoints/done_eval.jsonl"):
        self.agent = agent
        self.rollouts = rollouts  # per task, per prompt set
        self.rollout_limit = rollout_limit
        # How long the policy keeps running after success, giving done time to fire.
        self.post_steps = post_steps
        self.max_episode_steps = rollout_limit + post_steps + home_room  # homing steps count against the limit
        self.thresholds = thresholds
        self.patiences = patiences
        self.out_path = out_path

    def run(self):
        self.agent.vlm.model.eval()
        sets = {"orig": {t: [TASKS[t]] for t in EVAL_TASKS}, "heldout": HELDOUT_PHRASES}
        rows = []
        for name, phrases in sets.items():
            for task in EVAL_TASKS:
                for i in range(self.rollouts):
                    row = self.rollout(task, phrases[task][i % len(phrases[task])])
                    row["set"] = name
                    rows.append(row)
                    with open(self.out_path, "a") as f:
                        f.write(json.dumps(row) + "\n")
                    print(f"eval {name} {task} {i}: first success {row['first_success']}", flush=True)
        self.summarize(rows)
        return rows

    def rollout(self, task, prompt):
        """Run the policy past success without stopping; log done at each replan so stop rules can be scored afterward."""
        agent = self.agent
        # With both off, the env keeps running after success and reward fires on every frame where the task is complete.
        env = ArmHomeWrapper(agent._make_env(task, "rgb_array", max_episode_steps=self.max_episode_steps,
                                             terminate_on_tasks_completed=False, remove_task_when_completed=False))
        obs, _ = env.reset()
        agent.reset_policy()
        trace = []
        first_success = None
        complete = False

        for step in range(self.rollout_limit + self.post_steps):
            if first_success is None and step >= self.rollout_limit:
                break
            if first_success is not None and step >= first_success + self.post_steps:
                break
            replanned = not agent.action_queue
            images, joint_pos, _ = agent.process_observation(obs)
            action = agent.get_action(images, prompt, joint_pos)
            if replanned:
                trace.append((step, round(agent.done_prob, 4), bool(complete)))
            obs, reward, _, _, _ = env.step(action)
            complete = reward > 0
            if complete and first_success is None:
                first_success = step + 1

        complete_before_home = bool(complete)
        _, home_steps = env.return_to_home()
        _, reward, _, _, _ = env.step(np.zeros(9))
        env.close()
        return {"task": task, "prompt": prompt, "first_success": first_success, "trace": trace,
                "complete_before_home": complete_before_home, "home_steps": home_steps,
                "complete_after_home": bool(reward > 0)}

    @staticmethod
    def score(row, threshold, patience):
        """What a stop rule would have done: correct, false_stop, miss, or quiet_fail. Returns (outcome, lag from first success)."""
        streak = 0
        for step, prob, complete in row["trace"]:
            streak = streak + 1 if prob >= threshold else 0
            if streak >= patience:
                if complete:
                    return "correct", step - row["first_success"]
                return "false_stop", None
        return ("miss" if row["first_success"] is not None else "quiet_fail"), None

    def summarize(self, rows):
        for prompt_set in sorted({r["set"] for r in rows}):
            subset = [r for r in rows if r["set"] == prompt_set]
            print(f"\n== {prompt_set}: {len(subset)} rollouts")
            for task in EVAL_TASKS:
                task_rows = [r for r in subset if r["task"] == task]
                succeeded = [r for r in task_rows if r["first_success"] is not None]
                print(f"{task}: success {len(succeeded)}/{len(task_rows)}, "
                      f"still complete after home {sum(r['complete_after_home'] for r in succeeded)}/{len(succeeded)}, "
                      f"max home steps {max((r['home_steps'] for r in succeeded), default=0)}")
            for threshold in self.thresholds:
                for patience in self.patiences:
                    results = [self.score(r, threshold, patience) for r in subset]
                    counts = {k: sum(o == k for o, _ in results) for k in ("correct", "false_stop", "miss", "quiet_fail")}
                    lags = [lag for _, lag in results if lag is not None]
                    median_lag = int(np.median(lags)) if lags else None
                    print(f"threshold {threshold} patience {patience}: {counts} median lag {median_lag} steps")
