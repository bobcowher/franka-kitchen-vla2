"""Train the done head on frozen run 24 weights, then score it closed-loop. Set DONE_SMOKE=1 for a tiny end-to-end check."""
import glob
import json
import os
import sys
import urllib.request
from pathlib import Path

# Make the project root importable when run as scripts/<name>.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from agent import Agent, EVAL_TASKS
from gym_robotics_custom import ArmHomeWrapper
from tasks import TASKS, TASK_DESCRIPTIONS, eval_phrase, sample_phrases

SMOKE = os.environ.get("DONE_SMOKE") == "1"

CHECKPOINT_URL = "http://localhost:5000/api/v1/projects/franka-kitchen-vla2/runs/24/files/checkpoints/smolvlm2-step5000.pth"
OUT_DIR = "checkpoints"
FRAMES_DIR = os.environ.get("DONE_FRAMES", f"{OUT_DIR}/done_frames")
DATA_PATH = "dataset/slide_cabinet" if SMOKE else "dataset"

COLLECT_ROLLOUTS = 1 if SMOKE else 30  # per task
TRAIN_STEPS = 3 if SMOKE else 3000
BATCH_SIZE = 8 if SMOKE else 64
LEARNING_RATE = 3e-4
NEAR_END_STEPS = 16
EVAL_ROLLOUTS = 1 if SMOKE else 20  # per task, per prompt set
ROLLOUT_LIMIT = 40 if SMOKE else 400
OVERRUN_STEPS = 20  # collection only: policy keeps acting after success
HOLD_STEPS = 15
EVAL_POST_STEPS = 60  # eval: how long after success the policy keeps running, giving done time to fire

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


def make_env(agent, task):
    # With both off, the env keeps running after success and reward fires on every frame where the task is complete.
    env = agent._make_env(task, "rgb_array", terminate_on_tasks_completed=False, remove_task_when_completed=False)
    return ArmHomeWrapper(env)


def policy_action(agent, obs, prompt):
    images, joint_pos, _ = agent.process_observation(obs)
    return agent.get_action(images, prompt, joint_pos)


def collect_rollout(agent, task, prompt):
    """One rollout's frames, labelled by the env: run to success, overrun, hold still, then go home."""
    env = make_env(agent, task)
    obs, _ = env.reset()
    agent.reset_policy()
    images, joints, labels = [obs["camera_scene"]], [obs["joint_pos"]], [False]

    def step(action):
        nonlocal obs
        obs, reward, _, _, _ = env.step(action)
        images.append(obs["camera_scene"])
        joints.append(obs["joint_pos"])
        labels.append(reward > 0)

    while len(images) <= ROLLOUT_LIMIT and not labels[-1]:
        step(policy_action(agent, obs, prompt))

    if labels[-1]:
        for _ in range(OVERRUN_STEPS):
            step(policy_action(agent, obs, prompt))
        for _ in range(HOLD_STEPS):
            step(np.zeros(9))
        for _ in range(env.max_steps):
            action = env.home_action()
            if action is None:
                break
            step(action)
    env.close()
    return np.array(images), np.array(joints), np.array(labels)


def collect(agent):
    """Save one npz per rollout under FRAMES_DIR; shards already there count, so a crashed run can resume."""
    os.makedirs(FRAMES_DIR, exist_ok=True)
    for task in EVAL_TASKS:
        for i in range(COLLECT_ROLLOUTS):
            path = f"{FRAMES_DIR}/{task.replace(' ', '_')}_{i:03d}.npz"
            if os.path.exists(path):
                continue
            images, joints, labels = collect_rollout(agent, task, eval_phrase(task, i))
            tmp = f"{path}.tmp"
            with open(tmp, "wb") as f:
                np.savez_compressed(f, images=images, joints=joints, labels=labels, task=task)
            os.replace(tmp, path)
            print(f"collected {task} {i}: {len(labels)} frames, {labels.sum()} done", flush=True)


def load_rollouts_into_dataset(agent):
    """Append rollout frames after the demos, so every frame lives in one arena. Returns (done labels, rollout start row)."""
    data = agent.dataset
    rollout_start = data.mem_ctr
    labels = np.zeros(data.mem_size, dtype=bool)
    labels[:rollout_start] = data.terminal_memory[:rollout_start]  # each demo's last row is its moment of success

    row = rollout_start
    for path in sorted(glob.glob(f"{FRAMES_DIR}/*.npz")):
        shard = np.load(path)
        end = row + len(shard["labels"])
        data.camera_scene_memory[row:end] = shard["images"]
        data.joint_pos_memory[row:end] = shard["joints"]
        data.task_description_memory[row:end] = TASKS[str(shard["task"])]
        labels[row:end] = shard["labels"]
        row = end
    data.mem_ctr = row
    return labels, rollout_start


def wrong_prompts(descriptions):
    """A phrase from some other task: same frame, but this instruction hasn't been done."""
    others = [np.random.choice([d for d in TASK_DESCRIPTIONS if d != own]) for own in descriptions]
    return sample_phrases(others)


def train_done_head(agent, labels, rollout_start):
    data = agent.dataset
    rows = np.arange(data.mem_ctr)
    is_done = labels[rows]
    is_demo = rows < rollout_start
    near_end = is_demo & (data.episode_end_memory[rows] - rows <= NEAR_END_STEPS)

    pools = {
        "done": rows[is_done],
        "wrong_prompt": rows[is_done],
        "demo": rows[is_demo & ~is_done],
        "near_end": rows[near_end & ~is_done],
        "rollout": rows[~is_demo & ~is_done],
    }
    sizes = {name: BATCH_SIZE // 8 for name in pools}
    sizes["done"] = BATCH_SIZE - sum(n for name, n in sizes.items() if name != "done")
    print({name: len(pool) for name, pool in pools.items()}, flush=True)

    optimizer = torch.optim.Adam(agent.vlm.done_head.parameters(), lr=LEARNING_RATE)
    writer = SummaryWriter("runs/done_head")

    for step in range(TRAIN_STEPS):
        batch, prompts, targets, groups = [], [], [], []
        for name, pool in pools.items():
            picked = np.random.choice(pool, sizes[name])
            descriptions = list(data.task_description_memory[picked])
            prompts += wrong_prompts(descriptions) if name == "wrong_prompt" else sample_phrases(descriptions)
            targets += [float(name == "done")] * sizes[name]
            groups += [name] * sizes[name]
            batch.append(picked)
        batch = np.concatenate(batch)

        images = list(data.camera_scene_memory[batch])
        joint_pos = torch.tensor(data.joint_pos_memory[batch], device=agent.device)
        targets = torch.tensor(targets, device=agent.device)

        logits = agent.vlm.get_done_logits(images, prompts, joint_pos)
        loss = F.binary_cross_entropy_with_logits(logits, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0 or SMOKE:
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            means = {name: round(float(probs[np.array(groups) == name].mean()), 3) for name in pools}
            print(f"step {step} loss {loss.item():.4f} mean prob {means}", flush=True)
            writer.add_scalar("done/loss", loss, step)
            for name, value in means.items():
                writer.add_scalar(f"done/prob_{name}", value, step)

    agent.vlm.save_done_head(f"{OUT_DIR}/done_head.pth")


def eval_rollout(agent, task, prompt):
    """Run the policy past success without stopping; log done at each replan so thresholds can be compared afterward."""
    env = make_env(agent, task)
    obs, _ = env.reset()
    agent.reset_policy()
    trace = []
    first_success = None
    complete = False

    for step in range(ROLLOUT_LIMIT + EVAL_POST_STEPS):
        if first_success is None and step >= ROLLOUT_LIMIT:
            break
        if first_success is not None and step >= first_success + EVAL_POST_STEPS:
            break
        replanned = not agent.action_queue
        action = policy_action(agent, obs, prompt)
        if replanned:
            trace.append((step, round(agent.done_prob, 4), bool(complete)))
        obs, reward, _, _, _ = env.step(action)
        complete = reward > 0
        if complete and first_success is None:
            first_success = step + 1

    still_complete_at_end = bool(complete)
    _, home_steps = env.return_to_home()
    _, reward, _, _, _ = env.step(np.zeros(9))
    env.close()
    return {"task": task, "prompt": prompt, "first_success": first_success, "trace": trace,
            "complete_before_home": still_complete_at_end, "home_steps": home_steps,
            "complete_after_home": bool(reward > 0)}


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


def summarize(rows):
    for prompt_set in sorted({r["set"] for r in rows}):
        subset = [r for r in rows if r["set"] == prompt_set]
        print(f"\n== {prompt_set}: {len(subset)} rollouts")
        for task in EVAL_TASKS:
            task_rows = [r for r in subset if r["task"] == task]
            succeeded = [r for r in task_rows if r["first_success"] is not None]
            print(f"{task}: success {len(succeeded)}/{len(task_rows)}, "
                  f"still complete after home {sum(r['complete_after_home'] for r in succeeded)}/{len(succeeded)}, "
                  f"max home steps {max((r['home_steps'] for r in succeeded), default=0)}")
        for threshold in (0.5, 0.7, 0.9):
            for patience in (1, 2, 3):
                results = [score(r, threshold, patience) for r in subset]
                counts = {k: sum(o == k for o, _ in results) for k in ("correct", "false_stop", "miss", "quiet_fail")}
                lags = [lag for _, lag in results if lag is not None]
                median_lag = int(np.median(lags)) if lags else None
                print(f"threshold {threshold} patience {patience}: {counts} median lag {median_lag} steps")


def run_eval(agent):
    sets = {"orig": {t: [TASKS[t]] for t in EVAL_TASKS}, "heldout": HELDOUT_PHRASES}
    rows = []
    for name, phrases in sets.items():
        for task in EVAL_TASKS:
            for i in range(EVAL_ROLLOUTS):
                row = eval_rollout(agent, task, phrases[task][i % len(phrases[task])])
                row["set"] = name
                rows.append(row)
                with open(f"{OUT_DIR}/done_eval.jsonl", "a") as f:
                    f.write(json.dumps(row) + "\n")
                print(f"eval {name} {task} {i}: first success {row['first_success']}", flush=True)
    summarize(rows)


def check_feature_matches_action_input(agent):
    """The done feature should be exactly what the action head reads."""
    image = [agent.dataset.camera_scene_memory[0]]
    joint_pos = torch.tensor(agent.dataset.joint_pos_memory[:1], device=agent.device)
    with torch.no_grad():
        outputs = agent.vlm._forward(image, ["Open the microwave door"], joint_pos)
        feature = outputs.hidden_states[-1][:, -1, :]
        difference = (agent.vlm.model.lm_head(feature).float() - outputs.logits[:, -1, :].float()).abs().max().item()
    print(f"feature check: max difference {difference:.6f}", flush=True)
    assert difference < 1e-2, "hidden_states[-1] is not the action head's input"


def main():
    agent = Agent(data_path=DATA_PATH, name="done_head")
    # Eval rollouts run past the normal limit on purpose, and homing needs room too.
    agent.max_episode_steps = ROLLOUT_LIMIT + EVAL_POST_STEPS + 300

    weights = f"{OUT_DIR}/run24-step5000.pth"
    os.makedirs(OUT_DIR, exist_ok=True)
    if not os.path.exists(weights):
        urllib.request.urlretrieve(CHECKPOINT_URL, weights)
    agent.vlm.load_the_model(weights)
    agent.vlm.model.eval()

    check_feature_matches_action_input(agent)
    collect(agent)
    labels, rollout_start = load_rollouts_into_dataset(agent)
    train_done_head(agent, labels, rollout_start)
    agent.vlm.model.eval()
    run_eval(agent)


if __name__ == "__main__":
    main()
