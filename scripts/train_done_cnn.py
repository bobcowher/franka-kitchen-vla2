"""Experiment: train DoneCNN on frozen VLM features, then evaluate the policy with it making the stop call.

Env: TASK_DIRS (colon-separated v2 task folders), BACKBONE_URL (a policy checkpoint to fetch).
"""
import os
import sys
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import frames
from agent import Agent
from done_cnn import DoneCNN
from tasks import sample_phrases

DATA = "/data/datasets/franka-kitchen-v2"
STEPS = 3000
BATCH = 32
FEATURE_BATCH = 64
VAL_EVERY = 500


def compute_features(agent):
    """The VLM's done-head features for every stored frame, at VLM_SIZE with a sampled phrase."""
    ds = agent.dataset
    feats = torch.zeros(ds.mem_ctr, agent.vlm.config["emb_dim"], dtype=torch.float16)
    agent.vlm.model.eval()
    for start in range(0, ds.mem_ctr, FEATURE_BATCH):
        rows = slice(start, min(start + FEATURE_BATCH, ds.mem_ctr))
        images = list(frames.resize(ds.camera_scene_memory[rows], frames.VLM_SIZE))
        joint_pos = torch.tensor(ds.joint_pos_memory[rows], device=agent.device)
        with torch.no_grad():
            _, _, f = agent.vlm.predict(images, sample_phrases(list(ds.task_description_memory[rows])), joint_pos)
        feats[rows] = f.cpu().half()
    return feats


def batch_loss(cnn, agent, feats, rows):
    """Pos-weighted BCE on the done labels, matching Agent._done_loss."""
    images = torch.from_numpy(agent.dataset.camera_scene_memory[rows]).to(agent.device)
    targets = torch.tensor(agent.dataset.terminal_memory[rows], dtype=torch.float32, device=agent.device)
    logits = cnn(images, feats[rows].to(agent.device))
    positives = targets.sum()
    loss = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=(len(targets) - positives) / positives.clamp(min=1))
    return loss, logits


def validate(cnn, agent, feats, val_rows, episode_ends):
    """Frame accuracy on held-out episodes, and how far the first fire sits from the first done label."""
    cnn.eval()
    probs = np.zeros(len(val_rows))
    with torch.no_grad():
        for i in range(0, len(val_rows), 64):
            _, logits = batch_loss(cnn, agent, feats, val_rows[i:i + 64])
            probs[i:i + 64] = torch.sigmoid(logits).cpu().numpy()
    cnn.train()
    labels = agent.dataset.terminal_memory[val_rows]
    fired = probs >= 0.5
    offsets = []
    for end in np.unique(episode_ends):
        ep = episode_ends == end
        if labels[ep].any() and fired[ep].any():
            offsets.append(np.argmax(fired[ep]) - np.argmax(labels[ep]))
    offsets = np.array(offsets)
    print(f"  val: done recall {fired[labels].mean():.3f}, false fire {fired[~labels].mean():.3f}, "
          f"first-fire offset median {np.median(offsets):.0f}, fired >10 frames early in {(offsets < -10).sum()}/{len(offsets)}", flush=True)


task_dirs = os.environ["TASK_DIRS"].split(":")
agent = Agent(data_path=f"{DATA}/{task_dirs[0]}", image_size=frames.RENDER_SIZE, max_buffer_size=30000)
for task_dir in task_dirs[1:]:
    agent.dataset.load_data(path=f"{DATA}/{task_dir}")
os.makedirs("checkpoints", exist_ok=True)
urllib.request.urlretrieve(os.environ["BACKBONE_URL"], "checkpoints/backbone.pth")
agent.vlm.load_the_model("checkpoints/backbone.pth")

feats = compute_features(agent)
print(f"Features for {len(feats)} frames", flush=True)

n = agent.dataset.mem_ctr
episode_end = agent.dataset.episode_end_memory[:n]
val = np.isin(episode_end, np.unique(episode_end)[::10])  # every tenth episode held out
train_rows, val_rows = np.flatnonzero(~val), np.flatnonzero(val)
print(f"{len(train_rows)} train frames, {len(val_rows)} val frames", flush=True)

cnn = DoneCNN(feature_dim=agent.vlm.config["emb_dim"]).to(agent.device)
optimizer = torch.optim.Adam(cnn.parameters(), lr=1e-4)
writer = SummaryWriter("runs/done_cnn")
for step in range(STEPS + 1):
    loss, _ = batch_loss(cnn, agent, feats, np.sort(np.random.choice(train_rows, BATCH)))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % 100 == 0:
        writer.add_scalar("loss/done_cnn", loss.item(), step)
        print(f"Step {step} done_cnn loss {loss.item():.4f}", flush=True)
    if step % VAL_EVERY == 0:
        validate(cnn, agent, feats, val_rows, episode_end[val_rows])
torch.save(cnn.state_dict(), "checkpoints/done_cnn.pth")

del feats
agent.dataset = None  # frees ~52 GB of frames before the rollouts
cnn.eval()
agent.done_model = cnn
agent.eval(0, writer, rollouts=20)
