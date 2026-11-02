"""Overnight re-evaluation of saved checkpoints, each under the commit that trained it. Resumes from eval_results/."""
import os
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from huggingface_hub import snapshot_download

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from config import SMOLVLM2_CONFIG_500M

API = "http://localhost:5000/api/v1/projects/franka-kitchen-vla2/runs"
WORKERS = 4
STEPS = (1000, 2000, 3000, 4000, 5000)
FULL = "microwave:50,hinge cabinet:50,slide cabinet:50,top burner:20"
NO_BURNER = "microwave:50,hinge cabinet:50,slide cabinet:50"
OUT = os.path.join(ROOT, "eval_results", "results.jsonl")
WORK = tempfile.mkdtemp(prefix="eval_sweep_")

# Priority order. Runs 7 and 10 at steps 1000-3000 were finished on the desktop.
JOBS = [(r, "e359e9b", s, FULL) for s in (4000, 5000) for r in (7, 10)]
JOBS += [(11, "85b6699", s, FULL) for s in STEPS]
JOBS += [(12, "d635ae8", s, FULL) for s in STEPS]
JOBS += [(6, "fb8b34e", "final", FULL), (2, "6346e2f", "final", NO_BURNER)]
JOBS += [(9, "0bf3889", s, NO_BURNER) for s in STEPS]
JOBS += [(13, "0bf3889", s, NO_BURNER) for s in STEPS]
JOBS += [(8, "b4fcb52", s, NO_BURNER) for s in STEPS]


def worktree(commit):
    path = os.path.join(WORK, commit)
    if not os.path.exists(path):
        subprocess.run(["git", "-C", ROOT, "worktree", "add", "-q", "--detach", path, commit], check=True)
    return path


def run_job(job):
    run, commit, step, tasks = job
    label = f"{run}@{step}"
    name = "smolvlm2.pth" if step == "final" else f"smolvlm2-step{step}.pth"
    checkpoint = os.path.join(WORK, f"{run}-{name}")
    try:
        urllib.request.urlretrieve(f"{API}/{run}/files/checkpoints/{name}", checkpoint)
    except urllib.error.HTTPError as e:
        print(f"SKIP {label}: checkpoint unavailable ({e.code})", flush=True)
        return
    env = dict(os.environ, OMP_NUM_THREADS="6", MKL_NUM_THREADS="6", OPENBLAS_NUM_THREADS="6")
    result = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "eval_checkpoints.py"),
                             "--checkpoint", checkpoint, "--label", label, "--tasks", tasks, "--out", OUT],
                            cwd=worktree(commit), env=env, capture_output=True, text=True)
    os.remove(checkpoint)
    print(f"{'finished' if result.returncode == 0 else 'FAILED'} {label}", flush=True)
    if result.returncode:
        print(result.stderr[-2000:], flush=True)


os.makedirs(os.path.dirname(OUT), exist_ok=True)
subprocess.run(["git", "-C", ROOT, "fetch", "-q", "origin"], check=False)
snapshot_download(SMOLVLM2_CONFIG_500M["model_id"], revision=SMOLVLM2_CONFIG_500M["revision"])
for commit in {job[1] for job in JOBS}:
    worktree(commit)
with ThreadPoolExecutor(WORKERS) as pool:
    list(pool.map(run_job, JOBS))
subprocess.run(["git", "-C", ROOT, "worktree", "prune"], check=False)
print("SWEEP_DONE", flush=True)
