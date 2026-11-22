import runpy
from pathlib import Path

# Experiment branch only: Beekeeper launches train.py, so it runs the camera perturbation eval.
runpy.run_path(str(Path(__file__).resolve().parent / "eval_camera.py"), run_name="__main__")
