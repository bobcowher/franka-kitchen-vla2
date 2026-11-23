import runpy
from pathlib import Path

# Experiment branch only: Beekeeper launches train.py, so this hands off to the proprio ablation trainer.
runpy.run_path(str(Path(__file__).resolve().parent / "train_proprio.py"), run_name="__main__")
