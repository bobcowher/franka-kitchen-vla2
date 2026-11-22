import runpy
from pathlib import Path

# Experiment branch only: Beekeeper launches train.py, so it runs the DoneCNN experiment.
runpy.run_path(str(Path(__file__).resolve().parent / "train_done_cnn.py"), run_name="__main__")
