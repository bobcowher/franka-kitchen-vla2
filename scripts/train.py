import runpy
from pathlib import Path

# Experiment branch only: Beekeeper launches train.py, so it runs the encoder offset eval.
runpy.run_path(str(Path(__file__).resolve().parent / "eval_encoder.py"), run_name="__main__")
