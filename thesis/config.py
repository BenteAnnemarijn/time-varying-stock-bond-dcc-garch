import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

cur_dir = Path(__file__).parent.parent.absolute()

data_path = os.getenv("DATA_PATH", "data/input")
output_path = os.getenv("OUTPUT_PATH", "data/output")
plots_path = os.getenv("PLOTS_PATH", "data/output/plots")

DATA_PATH = cur_dir / data_path
OUTPUT_PATH = cur_dir / output_path
PLOTS_PATH = cur_dir / plots_path

DATA_PATH.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
PLOTS_PATH.mkdir(parents=True, exist_ok=True)
