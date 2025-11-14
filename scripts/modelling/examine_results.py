import pandas as pd
import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from neuralhydrology.evaluation import metrics
from neuralhydrology.nh_run import start_run, eval_run

cwd = os.getcwd()

run_dir = Path(os.path.join(cwd, r"data\NH_data\run_dir\25_epoch_complete\1st_attempt_1211_231454"))
eval_run(run_dir=run_dir, period="test")
