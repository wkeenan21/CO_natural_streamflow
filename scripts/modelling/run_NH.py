import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from neuralhydrology.evaluation import metrics
from neuralhydrology.nh_run import start_run, eval_run

start_run(config_file=Path(r"C:\Users\C830645719\OneDrive - Colostate\documents\GitHub\CO_natural_streamflow\scripts\configs\config1.yml"), gpu=-1)
