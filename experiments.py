from utils import Experiment
import numpy as np
from tqdm import tqdm



probs = np.linspace(.1, 1, 10)
lengths = np.logspace(1, 6, 10, dtype=int)

experiments = []
n_trials = 100
for n in tqdm(lengths, ncols=100):
    for p in probs:
        current_experiment = Experiment(p, n, n_trials)
        current_experiment.make_report()
        experiments.append(current_experiment)


