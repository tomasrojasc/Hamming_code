from utils import Experiment, plot_confussion_matrix, compressed_pickle
from utils import decompress_pickle, percentages_erros, dictionaries_to_lists
import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

file = 'reports.pbz2'
open_reports = True
n_trials = 1000

if not os.path.isfile(file):  # si no están los experimentos guardados
    probs = 1 / np.logspace(0, 4, 10)
    lengths = np.logspace(1, 5, 10, dtype=int)
    experiments = []
    for n in tqdm(lengths, ncols=100):
        for p in probs:
            current_experiment = Experiment(p, n, n_trials)
            current_experiment.make_report()
            experiments.append(current_experiment)

    reports = {experiment.__repr__(): experiment.report for experiment in experiments}
    compressed_pickle(file, reports)
    open_reports = False
    del experiments

if open_reports:  # si la variable "reports" no está asignada
    reports = decompress_pickle(file)

percentage_more_than_one = []

for name, report in reports.items():
    plot_confussion_matrix(report, title=name, save_as=f'cms/{name}')
    plt.close("all")

error_percentage = []
for _, report in reports.items():
    error_percentage.append(percentages_erros(report))
error_percentage = dictionaries_to_lists(error_percentage)
error_percentage['theoretical'] = np.array([i[0] for i in error_percentage['theoretical']])
error_percentage['empirical'] = np.array(error_percentage['empirical'])
plt.plot(error_percentage['theoretical'], error_percentage['empirical'], '+')
plt.xlabel("porcentaje con más de un error teórico")
plt.ylabel("porcentaje con más de un error empírico")

plt.savefig("plots/percentage.png", dpi=300)
plt.show()
plt.close("all")

distance = error_percentage['theoretical'] - error_percentage['empirical']
plt.hist(distance, bins=int(len(error_percentage['theoretical'])/4))
plt.xlabel("Distancia entre observado y predicho")
plt.title("media={:.2f}, desviación estándar={:.2f}".format(distance.mean(), np.std(distance)))
plt.savefig('plots/histogram.png', dpi=300)
plt.show()


