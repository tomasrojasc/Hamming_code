import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom

p = 1 / np.logspace(0, 4, 10)
n = np.logspace(0,5, 100)
legend = []

for prob in p:
    current = []
    for ns in n:
        rvs = binom(ns, prob)
        current.append(1 - rvs.pmf(0) - rvs.pmf(1))
    plt.plot(n, current, 'x--')
    legend.append('p = {:.4f}'.format(prob))
plt.xlabel("largo del bloque")
plt.ylabel("probabilidad de tener más de un error en el bloque")
plt.legend(legend, bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xscale("log")
plt.tight_layout()
plt.savefig('plots/p_more_than_1.png', dpi=300)
plt.show()