import numpy as np
import matplotlib.pyplot as plt

n = np.logspace(0, 5, 10, dtype=int)
ps = 1 / np.logspace(0, 4, 10)
ps = ps[::-1]
legend = []

p_correction_worked = []
for p in ps:
    p_correction_worked.append((n * p * (1 - p) ** (n - 1)) / (1 - (1 - p) ** n))
    legend.append("{:.5f}".format(p))


for p_worked in p_correction_worked:
    plt.plot(n, p_worked, 'o--')

plt.xscale("log")
plt.legend(legend)
plt.xlabel("largo del mensaje (Hamming)")
plt.ylabel("porcentaje de corrección exitosa")
plt.savefig("plots/percentage_correction_worked_theoretical.png", dpi=300)
plt.show()
