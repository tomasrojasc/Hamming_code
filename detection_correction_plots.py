from utils import decompress_pickle, get_correct_classification_rate, get_correction_worked_rate
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

def pltcolor(lst):
    paleta = list(mcolors.TABLEAU_COLORS.items())
    unicos = np.unique(lst)
    paleta = {str(unicos[i]): paleta[i][0] for i in range(len(unicos))}
    return [paleta[str(i)] for i in lst]

reports = decompress_pickle("reports.pbz2")

reports_keys = reports.keys()
reports_keys = list(reports_keys)
metrics_keys = reports[reports_keys[0]].keys()
metrics_keys = list(metrics_keys)


p_error = []
detection_ratio = []
len_hamming_message = []
correction_worked = []

for _, report in reports.items():
    p_error.append(report["p_error"][0])
    detection_ratio.append(get_correct_classification_rate(report))
    len_hamming_message.append(report["len_hamming_message"][0])
    correction_worked.append(get_correction_worked_rate(report))


p_error = np.array(p_error)
detection_ratio = np.array(detection_ratio)
len_hamming_message = np.array(len_hamming_message)
correction_worked = np.array(correction_worked)





filters = [np.where(p_error == i) for i in np.unique(p_error)]
color = np.array(pltcolor(p_error))


fig, ax = plt.subplots()
for filtro in filters:
    x, y = len_hamming_message[filtro], detection_ratio[filtro]
    order = x.argsort()
    ax.scatter(x[order], y[order], s=20, marker='o', c=color[filtro][0], label="p={:.5f}".format(p_error[filtro][0]))
    ax.plot(x[order], y[order], '--', c=color[filtro][0])

ax.set_xscale("log")
ax.legend()
ax.set_xlabel("largo del mensaje (Hamming)")
ax.set_ylabel("porcentaje de detección")
plt.savefig("plots/percentage_detection_correct.png", dpi=300)
plt.show()









fig, ax = plt.subplots()
for filtro in filters:
    x, y = len_hamming_message[filtro], correction_worked[filtro]
    order = x.argsort()
    ax.scatter(x[order], y[order], s=20, marker='o', c=color[filtro][0], label="p={:.5f}".format(p_error[filtro][0]))
    ax.plot(x[order], y[order], '--', c=color[filtro][0])

ax.legend()
ax.set_xscale("log")
plt.savefig("plots/percentage_correction_worked.png", dpi=300)
plt.show()


