import numpy as np
import bz2
from scipy.stats import binom
import _pickle as cPickle

def make_insertions(array, insertions, inserted_number):
    """
    esta función inserta en el array y en las posiciones
    :param array: np array
    :param insertions: list of ints
    :param inserted_number: number to insert
    :return: array with the insertions performed
    """
    if len(insertions) == 0:
        return array
    else:
        array = np.insert(array, insertions[0], inserted_number)
        insertions = insertions[1:]
        return make_insertions(array, insertions, inserted_number)


def binary_decomposition(binary_representation):
    """
    esta función descompone un str que representa un nro binario en una suma de decimales
    :param binary_representation: string de repr binaria, debe comenzar con '0b'
    :return: lista con los número decimales que hay que sumar para obtener el mismo numero binario pero en decimal
    """
    assert binary_representation[:2] == '0b'
    binary = np.flip(np.array(list(binary_representation[2:])).astype(int))
    base = np.array([2 ** i for i in range(len(binary))])
    decomposition = base * binary
    return decomposition[decomposition != 0]


class Mensaje:
    def __init__(self, n):
        """
        :param n: número de bits del mensaje
        """
        self.n = n
        self.msj = np.random.randint(2, size=n)
        self.parity_bits = None
        self.p_error = None
        self.hamming_msj = None
        self.received_msj = None
        self.corrected_msj = None
        self.metrics = None

    def __len__(self):
        if isinstance(self.hamming_msj, type(None)):
            return 0
        return len(self.hamming_msj)

    def make_hamming_msj(self):
        """
        esta función se encarga de generar el código Hamming para enviar y lo guarda en self.hamming_msj
        :return: None
        """
        n = 0  # némero de bits de paridad
        m = len(self.msj)  # número de bits del mensaje

        while not (m + n <= 2 ** n):
            n += 1

        self.parity_bits = n
        where_to_insert = [2 ** i - 1 for i in range(n)]
        no_parity_hamming = make_insertions(self.msj, where_to_insert, 0)

        assert len(no_parity_hamming) == n + m  # por si las moscas

        # vemos donde hay 1s y hacemos el xor correspondiente
        idx_to_xor = np.where(no_parity_hamming == 1)[0] + 1  # tomamos los indices y sumamos 1
        xor_result = np.bitwise_xor.reduce(idx_to_xor)  # aplicamos xor
        bin_xor_result = bin(xor_result)  # representación binaria del resultado
        positions_from_one_to_flip = binary_decomposition(bin_xor_result)  # lista de qué indices dar vuelta,
        # contando a partir de 1
        indeces_to_flip = positions_from_one_to_flip - 1
        no_parity_hamming[indeces_to_flip] = 1
        hamming_msj = no_parity_hamming.copy()  # explicitamos que ahora sí están bien
        self.hamming_msj = hamming_msj
        return

    def send_noisy(self, p):
        """
        Esta función agrega ruido con probabilidad p, guarda el resultado en self.received_msj
        :param p: probabilidad de bitflip
        :return: None
        """
        if isinstance(self.hamming_msj, type(None)):
            self.make_hamming_msj()
        self.p_error = p
        noise = np.random.binomial(1, p, len(self.hamming_msj))
        self.received_msj = np.mod(self.hamming_msj + noise, 2)
        return

    def get_metrics(self):
        """
        esta función genera métricas interesantes
        :return: None
        """
        if isinstance(self.received_msj, type(None)):
            print("primero ejecutar método 'send_noisy'")
            return
        hamming_distance = np.bitwise_xor(self.hamming_msj, self.received_msj).sum()  # número de errores

        if hamming_distance > 0:
            errors = True
        else:
            errors = False

        bits_for_parity_check = np.where(self.received_msj == 1)[0] + 1
        parity_check = np.bitwise_xor.reduce(bits_for_parity_check)
        if parity_check > 0:
            found_error = True
        else:
            found_error = False

        correction_worked = None
        if found_error:
            if parity_check > len(self.received_msj):
                out_of_bound = True
                hamming_distance_corrected = hamming_distance
                msj_corrected = self.received_msj
            else:
                out_of_bound = False
                index_to_flip = parity_check - 1

                msj_corrected = self.received_msj.copy()
                msj_corrected[index_to_flip] = np.mod(msj_corrected[index_to_flip] + 1, 2)

                hamming_distance_corrected = np.bitwise_xor(self.hamming_msj, msj_corrected).sum()
            if (hamming_distance_corrected > 0) or out_of_bound:
                correction_worked = False
            else:
                correction_worked = True
        else:
            out_of_bound = False
            msj_corrected = self.hamming_msj

        metrics = dict(
            hamming_distance=hamming_distance,
            errors=errors,
            found_error=found_error,
            out_of_bound=out_of_bound,
            correction_worked=correction_worked,
            len_hamming_message=len(self.hamming_msj),
            p_error=self.p_error
        )

        self.corrected_msj = msj_corrected
        self.metrics = metrics
        return


class Experiment:
    def __init__(self, p, bits, number_of_trials):
        self.p, self.n_bit, self.n = p, bits, number_of_trials
        self.experiments = []
        self.report = None

        for i in range(number_of_trials):
            current_experiment = Mensaje(self.n_bit)
            current_experiment.send_noisy(self.p)
            current_experiment.get_metrics()
            self.experiments.append(current_experiment)

        self.metric_keys = [key for key in self.experiments[0].metrics]

    def __repr__(self):
        largo_bloque = len(self.experiments[0].hamming_msj)
        text = "Experimento con p={:.2e}, largo de bloque={}, instancias={}".format(self.p, largo_bloque, self.n)
        return text

    def make_report(self):
        """
        esta función hace un resumen de toodo lo que pasó
        :return: None
        """
        self.report = {}
        for key in self.metric_keys:
            current_key = []
            for experiment in self.experiments:
                current_key.append(experiment.metrics[key])
            self.report[key] = current_key
        return


def plot_cm(cm,
            target_names,
            title='Confusion matrix',
            cmap=None,
            normalize=True,
            save_as=None):
    """
    ---------
    Esta función no es de mi autoría y se puede encontrar en
    https://www.kaggle.com/grfiv4/plot-a-confusion-matrix
    ---------
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    save_as:      si se guarda o no, de gusrdarse, se guarda con el nombre dado,
                  si no, ignorar

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    if not isinstance(save_as, type(None)):
        plt.savefig(save_as + '.png', dpi=300)
    else:
        plt.show()


def plot_confussion_matrix(metrics, title="Confusion matrix", save_as=None):
    from sklearn.metrics import confusion_matrix
    y_pred, y_true = np.array(metrics["found_error"], dtype=int), np.array(metrics["errors"], dtype=int)
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (1, 1):
        if y_pred.sum() == len(y_pred):
            cm = np.array([[0,    0   ],
                           [0, cm[0, 0]]])
        else:
            cm = np.array([[cm[0, 0],   0],
                           [0,          0]])
    plot_cm(cm, ['no error', 'error'], title=title, normalize=False, save_as=save_as)

def compressed_pickle(title, data):
    with bz2.BZ2File(title, "w") as f:
        cPickle.dump(data, f)

def decompress_pickle(file):
    data = bz2.BZ2File(file, "rb")
    data = cPickle.load(data)
    return data


def dictionaries_to_lists(list_of_dict):
    # se asume que las llaves son las mismas
    assert len(list_of_dict) > 0

    keys = list_of_dict[0].keys()
    final = {}
    for key in keys:
        final[key] = []

    for dictionarie in list_of_dict:
        for key in keys:
            final[key].append(dictionarie[key])

    return final


def percentages_erros(report):
    count = np.sum([1 for i in report["hamming_distance"] if i > 1])
    instances = len(report["hamming_distance"])
    empirical = count / instances
    len_hamming = report["len_hamming_message"]

    binomial = binom(c)
    theoretical = 1 - binomial.pmf(0) - binomial.pmf(1)
    return dict(theoretical=theoretical, empirical=empirical)


def get_correct_classification_rate(report):
    """
    esta función toma un reporte y ve el porcentaje de aciertos en cuanto a sí se encontró o no un error
    :param report: un reporte
    :return: float
    """
    y_true = report['errors']
    y_predict = report["found_error"]
    n = len(y_true)
    correct = 0
    for i in range(n):
        if y_true[i] == y_predict[i]:
            correct += 1
    return correct / n

def get_correction_worked_rate(report):
    """
    esta función nos dice cuantas correcciones funcionarion SOLO SE CUENTA CUANDO HUBO ERRORES
    :param report: un reporte
    :return: float
    """
    worked = [i for i in report["correction_worked"] if not isinstance(i, type(None))]
    n = len(worked)
    return np.sum(worked) / n

if __name__ == "__main__":
    msj = Mensaje(100)
    msj.make_hamming_msj()
    msj.send_noisy(.5)
    msj.get_metrics()
