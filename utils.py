import numpy as np


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
            correction_worked=correction_worked
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

if __name__ == "__main__":
    msj = Mensaje(100)
    msj.make_hamming_msj()
    msj.send_noisy(.5)
    msj.get_metrics()
