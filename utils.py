
import numpy as np
import matplotlib.pyplot as plt
from bitstring import BitArray
import math

class Mensaje:
    def __init__(self, text):
        self.text = text
        self.bitarray = BitArray(str.encode(text))
        self.binary = self.bitarray.bin
        self.hex = self.bitarray.hex
        self.numpy = self.as_numpy()

    def as_numpy(self):
        return np.array([int(i) for i in list(self.binary)])


    def get_chunks(self, l_chunks=11):
        n = len(self.binary)
        np_mensaje = self.numpy
        chunks = []
        for i in range(n // l_chunks):
            chunk = np_mensaje[i * l_chunks: (i + 1) * l_chunks]
            chunks.append(chunk)
        if n - l_chunks * (n // l_chunks) > 0:
            extra_chunks = np_mensaje[l_chunks * (n // l_chunks):]
            extra_chunks = [np.array([i]) for i in extra_chunks]
            for i in extra_chunks:
                chunks.append(i)
        return chunks



def get_bin_pos(size_block):
    """
    esta función toma unlargo del cuadrado del bloque del código y devuelve
    una representación binaria del indexado
    :param size_block: largo del lado del bloque. Debe ser potencia de 2
    :return: list
    """
    assert (size_block & (size_block - 1) == 0) and size_block != 0
    bin_ind = [np.array(list(bin(i)[1:].replace('b', ''))).astype(np.int) for i in range(size_block ** 2)]
    max_len = np.max([len(i) for i in bin_ind])

    for i in range(len(bin_ind)):
        current_len = len(bin_ind[i])
        missing = max_len - current_len
        bin_ind[i] = np.pad(bin_ind[i], (missing, 0), 'constant', constant_values=(0,))

    return bin_ind


def make_hamming_block_no_parity(chunk):  # hamming block no parity
    """
    esta función genera un bloque de hamming con la dimensiones correspondientes pero no se fija en la paridad
    :param chunk: un chunck antes de ser un código hamming
    :return: un código hamming sin chequear paridad
    """
    if len(chunk) == 1:
        i = chunk[0]
        return np.array([0, 0, 0, i])
    else:
        n = math.ceil(math.log2(len(chunk)))
        bin_positions = get_bin_pos(n)
        indices_to_insert = []
        for i in range(len(bin_positions)):
            if bin_positions[i].sum() == 1 or bin_positions[i].sum() == 0:
                chunk = np.insert(chunk, i, 0)
        return chunk

def decompose_binary(binary_representation):
    """
    esta función toma una representación binaria y la desompone en las sumas de sus componentes binarios
    ejemplo: [1, 0, 0, 1] -> [[1, 0, 0, 0], [0, 0, 0, 1]]
    :param binary_representation: representación de un número binario como un numpy array
    :return: list of np array
    """
    n = len(binary_representation)
    decomposition = []
    for i in range(n):
        if binary_representation[i] == 1:
            new_component = np.pad([1], (i, n - i - 1))
            decomposition.append(new_component)
    return decomposition



def make_parity_0(hamming_block):
    """
    esta función se encarga de ver la paridad de un bloque de hamming al final de ver la
    paridad de los bloques individuales
    :param hamming_block: un numpy array que representa un hamming code con la paridad lista
    :return: hamming block pero con la paridad además del bloque entero
    """
    if np.mod(hamming_block.sum(), 2) == 1:
        hamming_block[0] = 1
        return hamming_block
    return hamming_block

def make_parity(hamming_block_no_parity):
    """
    esta función se encarga de que la paridad en cada grupo del bloque hamming sea correcta,
    incluyendo la paridad del bloque completo
    :param hamming_block_no_parity: un bloque hamming sin paridad
    :return: un bloque hamming con paridad en forma de np array
    """
    index_bin = get_bin_pos(int(math.sqrt(hamming_block_no_parity.shape[0])))  # primero obtenemos la pos en binario
    for_xor = []

    for i, idx in enumerate(index_bin):
        if hamming_block_no_parity[i] == 1:
            for_xor.append(idx)

    if len(for_xor) == 0:
        hamming_block = make_parity_0(hamming_block_no_parity)  # explicitamos el chequeo de paridad
        return hamming_block

    parity_check = np.mod(np.vstack(for_xor).sum(axis=0), 2)  # equivalente a xor

    parity_check_components = decompose_binary(parity_check)

    for i, element in enumerate(index_bin):
        for parity_check_component in parity_check_components:
            if np.all(np.array(element) == np.array(parity_check_component)):
                hamming_block_no_parity[i] = 1
    hamming_block = make_parity_0(hamming_block_no_parity)  # explicitamos el chequeo de paridad
    return hamming_block

class Hamming:
    def __init__(self, mensaje, n):
        self.n = n
        self.mensaje = mensaje
        self.chunks = mensaje.get_chunks(n)
        self.hamming_blocks = []

        for i, chunk in enumerate(self.chunks):
            hamming_no_parity = make_hamming_block_no_parity(chunk)
            self.hamming_blocks.append(make_parity(hamming_no_parity))


def random_flip(bit, p):
    """
    Esta función da la probabilidad de dar vuelta un bit con prob p
    :param p: probabilidad de dar vuelta un bit
    :return: si se da o no vuelta el bit
    """
    assert p >= 0, p <= 1
    flip = np.random.binomial(1, p)
    if flip:
        return (bit + 1) % 2
    else:
        return bit



class Transit:
    def __init__(self, hamming):
        """
        este objeto recibe un objeto hamming y guarda cosas importantes
        :param hamming: objeto Hamming
        """
        self.hamming = hamming
        self.hamming_blocks = self.hamming.hamming_blocks.copy()
        self.received_blocks = None

    def noise(self, p):
        """
        pasa el mensaje por un canal ruidoso
        :param p: probabilidad de dar vuelta un bit
        :return: None
        """
        sending = self.hamming_blocks.copy()
        received = []
        for block in sending:
            new_block = []
            for bit in block:
                new_block.append(random_flip(bit, p))
            received.append(np.array(new_block))
        self.received_blocks = received
        return




"""
TODO: tengo que hacer la paridad de un hamming code, para eso tengo que revisar donde poner o no un uno en vez de un 0
una vez hecho eso el mensaje está listo
_____
después pasamos a hacer el decoder, donde hacemos el parity check y pillamos el o los errores
_____ 

finalmente hacemos el canal ruidoso para automatizar esto
-----
hacemos muchos experimentos y hacemos métricas
"""


if __name__ == "__main__":
    msj = Mensaje('Trabajo para el Laboratorio de Ingeniería')
    hamming = Hamming(msj, 11)