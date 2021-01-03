from utils import Hamming, Mensaje, Transit

msj = Mensaje('Trabajo para el Laboratorio de Ingeniería')
hamming = Hamming(msj, 11)
transit = Transit(hamming)
transit.noise(.1)
