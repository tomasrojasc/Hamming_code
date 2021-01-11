# Hamming Code
En este repositorio se encuentra todo el código usado para la experiencia final del curso _Laboratorio de ingeniería eléctrica_
donde se hizo estudio de los Hamming codes, una manera de cifrar mensajes para ser resilientes a errores, puede encontrar el paper original [acá](https://signallake.com/innovation/hamming.pdf)

Este proyecto consta de dos módulos principales, el más fundamental es `utils.py` donde se encuentra el grueso del proyecto y los objetos necesarios para poder llevar a cabo los experimentos.

En particular hay dos objetos importantes:

`Mensaje` es un objeto que representa un mensaje en binario que es generado de manera aleatoria, tiene métodos para poder enviar el mensaje binario a travez de un canal ruidoso con probabilidad `p` 

`Experimento` es un objeto encargado de administrar mucchas instancias de `Mensaje` con carácteristicas similares, solo difiriendo en el mensaje en sí, esto nos permite sacar métricas estadísticas sobre el comportamiento del algoritmo.
