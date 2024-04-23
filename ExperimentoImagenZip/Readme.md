#
En esta carpeta se encuentra AEzip-image.py y ZipDataSetGenerator.py

ZipDataSetGenerator.py:

Este archivo permite pasar un dataset de imágenes representadas con numpy arrays, a la compresión en zip
de estas imágenes pasadas a numpy array. 

Este es un experimento y puede que haya un par de problemas, primero el método para pasar el zip a un array
de nuevo puede no ser correcto, porque para que sea reversible habría que guardar aparte la etiqueta al inicio
del archivo zip. Después la desigualdad de los tamaños y el padding puesto parece ser un problema.

El código está optimizado para múltiples núcleos de un computador porque sino es muy lento, igual seria 
buen ejercicio optimizarlo para que use la GPU

AEzip-image.py:
Este es el modelo como tal, el que está ahora mismo es la versión que va de zip.npy a una imagen,
escalada que no tiene mucho sentido, lo que también puede ser un problema, para poder ver las 
imagenes habría que primero crear las funciones que permiten ir a zip.npy a el formato original
de igual forma se puede analizar el AE yendo de zip.npy a zip.npy con los recursos actuales

Por otro lado todo esto es un experimento y aunque considero de que son ideas interesantes de explorar
carecen de un paper o algún sustento teórico detrás. Pero en inicio intentar algo de por si algo ya
muy comprimido que no ha perdido nada de información respecto a su original no parece mala idea. 
