import numpy as np
import os
from utilities.generatorDS import npy_to_zip

    #Resultados Compresion
def get_Size(original: np.array, AEcompressed: np.array):
    # Crear archivos .npy temporales
    original_path = 'original.npy'
    AEcompressed_path = 'AEcompressed.npy'

    #Crear archivos numpy
    np.save(original_path, original)
    np.save(AEcompressed_path, AEcompressed)

    # Obtener tamaños
    original_size = os.path.getsize(original_path)
    AEcompressed_size = os.path.getsize(AEcompressed_path)

    # Eliminar archivos temporales y el archivo zip
    os.remove(original_path)
    os.remove(AEcompressed_path)

    return original_size, AEcompressed_size

def get_DataSetZipSize(original):
    original_path = 'original.npy'
    zip_path = 'compressed_data.zip'

    np.save(original_path, original)
    npy_to_zip('original.npy','compressed_data.zip', 9)

    zip_size = os.path.getsize(zip_path)

    os.remove(zip_path)
    os.remove(original_path)

    return zip_size