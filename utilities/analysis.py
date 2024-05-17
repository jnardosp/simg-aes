import numpy as np
import os
from utilities.generatorDS import npy_to_zip, zip_to_hexNPY

    #Resultados Compresion
def get_Size(original: np.array, AE: np.array):
    np.save('original.npy', original)
    np.save('AE.npy', AE)
    npy_to_zip('original.npy','ORcompressed_data.zip', 9)
    npy_to_zip('AE.npy','AEcompressed_data.zip', 9)
    a = zip_to_hexNPY('original.npy')
    b = zip_to_hexNPY('AE.npy')
    c = zip_to_hexNPY('ORcompressed_data.zip')
    d = zip_to_hexNPY('AEcompressed_data.zip')
    os.remove('original.npy')
    os.remove('ORcompressed_data.zip')
    os.remove('AE.npy')
    os.remove('AEcompressed_data.zip')
    sizes = [a.nbytes, b.nbytes, c.nbytes, d.nbytes]
    return sizes