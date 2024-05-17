import zipfile
import multiprocessing
import numpy as np
import os

from keras.datasets import fashion_mnist

import time
start_time = time.time()

# Se lee un archivo npy y se comprime a Zip
def npy_to_zip(npy_file,zip_file, cpLv):

    with open(npy_file, 'rb') as f:
        npy_file = f.read()
    
    with zipfile.ZipFile(zip_file, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=cpLv) as zinpy:
        zinpy.writestr('npyFile.npy', npy_file)

# Se lee el hexacimal de zip como enteros de 8 bits y se devuelve un numpy array
def zip_to_hexNPY(zip_file):
    with open(zip_file, 'rb') as z:
        return np.frombuffer(z.read(), dtype=np.uint8)
    
def hexNPY_to_zip(data, filename):
  # Open file en binary
  with open(filename, 'wb') as file:
    # Convierte el numpy int-8bits en Big-endian
    bytes_data = data.astype(np.uint8).tobytes(order='C')
    # Escribir informacion en el archivo
    file.write(bytes_data)
    
#Los dos pasos anteriores se ponen en una funcion
def transformation(data):
    np.save('mySSDisInPain', data)
    npy_to_zip('mySSDisInPain.npy','mySSDinLessPain.zip',9)
    return(zip_to_hexNPY('mySSDinLessPain.zip'))

#Se transforman los datos de un chunck
def chunkTransformation(chunk):
    return [transformation(item) for item in chunk]

#Se define el procesado multi nucleo, porque sino es muy lento
#Nose si se puede hacer por GPU habria que probar    
def dataSeToZipTozipNPY(data):
    num_cores = multiprocessing.cpu_count()
    chunk_size = len(data) // num_cores
    chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
    with multiprocessing.Pool(processes=num_cores) as pool:
        modified_arrays = pool.map(chunkTransformation, chunks)
    combined_modified_array = [item for sublist in modified_arrays for item in sublist]
    os.remove("mySSDinLessPain.zip")
    os.remove("mySSDisInPain.npy")
    return combined_modified_array

#Esta funcion dada un vector o vectores, los reforma, de forma que todos queden del mismo
#largo sin la necesidad de añadir padding, queda la metadata de la informacion original
def reshaping(vectors: np.ndarray, shape: int) -> np.ndarray: 
    new_vec = []
    try:
        longVector = np.concatenate(vectors) 
    except:
        longVector = vectors
    n = 0
    while(shape+n+1<len(longVector)):
        new_vec.append(longVector[0+n:shape+n])
        n = n + shape
    left = (longVector[0+n:(len(longVector))])
    padd = (shape - len(left))
    new_vec.append( np.pad(left, (0, padd), mode='constant'))
    return new_vec

#Se normalizan los datos poniendolos del 0 al 1
def normalize(array_2d: np.ndarray)-> np.ndarray:
    max_value = np.max(array_2d)
    normalized_array = array_2d / max_value
    return normalized_array

#Creacion del dataSet
def randomDataSetGenerate(sample_size: int, pol_maxGrade: int, fileName: str):
    #Note que esta información random está normalizada entre 0 y 1
    matrix = np.random.rand(sample_size, pol_maxGrade) 
    np.save(fileName, matrix)

# Se procesan los datos
if __name__ == "__main__":
    (x_train, _), (x_test, _) = fashion_mnist.load_data()
    x_train_transformed = normalize(reshaping((dataSeToZipTozipNPY(x_train)),784))
    x_test_transformed = normalize(reshaping((dataSeToZipTozipNPY(x_test)),784))
    np.savez('zipDataSet.npz', x_train=x_train_transformed, x_test=x_test_transformed)
print(time.time() - start_time) # 4.76 minutos en 12 cores