import zipfile
import multiprocessing
import numpy as np

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
def zip_to_NPY(zip_file):
    with open(zip_file, 'rb') as z:
        return np.frombuffer(z.read(), dtype=np.uint8)
    
#Los dos pasos anteriores se ponen en una funcion
def transformation(data):
    np.save('mySSDisInPain', data)
    npy_to_zip('mySSDisInPain.npy','mySSDinLessPain.zip',9)
    return(zip_to_NPY('mySSDinLessPain.zip'))

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
    return combined_modified_array

#Se normalizan los datos poniendolos del 0 al 1, y se pone padding para que queden 
#todos del mismo tamaño
def normalizeData(vectors):
    matriz = []
    max_val=0
    max_length = max(len(vector) for vector in vectors)
    for vector in vectors:
        for value in vector:
            if(value>=max_val): max_val = value
    for vector in vectors:
        vector = vector/max_val
        matriz.append(np.array(vector.tolist() + [0]*(max_length-len(vector))))
    return np.array(matriz) 

#Se procesan los datos
if __name__ == "__main__":
    (x_train, _), (x_test, _) = fashion_mnist.load_data()
    np.save('zipDataSet.npy', (normalizeData(dataSeToZipTozipNPY(x_train))))
    np.save('zipDataSet.npy', (normalizeData(dataSeToZipTozipNPY(x_test))))

    print(time.time() - start_time) # 4.39 minutos en 12 cores