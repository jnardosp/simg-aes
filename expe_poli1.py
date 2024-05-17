import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split
from keras import losses,  callbacks

from utilities.aeArchitectures import *
from utilities.generatorDS import randomDataSetGenerate
from utilities.analysis import *

def randomPolinomialDataset(sizeDataset, coefficient_size, pol_maxGrade, fileName):
    ds = []
    for i in range(sizeDataset):
        polinomial = []
        for grade in range(pol_maxGrade):
            polinomial.append(random.uniform(-coefficient_size, coefficient_size))
        ds.append(polinomial)
    print(ds)
    dataset = np.array(ds)
    np.save(fileName, dataset)

#Definicion parametros 
fileName: str = 'RandomSmall'
coefficient_size: int = 10000
pol_maxGrade: int = 16
latent_dim: int = 4
neuLayers = [pol_maxGrade, 8]

randomDataSetGenerate(coefficient_size, pol_maxGrade, fileName)
#randomPolinomialDataset(10000, coefficient_size, pol_maxGrade, fileName)

#Lectura del data 
x_train = (np.load(fileName+'.npy'))

#Split por train y test
x_train, x_test = train_test_split(x_train, test_size=0.2, random_state=42)

autoencoderSmall = Autoencoder(latent_dim, neuLayers) #Generacion del modelo

#Compilamos
autoencoderSmall.compile(optimizer='adam', loss=losses.MeanSquaredError())

#EarlyStop para que no haga Overfitting
early_stop = callbacks.EarlyStopping(monitor='val_loss',patience=5)

#Entrenamos
history = autoencoderSmall.fit(x_train, x_train,
            epochs=35,
            shuffle=True,
            validation_data=(x_test, x_test),
            callbacks=[early_stop])
            
#Resultados
loss = history.history['loss']
val_loss = history.history['val_loss']
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

#Para extraer nuestro ejemplo usamos las funciones 'getEncoded' y 'getDecoded' de Autoencoder(Model)
#Extraemos lo que pasaría con un ejemplo dado
example = np.array([0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
x_expanded = np.expand_dims(example, axis=0)
#Bug en el tipo de datos de entrada ?? Revisar kerastensor shape.
encodedExample = autoencoderSmall.getEncoded(x_expanded)
decodedExample = autoencoderSmall.getDecoded(encodedExample)
print("This is the polynomial example: {}".format(x_expanded))
print("This is the encoded Example : {}".format(encodedExample))
print("This is the decoded Example: {}".format(decodedExample))
