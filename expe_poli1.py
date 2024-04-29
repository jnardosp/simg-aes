import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras import losses,  callbacks

from utilities.aeArchitectures import *
from utilities.generatorDS import randomDataSetGenerate
from utilities.analysis import *

#Definicion parametros 
fileName: str = 'RandomSmall'
coefficient_size: int = 10000
pol_maxGrade: int = 16
latent_dim: int = 4
neuLayers = [pol_maxGrade, 8]

randomDataSetGenerate(coefficient_size, pol_maxGrade, fileName)

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
example = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
#Bug en el tipo de datos de entrada ?? Revisar kerastensor shape.
encodedExample = autoencoderSmall.getEncoded(example)
print("This is the polynomial example: {}".format(example))
print("This is the encodedExample : {}".format(encodedExample))
