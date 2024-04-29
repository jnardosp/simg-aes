import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras import losses,  callbacks

from utilities.aeArchitectures import *
from utilities.generatorDS import randomDataSetGenerate
from utilities.analysis import *

def AE(latent_dim):
    #Definicion parametros 
    fileName: str = 'Random'
    sample_size: int = 1000
    pol_maxGrade: int = 1024
    latent_dim: int = latent_dim
    #Lo defini asi el num neuronas para facilidad de modificacion respecto a la salida y entrada
    interval = (pol_maxGrade-latent_dim)//3
    neuLayers = [pol_maxGrade, interval*2+latent_dim, interval+latent_dim]

    randomDataSetGenerate(sample_size, pol_maxGrade, fileName)

    #Lectura del data 
    x_train = (np.load(fileName+'.npy'))

    #Split por train y test
    x_train, x_test = train_test_split(x_train, test_size=0.2, random_state=42)

    autoencoder = Autoencoder(latent_dim, neuLayers) #Generacion del modelo

    #Compilamos
    autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())

    #EarlyStop para que no haga Overfitting
    early_stop = callbacks.EarlyStopping(monitor='val_loss',patience=5)

    #Entrenamos
    history = autoencoder.fit(x_train, x_train,
            epochs=35,
            shuffle=True,
            validation_data=(x_test, x_test),
            callbacks=[early_stop])

    #Resultados Error
    loss = history.history['loss']

    #Obtencion tamaños compresiones
    sample = x_train[1] 
    x_expanded = np.expand_dims(sample, axis=0)
    __, AEcompressed_size = get_Size(sample, autoencoder.getEncoded(x_expanded))

    return AEcompressed_size, loss[-1]

inicio = 1024
compression = []
lost = []
parts = 10
intervals = [int((inicio/10.0)*n) for n in range(1,parts+1)]
for interval in intervals:
    size,loss = AE(interval)
    compression.append(size)
    lost.append(loss)

    compression = np.array(compression)
lost = np.array(lost)
max = np.max(compression)/np.max(lost)
compression = compression/max

plt.plot(compression, label='Compression')
plt.plot(lost, label='Loss')
plt.xlabel('Bottleneck')
plt.ylabel('Parameters')
plt.legend()
plt.show()