import numpy as np
import matplotlib.pyplot as plt
from keras import losses,  callbacks

from utilities.aeArchitectures import *
from utilities.analysis import *
from utilities.generatorDS import normalize
from keras.datasets import fashion_mnist

#Obtencion zip data
data = np.load('zipDataSet.npz')
x_train = data['x_train']
x_test= data['x_test']

#Obtencion data no lossless compression 
(x_trainN, _), (x_testN, _) = fashion_mnist.load_data()
x_trainN = normalize(x_trainN.reshape(x_trainN.shape[0], -1))
x_testN = normalize(x_testN.reshape(x_testN.shape[0], -1))

pol_maxGrade: int = 784
latent_dim: int = 128
neuLayers = [pol_maxGrade, 512, 256]

#Construccion Autoencoder con compressed zip data-----------------------------
autoencoderZip = Autoencoder(latent_dim, neuLayers) #Generacion del modelo

#Compilamos
autoencoderZip.compile(optimizer='adam', loss=losses.MeanSquaredError())

#EarlyStop para que no haga Overfitting
early_stop = callbacks.EarlyStopping(monitor='val_loss',patience=10)

#Entrenamos
history = autoencoderZip.fit(x_train, x_train,
        epochs=1,
        shuffle=True,
        validation_data=(x_test, x_test),
        callbacks=[early_stop])

#Construccion auto encoder normal----------------------------------------
autoencoderN = Autoencoder(latent_dim, neuLayers) #Generacion del modelo

#Compilamos
autoencoderN.compile(optimizer='adam', loss=losses.MeanSquaredError())

#EarlyStop para que no haga Overfitting
early_stopN = callbacks.EarlyStopping(monitor='val_loss',patience=10)

#Entrenamos
historyN = autoencoderN.fit(x_trainN, x_trainN,
        epochs=1,
        shuffle=True,
        validation_data=(x_testN, x_testN),
        callbacks=[early_stopN])

#Obtencion tamaños compresiones
sample = x_train[1] 
x_expanded = np.expand_dims(sample, axis=0) #esto es porque un problema con la shape
original_size, AEcompressed_size = get_Size(sample, autoencoderZip.getEncoded(x_expanded))

#Obtencion tamaños compresiones AE-N
sample = x_trainN[1] 
x_expanded = np.expand_dims(sample, axis=0) #esto es porque un problema con la shape
original_sizeN, AEcompressed_sizeN = get_Size(sample, autoencoderN.getEncoded(x_expanded))
dataSetZip_sizeN = get_DataSetZipSize(x_trainN)

#Resultados Error
loss = history.history['loss']
val_loss = history.history['val_loss']
lossN = historyN.history['loss']
val_lossN = historyN.history['val_loss']
plt.plot(loss, label='Training LossZ')
plt.plot(val_loss, label='Validation LossZ')
plt.plot(lossN, label='Training LossN')
plt.plot(val_lossN, label='Validation LossN')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

#Resultados Compression    
# Datos originales
labels = [ 'Original', 'AE', 'ZipN', 'OriginalN', 'AEN']
values = [ original_size*len(x_train), AEcompressed_size*len(x_train),
        dataSetZip_sizeN, original_sizeN*len(x_trainN), AEcompressed_sizeN*len(x_trainN)]

# Ordenar los datos
sorted_indices = sorted(range(len(values)), key=lambda k: values[k], reverse=True)
sorted_labels = [labels[i] for i in sorted_indices]
sorted_values = [values[i] for i in sorted_indices]

# Asignar colores
colors = ['blue' if label.endswith('N') else 'orange' for label in sorted_labels]

# Crear el gráfico de barras
plt.bar(sorted_labels, sorted_values, color=colors)
plt.xlabel('file num')
plt.ylabel('Size')
plt.show()