import io
import zipfile
import sys
import os

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from keras import layers, losses, Model, callbacks

from ZipDataSetGenerator import npy_to_zip

#Por si se quiere ver como polinomio
def poliVisualizer():
    pass

#Creacion del dataSet
def randomDataSetGenerate(sample_size: int, pol_maxGrade: int, fileName: str):
    matrix = np.random.rand(sample_size, pol_maxGrade) 
    np.save(fileName, matrix)

#Generador Estructura model
class Autoencoder(Model):
    def __init__(self, latent_dim, neuLayers):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.neuLayers = neuLayers

        self.encoder_layers = [] 
        for neurons in self.neuLayers: #Hidden layers y #input layer
            self.encoder_layers.append(layers.Dense(neurons, activation='relu'))
        self.encoder_layers.append(layers.Dense(latent_dim, activation='relu', name="middleEncoder")) #output layer
        ###### Bottle Neck
        self.decoder_layers = [] 
        self.decoder_layers.append(layers.Dense(latent_dim, activation='relu', name="middleDecoder")) #input layer
        for neurons in reversed(self.neuLayers): #Hidden layers y #output layer
            self.decoder_layers.append(layers.Dense(neurons, activation='relu'))
        

        self.encoder = tf.keras.Sequential(self.encoder_layers) # create encoder
        self.decoder = tf.keras.Sequential(self.decoder_layers) # create decoder

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def getEncoded(self, data):
        encoded = self.encoder(data).numpy()
        return encoded


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

def generalAE():
    #Definicion parametros 
    fileName: str = 'Random'
    sample_size: int = 10000
    pol_maxGrade: int = 1024
    latent_dim: int = 128
    neuLayers = [pol_maxGrade, 512, 256]

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
    val_loss = history.history['val_loss']
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    #Obtencion tamaños compresiones
    sample = x_train[1] 
    x_expanded = np.expand_dims(sample, axis=0)
    original_size, AEcompressed_size = get_Size(sample, autoencoder.getEncoded(x_expanded))
    print( ' original:' + str(original_size) + ' AE:' + str(AEcompressed_size) )
    dataSetZip_size = get_DataSetZipSize(x_train)

    #Resultados Compression    
    plt.bar(['Zip','Original','AE'],
            [dataSetZip_size, original_size*len(x_train), AEcompressed_size*len(x_train)])
    plt.xlabel('file num')
    plt.ylabel('Size')
    plt.show()

'''Hasta este punto parece que el AE funciona perfectamente, haré el experimento de introducir un polinomio, revisar su representación comprimida
   Y después su output. (Para efectos del experimento pondré un polinomio de un grado mucho más pequeño)'''

def experimentAE():
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

    #Para extraer nuestro ejemplo creamos un nuevo modelo
    layerToCheck = 'middleEncoder'
    compressedLayerModel = Model(inputs=autoencoderSmall.input,  outputs=autoencoderSmall.get_layer(layerToCheck).output())
    #Extraemos lo que pasaría con un ejemplo dado
    example = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    compressedLayerOutput = compressedLayerModel.predict(example)
    print(compressedLayerOutput)


'''Ya toca organizar esto mejor pero el siguiente experimento es para ver como el tamaño del botleneck afecta la perdida de informacion y como afecta el tamaño de compresion'''
def experiment2AE():
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
    

if __name__ == '__main__':
    experiment2AE()