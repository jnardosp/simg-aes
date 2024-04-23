import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from keras import layers, losses, Model, callbacks

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

    #Resultados
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
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

if __name__ == '__main__':
    experimentAE()