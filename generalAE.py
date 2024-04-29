import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras import losses,  callbacks

from utilities.aeArchitectures import *
from utilities.generatorDS import randomDataSetGenerate
from utilities.analysis import *

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