#Me falta organizar esto 

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from keras import layers, losses
from keras.datasets import fashion_mnist
from keras.models import Model

tf.config.experimental_run_functions_eagerly(False) 

x_trainZip = (np.load('ZipTrainDataSet0.npy'))
x_testZip = np.load('ZipTestDataSet0.npy')
(x_train, _), (x_test, _) = fashion_mnist.load_data()
print(len(x_testZip))
print(len(x_test))
def custom_loss(target, generated):
    loss = tf.reduce_mean(tf.square(generated - target))
    return loss
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
latent_dim = 64
class Autoencoder(Model):
    def __init__(self, latent_dim):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential([
        layers.Dense(len(x_testZip[0]), activation='relu'),
        layers.Dense(80, activation='relu'),
        ])
        self.decoder = tf.keras.Sequential([
        layers.Dense(latent_dim, activation='sigmoid'),
        layers.Dense(784, activation='sigmoid'),
        layers.Reshape((28, 28))
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

autoencoder = Autoencoder(latent_dim)

autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())
history = None  

def custom_train_step(x, y):
    global history  
    with tf.device("/cpu:0"):
        try:
            history = autoencoder.fit(x_trainZip, x_train, epochs=50, shuffle=True)
        except ValueError:
            tf.keras.backend.set_value(tf.keras.backend.learning_phase(), 0)


custom_train_step(x_trainZip, x_train)
#--------------------------------------------------------------------------------------
encoded_imgs = autoencoder.encoder(x_trainZip).numpy()
original_imgs = x_test
np.save('original_imgs.npy', original_imgs)
np.save('encoded_imgs.npy', encoded_imgs) 

# #encoded_imgs = np.load('encoded_imgs.npy')
#--------------------------------------------------------------------------------------
# loss = history.history['loss']
# val_loss = history.history['val_loss']
# metadata = {'loss': loss, 'val_loss': val_loss}
# np.save('training_metadata.npy', metadata)
# #--------------------------------------------------------------------------------------

# loaded_metadata = np.load('training_metadata.npy', allow_pickle=True).item()


# loss = loaded_metadata['loss']
# val_loss = loaded_metadata['val_loss']


# plt.plot(loss, label='Training Loss')
# plt.plot(val_loss, label='Validation Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()
# #--------------------------------------------------------------------------------------

decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()
n = 10

plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i])
    plt.title("original")
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i])
    plt.title("reconstructed")
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()