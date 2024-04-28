import tensorflow as tf
from keras import layers, losses, Model, callbacks

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
        encoded = self.encoder(data)
        return encoded

    def getDecoded(self, data):
        decoded = self.decoder(data)
        return decoded