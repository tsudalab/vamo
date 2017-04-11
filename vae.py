from keras.datasets import mnist
from keras.optimizers import RMSprop
from keras import callbacks
from keras import objectives
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Input, Dense, Lambda, merge, Dropout
from keras.models import Model, Sequential
from keras import backend as K
from keras import metrics

import os
import shutil


class VAE(object):
    def __init__(self, xdim, zdim=2):
        input_x = Input(shape=(xdim, ))
        input_z = Input(shape=(zdim, ))
        
        h = Dense(100, input_shape=(xdim,), activation="relu")(input_x)
        h = Dropout(0.5)(h)
        
        z_mean = Dense(zdim, activation="linear")(h)
        z_log_var = Dense(zdim, activation="linear")(h)
        z = Lambda(self._sampling)([z_mean, z_log_var])
        
        decoder_h = Dense(100, activation="relu")
        d_decoder_h = Dropout(0.5)
        _decoder_h = Dense(xdim, activation="sigmoid")
        reconst_xmean = _decoder_h(d_decoder_h(decoder_h(z)))
        
        self.vae = Model(input_x, reconst_xmean)
        self.encoder = Model(input_x, z_mean)
        decoded_xmean = _decoder_h(d_decoder_h(decoder_h(input_z)))
        self.decoder = Model(input_z, decoded_xmean)
        
        def vae_loss(x, x_decoded_mean):
            loss = xdim * metrics.binary_crossentropy(x, x_decoded_mean)
            kl_loss=  - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
            return loss + kl_loss
        
        self.vae.compile(optimizer=RMSprop(lr=0.0001, rho=0.9, epsilon=1e-08, decay=0.), loss=vae_loss) 
    
    def fit(self, Xtrain, Xtest, max_epochs=100, batch_size=500):
        es_cb = EarlyStopping(monitor='val_loss', patience=50, verbose=0, mode='auto')
        cp_cb = ModelCheckpoint(filepath = os.path.join("checkpoint",'vae{epoch:02d}.hdf5'), 
                                                      monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
        cbks = [cp_cb, es_cb]
        
        if os.path.exists("checkpoint"):
            shutil.rmtree("checkpoint")
        os.mkdir("checkpoint")
        
        history = self.vae.fit(Xtrain, Xtrain, epochs=max_epochs, batch_size=batch_size, callbacks=cbks, shuffle = True,  validation_data=(Xtest, Xtest))
        
    
    def _sampling(self, args):
        mean, log_var = args
        epsilon = K.random_normal(K.shape(mean), mean=0.0, stddev=1.0)
        return mean + K.exp(log_var/2) * epsilon