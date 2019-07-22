'''VAE and unsupervised clustering based on keras and tensorflow

 #modified from
  https://github.com/keras-team/keras/blob/master/examples/variational_autoencoder.py
'''
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import os
import sys
sys.path.append("..")
import scipy.io as sio 

import keras
from keras.layers import Dense, Input, Layer
from keras.layers import Conv2D, Flatten, Lambda, Dropout, Softmax
from keras.layers import Reshape, Conv2DTranspose, LeakyReLU
from keras.models import Model
from keras.utils import plot_model
from keras import backend as K
import imageio,os
from keras.datasets import cifar10
from keras.utils import to_categorical

from matplotlib import pyplot
import tensorflow as tf
# from keras.datasets import fashion_mnist as mnist

batch_size = 100
img_dim = 32
channel_dim = 3 # for Cifar channel dim is 3 while for mnist and shvn channel_dim is 1
latent_dim = 20 # hidden variable z, with dim=2 for ploting, here we set it as 20
intermediate_dim = 64
epochs = 20
epsilon_std = 1.0
num_classes = 10
filters = 16

# loading dataset
(x_train, y_train_), (x_test, y_test_) = cifar10.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

x_train = x_train.reshape((-1, img_dim, img_dim, channel_dim))
x_test = x_test.reshape((-1, img_dim, img_dim, channel_dim))

y_train = to_categorical(y_train_, num_classes)
y_test = to_categorical(y_test_, num_classes)

###############################################################
# VAE model = encoder + decoder
# build encoder model
x = Input(shape=(img_dim, img_dim, channel_dim), name='encoder_input')
h = x

for i in range(3):
    filters *= 2
    # h = Conv2D(filters=filters,
    #            kernel_size=3,
    #            activation= None,
    #            strides=1,
    #            padding='same')(h)
    # h = LeakyReLU(0.2)(h)
    #h = Dropout(0.5)(h)
    h = Conv2D(filters=filters,
               kernel_size=3,
               strides=2,
               padding='same')(h)
    h = LeakyReLU(0.2)(h)
    #h = Dropout(0.5)(h)

# shape info needed to build decoder model
h_shape = K.int_shape(h)[1:]
h = Flatten()(h)
z_mean = Dense(latent_dim, name='z_mean')(h) # calculate the postp mean of Q(z|x)
z_log_var = Dense(latent_dim, name='z_log_var')(h) # calculate the log(postp std) of Q(z|x)

#instantiate the Encoder
encoder = Model(x, [z_mean,z_log_var], name='encoder') 
encoder.summary()
plot_model(encoder, to_file='cifar_cvae_cnn_encoder.png', show_shapes=True)

###############################################################
z = Input(shape=(latent_dim,), name='z_sampling')
h = z
h = Dense(np.prod(h_shape))(h)
h = Reshape(h_shape)(h)

for i in range(3):
    h = Conv2DTranspose(filters=filters,
                        kernel_size=3,
                        activation= None,
                        strides=2,
                        padding='same')(h)
    h = LeakyReLU(0.2)(h)
    #h = Dropout(0.5)(h)
    # if i < 2:
    #     h = Conv2DTranspose(filters=filters,
    #                         kernel_size=3,
    #                         activation= None,
    #                         strides=1,
    #                         padding='same')(h)
    #     h = LeakyReLU(0.2)(h)
        #h = Dropout(0.5)(h)
    # else:
    if i == 2:
        x_recon = Conv2DTranspose(filters=channel_dim,
                          kernel_size=3,
                          activation='sigmoid',
                          padding='same',
                          name='decoder_output')(h)
    filters //= 2


# instantiate decoder model
decoder = Model(z, x_recon, name='decoder')
decoder.summary()
plot_model(decoder, to_file='cifar_cvae_cnn_decoder.png', show_shapes=True)
generator = decoder

###############################################################
#z = Input(shape=(latent_dim,), name='z_sampling')
y = Dense(intermediate_dim, activation='relu')(z)
#y = Dropout(0.5)(y)
y = Dense(num_classes)(y)
y = Softmax()(y)

classfier = Model(z, y, name='classifier') # classifier of hidden variable
classfier.summary()
plot_model(classfier, to_file='cifar_cvae_cnn_classfier.png', show_shapes=True)

###############################################################
y_x = Input(shape=(num_classes,)) # class y_x of input x
yh = Dense(latent_dim)(y_x) # mean of class
class_mean_estimator = Model(y_x, yh, name='class_mean_estimator')
class_mean_estimator.summary()
plot_model(class_mean_estimator, to_file='cifar_cvae_class_mean_estimator.png', show_shapes=True)
###############################################################
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim))
    return z_mean + K.exp(z_log_var / 2) * epsilon

z_mean_v,z_log_var_v = encoder(x)
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean_v, z_log_var_v])
x_recon = decoder(z)
y = classfier(z)
yh_v = class_mean_estimator(y_x)

# instantiate VAE model
vae = Model([x,y_x], [x_recon, y, yh_v], name='CVAE')
vae.summary()
plot_model(vae, to_file='cifar_cvae_cnn.png', show_shapes=True)


# define loss
lamb = 2.5 # weight of recon
xent_loss = 0.5 * K.sum(K.mean((x - x_recon)**2, 0))
kl_loss = - 0.5 * K.mean(K.sum(1 + z_log_var - K.square(z_mean - yh) - K.exp(z_log_var), axis=-1))
crossentropy_loss = K.mean(K.categorical_crossentropy(y_x,y),0)
vae_loss = 1*xent_loss + 1*kl_loss + 10*crossentropy_loss


vae.add_loss(vae_loss)
vae.compile(optimizer='adam')

checkpointer = keras.callbacks.ModelCheckpoint(
            filepath = './models/checkpoints/checkpoint.h5', 
            monitor='val_loss', 
            verbose=1, 
            save_best_only=True, 
            save_weights_only=False, 
            mode='min', 
            period=1
)
lrate = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.2, 
            patience=2, 
            verbose=0, 
            mode='min', 
            epsilon=0.0001, 
            cooldown=0, 
            min_lr=0
)

history = vae.fit([x_train,y_train],
            shuffle=True,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[checkpointer, lrate],
            validation_data=([x_test, y_test], None))

	
# list all data in history
print(history.history.keys())
# plot metrics
pyplot.plot(history.history['loss'])
pyplot.plot(history.history['val_loss'])
# pyplot.plot(history.history[xent_loss])
# pyplot.plot(history.history[kl_loss])
# pyplot.plot(history.history[crossentropy_loss])
pyplot.title('model loss')
pyplot.ylabel('loss')
pyplot.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
pyplot.show()


means = class_mean_estimator.predict(to_categorical(range(num_classes), num_classes))
x_train_encoded,_ = encoder.predict(x_train)
y_train_pred = classfier.predict(x_train_encoded).argmax(axis=1)
x_test_encoded,_ = encoder.predict(x_test)
y_test_pred = classfier.predict(x_test_encoded).argmax(axis=1)

###################save encoded
adict2 = {}
adict2['x_train_encoded'] = x_train_encoded
adict2['x_test_encoded'] = x_test_encoded
adict2['y_train_pred'] = y_train_pred
adict2['y_test_pred'] = y_test_pred
adict2['x_train'] = x_train
adict2['y_train'] = y_train_
sio.savemat('../data/Encoded_CIFAR.mat',adict2)

#save trained model
if not os.path.exists('models'):
    os.mkdir('models')

encoder.save('models/cifar_cvae_encoder.h5')
print('Model encoder saved')
class_mean_estimator.save('models/cifar_cvae_class_mean_estimator.h5')
print('Model class_mean_estimator saved')
decoder.save('models/cifar_cvae_decoder.h5')
print('Model decoder saved')
classfier.save('models/cifar_cvae_classfier.h5')
print('Model classfier saved')
vae.save('models/cifar_cvae.h5')
print('Model vae saved')

def class_sample(path, category=0):
    """observe the samples within same cluster
    """
    n = 8
    figure = np.zeros((img_dim * n, img_dim * n,channel_dim))
    idxs = np.where(y_train_pred == category)[0]
    for i in range(n):
        for j in range(n):
            digit = x_train[np.random.choice(idxs)]
            digit = digit.reshape((img_dim, img_dim,channel_dim))
            figure[i * img_dim: (i + 1) * img_dim,
            j * img_dim: (j + 1) * img_dim,:] = digit
    imageio.imwrite(path, figure * 255)


def random_sample(path, category=0, std=1):
    """random generating based on cluster results
    """
    n = 8
    figure = np.zeros((img_dim * n, img_dim * n,channel_dim))
    for i in range(n):
        for j in range(n):
            noise_shape = (1, latent_dim)
            z_sample = np.array(np.random.randn(*noise_shape)) * std + means[category]
            x_recon = generator.predict(z_sample)
            digit = x_recon[0].reshape((img_dim, img_dim,channel_dim))
            figure[i * img_dim: (i + 1) * img_dim,
            j * img_dim: (j + 1) * img_dim,:] = digit
    imageio.imwrite(path, figure * 255)


if not os.path.exists('samples'):
    os.mkdir('samples')

for i in range(10):
    class_sample(u'samples/classsample_%s.png' % i, i)
    random_sample(u'samples/random_sample_%s.png' % i, i)


right = 0.
right = (y_train_pred==y_train_.squeeze(1)).sum().astype(float)
print ('train acc: %s' % (right / len(y_train_)))


right = 0.
right = (y_test_pred==y_test_.squeeze(1)).sum().astype(float)
print ('train acc: %s' % (right / len(y_test_)))



