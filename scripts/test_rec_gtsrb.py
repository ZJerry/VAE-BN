
import numpy as np
from skimage import io, color, exposure, transform
#from sklearn.cross_validation import train_test_split
import os
import sys
sys.path.append("..")
import scipy.io as sio 
import glob
import h5py

from keras.preprocessing.image import ImageDataGenerator
import keras
from keras.layers import Dense, Input, Layer
from keras.layers import Conv2D, Flatten, Lambda, Dropout, Softmax
from keras.layers import Reshape, Conv2DTranspose, LeakyReLU
from keras.models import Model
from keras.utils import plot_model
from keras import backend as K
import imageio,os
from keras.models import load_model
#from keras.utils import np_utils
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.utils import to_categorical

from matplotlib import pyplot
import tensorflow as tf

num_classes = NUM_CLASSES = 43
img_dim = IMG_SIZE = 48
batch_size = 100
channel_dim = 3 # for Cifar channel dim is 3 while for mnist and shvn channel_dim is 1
latent_dim = 32 # hidden variable z, with dim=2 for ploting, here we set it as 20
intermediate_dim = 64
epochs = 20
epsilon_std = 1.0
filters = 16

with  h5py.File('../data/X.h5') as hf: 
    X, Y = hf['imgs'][:], hf['labels'][:]
print("Loaded images from X.h5")
print(X.shape,np.max(X),Y.shape)


with  h5py.File('../data/X_test.h5') as hf: 
    x_test, y_test_ = hf['imgs'][:], hf['labels'][:]
print("Loaded images from X_test.h5")

x_train = X
y_train_ = Y
y_train = to_categorical(y_train_, num_classes)
y_test = to_categorical(y_test_, num_classes)

###########define cvae model and optimizer/loss

###############################################################
# VAE model = encoder + decoder
# build encoder model
encoder = load_model('../models/gtsrb_cvae_encoder.h5')
decoder = load_model('../models/gtsrb_cvae_decoder.h5')
classifier = load_model('../models/gtsrb_cvae_classfier.h5')

x_train_encoded,_ = encoder.predict(x_train)
y_train_pred = classifier.predict(x_train_encoded).argmax(axis=1)
x_test_encoded,_ = encoder.predict(x_test)
y_test_pred = classifier.predict(x_test_encoded).argmax(axis=1)
x_train_recon = decoder.predict(x_train_encoded)
x_test_recon = decoder.predict(x_test_encoded)

acc = float((y_test_pred == np.array(y_test[:len(x_test)]).argmax(1)).sum())/len(y_test_pred)
print("Accuracy on the test set: %0.2f%%" % (100*acc))

right = 0.
right = (y_train_pred==y_train_).sum().astype(float)
print ('train acc: %s' % (right / len(y_train_)))


right = 0.
right = (y_test_pred==y_test_).sum().astype(float)
print ('test acc: %s' % (right / len(y_test_)))

x_test_rec = sio.loadmat('../data/Rec_gtsrb_cw.mat')
test_rec = x_test_rec['x_test_new']
x_rec_encoded,_ = encoder.predict(test_rec)
y_test_rec_pred = classifier.predict(x_rec_encoded).argmax(axis=1)
acc = float((y_test_rec_pred == np.array(y_test[:len(test_rec)]).argmax(1)).sum())/len(y_test_rec_pred)
print("Accuracy on the recovered test set: %0.2f%%" % (100*acc))


#####
x_test_org = sio.loadmat('../data/Adv_gtsrb_cw.mat')
x_test_enc = x_test_org['X_adv_encoder']
# x_rec_encoded,_ = encoder.predict(x_test_enc)
y_test_rec_pred = classifier.predict(x_test_enc).argmax(axis=1)
acc = float((y_test_rec_pred == np.array(y_test[:len(test_rec)]).argmax(1)).sum())/len(y_test_rec_pred)
print("Accuracy on the adv test set: %0.2f%%" % (100*acc))
###################save encoded
# adict2 = {}
# adict2['x_rec_encoded'] = x_rec_encoded
# sio.savemat('../data/Encoded_rec_svhn.mat',adict2)