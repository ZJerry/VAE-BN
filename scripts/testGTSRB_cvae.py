
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


######import data from h5 file(without preprocessing)
def preprocess_img(img):
    # Histogram normalization in y
    hsv = color.rgb2hsv(img)
    hsv[:,:,2] = exposure.equalize_hist(hsv[:,:,2])
    img = color.hsv2rgb(hsv)

    # central scrop
    min_side = min(img.shape[:-1])
    centre = img.shape[0]//2, img.shape[1]//2
    img = img[centre[0]-min_side//2:centre[0]+min_side//2,
              centre[1]-min_side//2:centre[1]+min_side//2,
              :]

    # rescale to standard size
    img = transform.resize(img, (IMG_SIZE, IMG_SIZE))

    # roll color axis to axis 0
    img = np.rollaxis(img,-1)

    return img

def get_class(img_path):
    return int(img_path.split('/')[-2])

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
###################save encoded
adict2 = {}
adict2['x_train_encoded'] = x_train_encoded
adict2['x_test_encoded'] = x_test_encoded
adict2['y_train_pred'] = y_train_pred
adict2['y_test_pred'] = y_test_pred
adict2['x_train'] = x_train
adict2['y_train'] = y_train_
adict2['x_test'] = x_test
adict2['y_test'] = y_test_
adict2['x_train_recon'] = x_train_recon
adict2['x_test_recon'] = x_test_recon
sio.savemat('../data/Encoded_GTSRB.mat',adict2)

#save trained model
if not os.path.exists('models'):
    os.mkdir('models')

# encoder.save('models/gtsrb_cvae_encoder.h5')
# print('Model encoder saved')
# class_mean_estimator.save('models/gtsrb_cvae_class_mean_estimator.h5')
# print('Model class_mean_estimator saved')
# decoder.save('models/gtsrb_cvae_decoder.h5')
# print('Model decoder saved')
# classfier.save('models/gtsrb_cvae_classfier.h5')
# print('Model classfier saved')
# vae.save('models/gtsrb_cvae.h5')
# print('Model vae saved')

right = 0.
right = (y_train_pred==y_train_).sum().astype(float)
print ('train acc: %s' % (right / len(y_train_)))


right = 0.
right = (y_test_pred==y_test_).sum().astype(float)
print ('test acc: %s' % (right / len(y_test_)))

