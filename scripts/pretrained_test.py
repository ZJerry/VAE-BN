
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
from keras.utils import to_categorical

from matplotlib import pyplot
import tensorflow as tf

import argparse

def import_model(args):
    assert args.dataset in ['mnist', 'svhn','gtsrb'], \
        "Dataset parameter must be either 'mnist', 'gtsrb' or 'svhn'"
    assert os.path.isfile('../models/%s_cvae_decoder.h5'% args.dataset), \
        'model file not found...'
    assert os.path.isfile('../models/%s_cvae_class_mean_estimator.h5'% args.dataset), \
        'model file not found...'
    if args.dataset == 'mnist' or args.dataset == 'svhn':
        num_classes = 10
    elif args.dataset == 'gtsrb':
        num_classes = 43

    # load the cvae classifier model
    decoder = load_model('../models/%s_cvae_decoder.h5'% args.dataset)

    for layer in decoder.layers:
        layer.trainable = False

    class_mean_estimator = load_model('../models/%s_cvae_class_mean_estimator.h5'% args.dataset)
    means = class_mean_estimator.predict(to_categorical(range(num_classes), num_classes))

    return decoder, means

def import_classifier(args):
    assert args.dataset in ['mnist', 'svhn','gtsrb'], \
        "Dataset parameter must be either 'mnist', 'gtsrb' or 'svhn'"
    assert os.path.isfile('../models/%s_cvae_classifier.h5'% args.dataset), \
        'model file not found...'
    # load the cvae classifier model
    classifier = load_model('../models/%s_cvae_classifier.h5'% args.dataset)

    for layer in classifier.layers:
        layer.trainable = False
    return classifier

def import_encoder(args):
    assert args.dataset in ['mnist', 'svhn','gtsrb'], \
        "Dataset parameter must be either 'mnist', 'gtsrb' or 'svhn'"
    assert os.path.isfile('../models/%s_cvae_encoder.h5'% args.dataset), \
        'model file not found...'
    # load the cvae classifier model
    encoder = load_model('../models/%s_cvae_encoder.h5'% args.dataset)

    for layer in encoder.layers:
        layer.trainable = False

    return encoder

def cal_acc(y_list):
    y_l_p = np.transpose(y_list)
    table = (y_l_p == y_l_p[-2])
    acc = table.sum(1)/float(table.shape[1])
    return acc

def main(args):
    if args.dataset == 'mnist':
        num_classes = 10
        latent_dim = 20
    elif args.dataset == 'svhn':
        num_classes = 10
        latent_dim = 20      
    elif args.dataset == 'gtsrb':
        num_classes = 43
        latent_dim = 32

    assert os.path.isfile('../data/Adv_%s_%s.mat' %
                          (args.dataset, args.attack)), \
        'adversarial sample file not found... must first craft adversarial ' \
        'samples using craft_adv_cave.py'
    assert os.path.isfile('../data/det_%s_%s_all.mat' %
                          (args.dataset, args.attack)), \
        'detected label file not found... must implement detection first ' 
    assert os.path.isfile('../data/Adv_%s_%s_r.mat' %
                          (args.dataset, args.attack)), \
        'recovered data file not found... must implement recovery first ' \
        'samples using reclassification.py'

    # Load data
    data = sio.loadmat('../data/Adv_%s_%s.mat' % (args.dataset, args.attack))
    data_det = sio.loadmat('../data/det_%s_%s_all.mat' % (args.dataset, args.attack))
    data_rec = sio.loadmat('../data/Adv_%s_%s_r.mat' % (args.dataset, args.attack))

    X_test =  data['X']
    X_test_adv =  data['X_adv']
    Y = data['Y']
    # Y_ = Y.argmax(1)
    Y_adv = data['Y_preds']
    # print(Y.shape,Y_.shape,Y_adv.shape)
    # print(data_det.keys())
    detected = data_det['detected']
    Y_r = data_rec['y_list']

    img_dim,channel_dim = X_test_adv.shape[-2:]
    img_num = min(X_test_adv.shape[0],detected.shape[1])

    decoder, means = import_model(args)
    classifier = import_classifier(args)
    encoder = import_encoder(args)

    print("Data set %s under %s attack:" % (args.dataset, args.attack))

    ##### pretrained model test acc
    if args.part.lower()=='a':
        x_test_encoded,_ = encoder.predict(X_test)
        y_test_pred = classifier.predict(x_test_encoded).argmax(axis=1)
        acc = float((y_test_pred == Y[:len(X_test)].argmax(1)).sum())/len(y_test_pred)
        print("   Classification Acc on the clean test set: %0.2f%%" % (100*acc))

    ##### Adversarial Acc on the test set
    if args.part.lower()=='b':
        acc = float((Y_adv.argmax(1) == Y[:len(X_test_adv)].argmax(1)).sum())/len(Y_adv)
        print("   Classification Acc on the adversarial test set: %0.2f%%" % (100*acc))
 
    #####reclassification acc
    if args.part.lower()=='c':
        print("   ** %s adversarial samples are detected and recovered **"%(len(Y_r)))
        #detection reate
        acc = float((detected == 1).sum())/detected.shape[1]
        print("   Comprehensive detection rate on the adversarials: %0.2f%%" % (100*acc))

    if args.part.lower()=='d':
        acc = cal_acc(Y_r)
        print("   ** %s adversarial samples are detected and recovered **"%(len(Y_r)))
        print("   Reclassification Acc without recovery: %0.2f%%"%(100*acc[5]))
        print("   Reclassification Acc with recovery strategy A: %0.2f%%"%(100*acc[0]))
        print("   Reclassification Acc with recovery strategy B(Reformer): %0.2f%%"%(100*acc[1]))
        print("   Reclassification Acc with recovery strategy C(Reformer+Decoder+Encoder): %0.2f%%"%(100*acc[2]))
        # print("Reclassification Acc with recovery strategy 4(Reformer+(Decoder+Encoder)*5): %0.2f%%"%(100*acc[3]))

    if args.part.lower()=='overall':
        x_test_encoded,_ = encoder.predict(X_test)
        y_test_pred = classifier.predict(x_test_encoded).argmax(axis=1)
        acc = float((y_test_pred == Y[:len(X_test)].argmax(1)).sum())/len(y_test_pred)
        print("   Classification Acc on the clean test set: %0.2f%%" % (100*acc))

        acc = float((Y_adv.argmax(1) == Y[:len(X_test_adv)].argmax(1)).sum())/len(Y_adv)
        print("   Classification Acc on the adversarial test set: %0.2f%%" % (100*acc))

        print("   ** %s adversarial samples are detected and recovered **"%(len(Y_r)))
        #detection reate
        acc = float((detected == 1).sum())/detected.shape[1]
        print("   Comprehensive detection rate on the adversarials: %0.2f%%" % (100*acc)) 

        acc = cal_acc(Y_r)
        print("   Reclassification Acc without recovery: %0.2f%%"%(100*acc[5]))
        print("   Reclassification Acc with recovery strategy A: %0.2f%%"%(100*acc[0]))
        print("   Reclassification Acc with recovery strategy B(Reformer): %0.2f%%"%(100*acc[1]))
        print("   Reclassification Acc with recovery strategy C(Reformer+Decoder+Encoder): %0.2f%%"%(100*acc[2]))      
# y_test_rec_pred = classifier.predict(x_rec_encoded).argmax(axis=1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--dataset',
        help="Dataset to use; either 'mnist', 'cifar','gtsrb' or 'svhn'",
        required=False, type=str
    )
    parser.add_argument(
        '-a', '--attack',
        help="Attack to use; either 'fgsm', 'bim-a', 'bim-b', 'jsma' 'cw' "
             "or 'all'",
        required=False, type=str
    )
    parser.add_argument(
        '-p', '--part',
        help="Parts of reproduce pipeline; 'A', 'B', 'C', 'D' or 'overall'",
        required=False, type=str
    )
    parser.set_defaults(dataset='gtsrb')
    parser.set_defaults(attack='fgsm')
    parser.set_defaults(part='overall')
    args = parser.parse_args()
    main(args)

