from __future__ import division, absolute_import, print_function

import os
import argparse
import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.models import load_model, Model
from keras.layers import Input

import sys
sys.path.append("..")
 
import scipy.io as sio

from attacks import (fast_gradient_sign_method, basic_iterative_method,
                            saliency_map_method,CWL2)
from keras.datasets import cifar10
from keras.datasets import mnist
from keras.utils import to_categorical

from keras.utils import plot_model

import h5py

#FGSM & BIM attack parameters that were chosen
ATTACK_PARAMS = {
    'mnist': {'eps': 0.300, 'eps_iter': 0.010},
    'cifar': {'eps': 0.050, 'eps_iter': 0.005},
    'svhn': {'eps': 0.130, 'eps_iter': 0.010},
    'gtsrb': {'eps': 0.070, 'eps_iter': 0.005}
}



def craft_one_type(sess, model, X, Y, dataset, attack, batch_size):
    """
    TODO
    :param sess:
    :param model:
    :param X:
    :param Y:
    :param dataset:
    :param attack:
    :param batch_size:
    :return:
    """

    if attack == 'fgsm':
        # FGSM attack
        print(X.shape, Y.shape) #(?,33) (?,6)
        print('Crafting fgsm adversarial samples...')
        X_adv = fast_gradient_sign_method(
            sess, model, X, Y, eps=ATTACK_PARAMS[dataset]['eps'], clip_min=0.,
            clip_max=1., batch_size=batch_size
        )
    elif attack in ['bim-a', 'bim-b']:
        # BIM attack
        print('Crafting %s adversarial samples...' % attack)
        its, results = basic_iterative_method(
            sess, model, X, Y, eps=ATTACK_PARAMS[dataset]['eps'],
            eps_iter=ATTACK_PARAMS[dataset]['eps_iter'], clip_min=0.,
            clip_max=1., batch_size=batch_size
        )
        if attack == 'bim-a':
            # BIM-A
            # For each sample, select the time step where that sample first
            # became misclassified
            X_adv = np.asarray([results[its[i], i] for i in range(len(Y))])
        else:
            # BIM-B
            # For each sample, select the very last time step
            X_adv = results[-1]
    elif attack == 'jsma':
        # JSMA attack
        print('Crafting jsma adversarial samples. This may take a while...')
        X_adv = saliency_map_method(
            sess, model, X, Y, theta=1, gamma=0.1, clip_min=0., clip_max=1.
        )
    else:
        # CW attack
        X_adv = CWL2(sess, model, X, Y,targeted=True, batch_size=batch_size, max_iterations=1000, confidence=0)


    #evaluate the data after implementing an attack
    y_ = Y.argmax(1)
    y_test_pred = model.predict(X_adv).argmax(axis=1)
    y_ = y_[:len(y_test_pred)]
    right = 0.
    right = (y_test_pred==y_).sum().astype(float)
    print ('test acc: %0.2f%%' % (100*float(right) / len(y_)))

    for i in range(Y.shape[1]):
        acc = (y_[y_test_pred==y_]==i).sum().astype(float)/(y_==i).sum()
        print('fot testing dataset, the acc of class %s is  %0.2f%%'%(i, 100*acc))

    #start cal L2 distoration
    dis_l2 = 0
    for i in range(len(X_adv)):
        dis_l2 += np.sum((X_adv[i]-X[i])**2)**.5
    dis_l2 /= len(X_adv)
    print("Total distortion:", dis_l2)
    np.save('../data/Adv_%s.npy' % (args.attack), X_adv)
    Y_preds = model.predict(X_adv)
    model_encoder = model.layers[1]
    X_encoder,_ = model_encoder.predict(X)
    X_adv_encoder,_ = model_encoder.predict(X_adv)
    adict = {}
    adict['X'] = X
    adict['X_encoder'] = X_encoder
    adict['Y'] = Y
    adict['X_adv'] = X_adv
    adict['X_adv_encoder'] = X_adv_encoder
    adict['Y_preds'] = Y_preds
    sio.savemat('../data/Adv_%s_%s.mat'%(args.dataset,args.attack),adict)

def main(args):
    # loading dataset
    if args.dataset == 'cifar':
        num_classes = 10
        img_dim = 32
        channel_dim = 3 # for Cifar channel dim is 3 while for mnist and shvn channel_dim is 1
        (x_train, y_train_), (x_test, y_test_) = cifar10.load_data()

        x_train = x_train.astype('float32') / 255.
        x_test = x_test.astype('float32') / 255.

        x_train = x_train.reshape((-1, img_dim, img_dim, channel_dim))
        x_test = x_test.reshape((-1, img_dim, img_dim, channel_dim))

        y_train = to_categorical(y_train_, num_classes)
        y_test = to_categorical(y_test_, num_classes)

        y_train_ = y_train_.squeeze(1)
        y_test_ = y_test_.squeeze(1)

    elif args.dataset == 'mnist':
        num_classes = 10
        img_dim = 28
        channel_dim = 1 # for Cifar channel dim is 3 while for mnist and shvn channel_dim is 1      
        (x_train, y_train_), (x_test, y_test_) = mnist.load_data()

        x_train = x_train.astype('float32') / 255.
        x_test = x_test.astype('float32') / 255.

        x_train = x_train.reshape((-1, img_dim, img_dim, channel_dim))
        x_test = x_test.reshape((-1, img_dim, img_dim, channel_dim))

        y_train = to_categorical(y_train_, num_classes)
        y_test = to_categorical(y_test_, num_classes)

        y_train_ = y_train_.squeeze(1)
        y_test_ = y_test_.squeeze(1)

    elif args.dataset == 'gtsrb':
        num_classes = 43
        img_dim = 48
        channel_dim = 3 # for Cifar channel dim is 3 while for mnist and shvn channel_dim is 1      
        with  h5py.File('../data/X.h5') as hf: 
            X, Y = hf['imgs'][:], hf['labels'][:]
            print("Loaded images from X.h5")

        with  h5py.File('../data/X_test.h5') as hf: 
            x_test, y_test_ = hf['imgs'][:], hf['labels'][:]
            print("Loaded images from X_test.h5")

        x_train = X
        y_train_ = Y
        y_train = to_categorical(y_train_, num_classes)
        y_test = to_categorical(y_test_, num_classes)

    assert args.attack in ['fgsm', 'bim-a', 'bim-b', 'jsma', 'cw', 'all'], \
        "Attack parameter must be either 'fgsm', 'bim-a', 'bim-b', " \
        "'jsma' or 'cw'"
    assert os.path.isfile('../models/cifar_cvae_encoder.h5'), \
        'model file not found...'
    assert os.path.isfile('../models/cifar_cvae_classfier.h5'), \
        'model file not found...'

    # Create TF session, set it as Keras backend
    sess = tf.Session()
    K.set_session(sess)
    K.set_learning_phase(0)

    # load the cvae classifier model
    encoder = load_model('../models/%s_cvae_encoder.h5'% args.dataset)
    classifier = load_model('../models/%s_cvae_classfier.h5'% args.dataset)
    input_ = Input(shape = encoder.input.shape[1:])
    h,_ = encoder(input_)
    y = classifier(h)
    model = Model(input_, y)
    model.summary()
    plot_model(model, to_file='%s_attack_model.png'% args.dataset, show_shapes=True)


    #evaluate the data before implementing an attack
    # x_test_encoded,_ = encoder.predict(x_test)
    # y_test_pred = classifier.predict(x_test_encoded).argmax(axis=1)
    y_test_pred = model.predict(x_test).argmax(axis=1)

    right = 0.
    right = (y_test_pred==y_test_).sum().astype(float)
    print ('test acc: %0.2f%%' % (100*float(right) / len(y_test_)))
    for i in range(num_classes):
        acc = (y_test_[y_test_pred==y_test_]==i).sum().astype(float)/(y_test_==i).sum()
        print('fot testing dataset, the acc of class %s is  %0.2f%%'%(i, 100*acc))

    if args.attack == 'all':
        # Cycle through all attacks
        for attack in ['fgsm', 'bim-a', 'bim-b', 'jsma', 'cw']:
            craft_one_type(sess, model, x_test, y_test, args.dataset, attack,
                           args.batch_size)
    else:
        # Craft one specific attack type
        craft_one_type(sess, model, x_test, y_test, args.dataset, args.attack,
                       args.batch_size)
    print('Adversarial samples crafted and saved to data/ subfolder.')
    sess.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--dataset',
        help="Dataset to use; either 'mnist', 'cifar','gtsrb' or 'svhn'",
        required=False, type=str
    )
    parser.add_argument(
        '-a', '--attack',
        help="Attack to use; either 'fgsm', 'bim-a', 'bim-b', 'cw' "
             "or 'all'",
        required=False, type=str
    )
    parser.add_argument(
        '-b', '--batch_size',
        help="The batch size to use for training.",
        required=False, type=int
    )
    parser.set_defaults(batch_size=250)
    parser.set_defaults(attack='bim-b')
    parser.set_defaults(dataset='cifar')
    args = parser.parse_args()
    main(args)