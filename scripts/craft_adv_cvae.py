from __future__ import division, absolute_import, print_function

import os
import argparse
import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.models import load_model

import sys
sys.path.append("..")

import scipy.io as sio

from detect.util import get_data
from attacks import (fast_gradient_sign_method, basic_iterative_method,
                            saliency_map_method,CWL2)

import h5py
from keras.utils import to_categorical

# FGSM & BIM attack parameters that were chosen
ATTACK_PARAMS = {
    'mnist': {'eps': 0.250, 'eps_iter': 0.010},
    'cifar': {'eps': 0.050, 'eps_iter': 0.005},
    'svhn': {'eps': 0.050, 'eps_iter': 0.005},
    'gtsrb': {'eps': 0.070, 'eps_iter': 0.005}
}

# ATTACK_PARAMS = {
#     'mnist': {'eps': 0.250, 'eps_iter': 0.010},
#     'cifar': {'eps': 0.050, 'eps_iter': 0.005},
#     'svhn': {'eps': 0.050, 'eps_iter': 0.005},
#     'gtsrb': {'eps': 0.070, 'eps_iter': 0.005}
# }


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
        print(X.shape, Y.shape) #(10000, 28, 28, 1) (10000, 10)
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
    # _, acc = model.evaluate(X_adv, Y[:len(X_adv)], batch_size=batch_size,
    #                         verbose=0)
    y_p = model.predict(X_adv).argmax(1)
    acc =  float((y_p == np.array(Y[:len(X_adv)]).argmax(1)).sum())/len(y_p)
    print("Model accuracy on the adversarial test set: %0.2f%%" % (100 * acc))
    #start cal L2 distoration
    dis_l2 = 0
    for i in range(len(X_adv)):
        dis_l2 += np.sum((X_adv[i]-X[i])**2)**.5
    dis_l2 /= len(X_adv)
    print("Total distortion:", dis_l2)
    #np.save('../data/Adv_%s_%s.npy' % (args.dataset, args.attack), X_adv)
    Y_preds = model.predict(X_adv)
    model_encoder = model.layers[1]
    X_encoder,_ = model_encoder.predict(X)
    X_adv_encoder,_ = model_encoder.predict(X_adv)
    model_decoder = load_model('../models/%s_cvae_decoder.h5'% args.dataset)
    X_decoder= model_decoder.predict(X_encoder)
    X_adv_decoder = model_decoder.predict(X_adv_encoder)
    adict = {}
    adict['X'] = X
    adict['X_encoder'] = X_encoder
    adict['Y'] = Y
    adict['X_adv'] = X_adv
    adict['X_adv_encoder'] = X_adv_encoder
    adict['Y_preds'] = Y_preds
    adict['X_adv_decoder'] = X_adv_decoder
    adict['X_decoder'] = X_decoder
    sio.savemat('../data/Adv_%s_%s.mat'%(args.dataset,args.attack),adict)


def main(args):
    assert args.dataset in ['mnist', 'cifar', 'svhn','gtsrb'], \
        "Dataset parameter must be either 'mnist', 'cifar' or 'svhn'"
    assert args.attack in ['fgsm', 'bim-a', 'bim-b', 'jsma', 'cw', 'all'], \
        "Attack parameter must be either 'fgsm', 'bim-a', 'bim-b', " \
        "'jsma' or 'cw'"
    # assert os.path.isfile('../data/model_%s.h5' % args.dataset), \
    #     'model file not found... must first train model using train_model.py.'
    assert os.path.isfile('../models/%s_cvae_c.h5' % args.dataset), \
        'model file not found... must first train model using train_model.py.'    
    print('Dataset: %s. Attack: %s' % (args.dataset, args.attack))
    # Create TF session, set it as Keras backend
    sess = tf.Session()
    K.set_session(sess)
    K.set_learning_phase(0)
    model = load_model('../models/%s_cvae_c.h5' % args.dataset)
    print(model.layers)
    if args.dataset == 'cifar'or args.dataset =='mnist' or args.dataset =='svhn':
        _, _, X_test, Y_test = get_data(args.dataset)
    elif args.dataset == 'gtsrb':
        with  h5py.File('../data/X_test.h5') as hf: 
            X_test, Y_test_ = hf['imgs'][:], hf['labels'][:]
            print("Loaded images from X_test.h5")
        # X_test = X_test[:5000] 
        # Y_test_ = Y_test_[:5000] 
        Y_test = to_categorical(Y_test_, 43)       
    # _, acc = model.evaluate(X_test, Y_test, batch_size=args.batch_size,
    #                         verbose=0)
    y_p = model.predict(X_test).argmax(1)
    acc =  float((y_p == np.array(Y_test[:len(X_test)]).argmax(1)).sum())/len(y_p)
    print("Accuracy on the test set: %0.2f%%" % (100*acc))
    if args.attack == 'all':
        # Cycle through all attacks
        for attack in ['fgsm', 'bim-a', 'bim-b', 'jsma', 'cw']:
            craft_one_type(sess, model, X_test, Y_test, args.dataset, attack,
                           args.batch_size)
    else:
        # Craft one specific attack type
        craft_one_type(sess, model, X_test[:5000], Y_test[:5000], args.dataset, args.attack,
                       args.batch_size)
    print('Adversarial samples crafted and saved to data/ subfolder.')
    sess.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--dataset',
        help="Dataset to use; either 'mnist', 'cifar' or 'svhn'",
        required=True, type=str
    )
    parser.add_argument(
        '-a', '--attack',
        help="Attack to use; either 'fgsm', 'bim-a', 'bim-b', 'jsma', 'cw' "
             "or 'all'",
        required=True, type=str
    )
    parser.add_argument(
        '-b', '--batch_size',
        help="The batch size to use for training.",
        required=False, type=int
    )
    parser.set_defaults(batch_size=250)
    args = parser.parse_args()
    main(args)
