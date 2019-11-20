from __future__ import division, absolute_import, print_function

import os
import argparse
import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.models import load_model, Model
from keras.layers import Input,Softmax

import sys
sys.path.append("..")

# from keras.utils import plot_model

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

def main(args):
    assert args.dataset in ['mnist', 'cifar', 'svhn','gtsrb'], \
        "Dataset parameter must be either 'mnist', 'cifar', 'gtsrb' or 'svhn'"
    assert os.path.isfile('../models/%s_cvae_encoder.h5'% args.dataset), \
        'model file not found...'
    assert os.path.isfile('../models/%s_cvae_classfier.h5'% args.dataset), \
        'model file not found...'


    # load the cvae classifier model
    encoder = load_model('../models/%s_cvae_encoder.h5'% args.dataset)
    classifier = load_model('../models/%s_cvae_classfier.h5'% args.dataset)
    classifier_pre = Model(input=classifier.input, output=classifier.get_layer(classifier.layers[-2].name).output)

    input_ = encoder.input
    h,_ = encoder(input_)
    g = classifier_pre(h)
    y = Softmax()(g)
            
    model = Model(input_, y)
    model.summary()
    # plot_model(model, to_file='craft_attack_model_%s.png'% args.dataset, show_shapes=True)

    model.save('../models/%s_cvae_c.h5'% args.dataset)
    print('Model vae saved')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--dataset',
        help="Dataset to use; either 'mnist', 'cifar','gtsrb' or 'svhn'",
        required=True, type=str
    )
    args = parser.parse_args()
    main(args)
