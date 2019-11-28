#!/bin/bash

##################### Get pretrained data files from Google Drive

#### pretrained adversarial data #####
gdown --id 16mFnxy_ZN0yZWY7sl2G28xqdAd0vf2eJ --output ../data/Adv_mnist_fgsm.mat
gdown --id 1t2ehe9UjosJmqmkUkYsHCQeysywMOXFp --output ../data/Adv_mnist_cw.mat
gdown --id 19fcHP3S7ewKYDpE4UtLDl06Cg9Uerc2d --output ../data/Adv_mnist_bim-a.mat

#### pretrained parameters for detection model ####
gdown --id 1H6ZL6-IEttnomabPBiUFARKFgCy8gSDY --output ../data/bnet2_mnist_m2_com.mat

#### pretrained detection results (three files for fgsm/cw/bim-a respectively) ####
gdown --id 1CqLmQFYrtJ4hjo_iC8hFy3osXglthLEJ --output ../data/det_mnist_fgsm_all.mat
gdown --id 1cHFrBTDU9iC8MavZWveSfV_016u9YnlK --output ../data/det_mnist_cw_all.mat
gdown --id 1canf60Ut7uCwViZ72RYlM3erew1iqbkj --output ../data/det_mnist_bim-a_all.mat

#### clean data for unsupervised defense training ####
gdown --id 1T1S1GYAGhVOMSR2ZOj5lovkwOzYj0KH8 --output ../data/Encoded_MNIST.mat


##################### Get pretrained model files from Google Drive

#gdown --id xxx --output ../models/mnist_cvae_encoder.h5

#gdown --id xxx --output ../models/mnist_cvae_decoder.h5

#gdown --id xxx --output ../models/mnist_cvae_classifier.h5

#gdown --id xxx --output ../models/mnist_cvae_class_mean_estimator.h5

#gdown --id xxx --output ../models/mnist_cvae_c.h5