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
#### recovered data for reclassification
gdown --id 1Hhkrtkpv4zKOu1Lu0rfVngA18RoM-Gmq --output ../data/Adv_mnist_fgsm_r.mat
gdown --id 1M0TFKHhawqKFw2FVsT0Dug5VQpyv5XjP --output ../data/Adv_mnist_cw_r.mat
gdown --id 1Lv0p-Pdw2hJdRUHkDLWzO7tgDqVNMso_ --output ../data/Adv_mnist_bim-a_r.mat
#### 
gdown --id 1t5Lc0NTfuI617DmXLgvVXRzsLtWKaqZ4 --output ../data/mnist_normal_t2_all.mat
##################### Get pretrained model files from Google Drive
gdown --id 15zLUC5vJ7C_uRDYfo8IJQAjViGVtGHlT --output ../models/mnist_cvae.h5
gdown --id 1INB9zDFTaikHWeFVEA_E4QJuxKvoBjBV --output ../models/mnist_cvae_encoder.h5
gdown --id 1CfxBvyLxkb1L3dHYAeX1cOTdOmg0wzlp --output ../models/mnist_cvae_decoder.h5
gdown --id 1bZGC7ReIAB5ON5aBk6n1FlnKKmTTiB92 --output ../models/mnist_cvae_classifier.h5
gdown --id 1YRolOiY9bFV1wDWnPs7Ud7hfTsDgQGjx --output ../models/mnist_cvae_class_mean_estimator.h5
gdown --id 1nrciUBtZHtxXgUHvvqLVLsaiA-bxYerq --output ../models/mnist_cvae_c.h5
#####################remove the ^M(\r) in the name of generated files
for file in `ls ../data/*.mat?`;do mv $file `echo $file|sed 's/\.mat\r/\.mat/g'`;done;for file in `ls ../models/*.h5?`;do mv $file `echo $file|sed 's/\.h5\r/\.h5/g'`;done;