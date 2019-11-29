#!/bin/bash
##################### Get pretrained data files from Google Drive
#### pretrained adversarial data #####
gdown --id 17LvTRFYzIRDLQ98NXJfP4QX1ZzeYtadB --output ../data/Adv_svhn_fgsm.mat
gdown --id 11ZIc5GfrO6w8ksej79FHjncGH0DkFowv --output ../data/Adv_svhn_cw.mat
gdown --id 1t05jOjAIYBgRjLLvpS7XqeF7yz7rq07x --output ../data/Adv_svhn_bim-a.mat
#### pretrained parameters for detection model ####
gdown --id 1DtWx_U42GQH2T7ZYTpNzJM8RtAz3Ne4W --output ../data/bnet2_svhn_m2_com.mat
#### pretrained detection results (three files for fgsm/cw/bim-a respectively) ####
gdown --id 1RV7pAAG01o30n4rF0ptT7UMoafgA38Li --output ../data/det_svhn_fgsm_all.mat
gdown --id 1qdt1ixRvqwhbpS046MlgwnU7khIojoGQ --output ../data/det_svhn_cw_all.mat
gdown --id 12HJaduJ_fHFI1eEUkbyvvEXTR9-1j2UG --output ../data/det_svhn_bim-a_all.mat
#### clean data for unsupervised defense training ####
gdown --id 1cFwsd6Zo0kVLtuk_37yZEYhE4Y7z9eMf --output ../data/Encoded_SVHN.mat
#### recovered data for reclassification
gdown --id 1NXM-UNIIYBpyirqTCHc9zXR8NXIRKBXt --output ../data/Adv_svhn_fgsm_r.mat
gdown --id 1IGevKzKIAnY10YFi9R6rkqa25bFf7EAy --output ../data/Adv_svhn_cw_r.mat
gdown --id 1MTGCp-PSJ-HnbD9fOtuVFXRW8Nlgow83 --output ../data/Adv_svhn_bim-a_r.mat
#### 
gdown --id 1SQUPhrdKm_zYt42wIAo-T4HoY2GJRM1J --output ../data/svhn_normal_t2_all.mat
##################### Get pretrained model files from Google Drive
gdown --id 17WkfvsmEpAWFZAteEusl6qf43cbFSQA_ --output ../models/svhn_cvae.h5
gdown --id 1VhcgMvdA6s3xXZDjLhYZmrd3sD8lcfix --output ../models/svhn_cvae_encoder.h5
gdown --id 1Om42LXNu2ZTNk3HUDrLC5OyLZkF1GH6_ --output ../models/svhn_cvae_decoder.h5
gdown --id 1Rpme1Kx7u5VnAe381baV5sgBQQnhxnk8 --output ../models/svhn_cvae_classifier.h5
gdown --id 1s7o9jMAbI4sfqy6nttfB7xC4hPuII0Jx --output ../models/svhn_cvae_class_mean_estimator.h5
gdown --id 1PbsN2sotkrFRSxU1S12mMm0a4q6xw__m --output ../models/svhn_cvae_c.h5
#####################remove the ^M(\r) in the name of generated files
for file in `ls ../data/*.mat?`;do mv $file `echo $file|sed 's/\.mat\r/\.mat/g'`;done;for file in `ls ../models/*.h5?`;do mv $file `echo $file|sed 's/\.h5\r/\.h5/g'`;done;