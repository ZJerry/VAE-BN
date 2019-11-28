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


##################### Get pretrained model files from Google Drive

#gdown --id xxx --output ../models/svhn_cvae_encoder.h5

#gdown --id xxx --output ../models/svhn_cvae_decoder.h5

#gdown --id xxx --output ../models/svhn_cvae_classifier.h5

#gdown --id xxx --output ../models/svhn_cvae_class_mean_estimator.h5

#gdown --id xxx --output ../models/svhn_cvae_c.h5