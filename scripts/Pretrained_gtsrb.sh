#!/bin/bash

##################### Get pretrained data files from Google Drive

#### pretrained adversarial data #####
gdown --id 1TrTl3zmTcAPSJD163U5C0-cBAqw-EdmS --output ../data/Adv_gtsrb_fgsm.mat
gdown --id 1Q_feADMTQ2X8VS-nV57tMhBhwRswGqSl --output ../data/Adv_gtsrb_cw.mat
gdown --id 1HrBjhGR1t3PVfm7LwORJayMZ3j5Zt6Xc --output ../data/Adv_gtsrb_bim-a.mat

#### pretrained parameters for detection model ####
gdown --id 1onbwxZsNThvhulSgpYmw71fYvO-OnUqy --output ../data/bnet2_gtsrb_m2_com.mat

#### pretrained detection results (three files for fgsm/cw/bim-a respectively) ####
gdown --id 1canf60Ut7uCwViZ72RYlM3erew1iqbkj --output ../data/det_gtsrb_fgsm_all.mat
gdown --id 1DxjXPbeIdzgIuVR_AwpE7y7HsavIn2PW --output ../data/det_gtsrb_cw_all.mat
gdown --id 1PRUNVbTof7ZT38oAM9ghEy8asxLqrtaf --output ../data/det_gtsrb_bim-a_all.mat

#### clean data for unsupervised defense training ####
gdown --id 1YOC5XnSdVhr3ViUHYb7StIHpgWai0-q8 --output ../data/Encoded_GTSRB.mat


##################### Get pretrained model files from Google Drive

#gdown --id xxx --output ../models/gtsrb_cvae_encoder.h5

#gdown --id xxx --output ../models/gtsrb_cvae_decoder.h5

#gdown --id xxx --output ../models/gtsrb_cvae_classifier.h5

#gdown --id xxx --output ../models/gtsrb_cvae_class_mean_estimator.h5

#gdown --id xxx --output ../models/gtsrb_cvae_c.h5