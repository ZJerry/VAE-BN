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
#### recovered data for reclassification
gdown --id 1HtYwiU8C2oYwS3ifR-XtXd05k7MZGFJH --output ../data/Adv_gtsrb_fgsm_r.mat
gdown --id 1JR-7UQ6E_pC0N5iaucvQQNLE5wxsJmfZ --output ../data/Adv_gtsrb_cw_r.mat
gdown --id 1wGnjM_aHl3wObQQMmueBqXh44BHfoqVF --output ../data/Adv_gtsrb_bim-a_r.mat
#### 
gdown --id 1KIaovDjZdJOWqCChiPUIwlARFYHpP1V9 --output ../data/gtsrb_normal_t2_all.mat
##################### Get pretrained model files from Google Drive
gdown --id 1hC7uprPBg1sf5OcuhvoeEBemLEj0HsHc --output ../models/gtsrb_cvae.h5
gdown --id 1PvoNhbg7g8j-3HepGQ4vniz3jYTZGvTh --output ../models/gtsrb_cvae_encoder.h5
gdown --id 1CozfNfcV_M163AFsCEHZQL7DH43hEJu9 --output ../models/gtsrb_cvae_decoder.h5
gdown --id 1M4S9kTzicE2cLSbORqEAxwaP_VCb6sn6 --output ../models/gtsrb_cvae_classifier.h5
gdown --id 1mchyPl6AlSOE6m0obKOkE_hcDCKrKQ58 --output ../models/gtsrb_cvae_class_mean_estimator.h5
gdown --id 1VFKBN-_QEnewcCTOObJ4JQuzDQZj8mWB --output ../models/gtsrb_cvae_c.h5
#####################remove the ^M(\r) in the name of generated files
for file in `ls ../data/*.mat?`;do mv $file `echo $file|sed 's/\.mat\r/\.mat/g'`;done;for file in `ls ../models/*.h5?`;do mv $file `echo $file|sed 's/\.h5\r/\.h5/g'`;done;