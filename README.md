# VAE-BN

## Descriptions

   This repository contains the codes for reproducing the paper **Dual-Domain based Adversarial Defense with Conditional VAE and Bayesian Network**, which has been submitted to IEEE Transactions on Indistrial Informatics. 
     
   The code has been organized in four parts:
   
   * The training of CVAE models
   * Craft attacks
   * Make the Bayesian network to detection and diagnosis
   * Use the reclassification module to make the reform and reclassificaion
   
## Train CVAE models
   
     python mnist_cvae.py
     python svhn_cvae.py
     python GTSRB_cvae.py  
     
   >**Note**: There is a preprocessing to generate the data for GTSRB.
    
   Combine model to concatenate the **classifier** and **encoder**:
   
     python model_combine.py -d=mnist
     
   We combine two models (encoder+classifier) as classifier for subsequent attacks


## Use craft_adv_cvae.py to attack

     python craft_adv_cvae.py -d=gtsrb -a=fgsm
   
## Use Bayesian Network (in Matlab) for adversary detection and diagnosis

   One should install the Bayesian network toolbox for Matlab, which have been provided in the following google drive. 
   One can refer to https://github.com/bayesnet/bnt for how to use the toolbox.
   There are two ways to run our code, either use the pretrained models and reproduce the results in the paper, or train the network.
   
  **Note**: All the data, Bayesian net toolbox and Bayesian network models used in this work can be downloaded at https://drive.google.com/drive/folders/17_ZCNZZpBbiGawmBt3nal3lSFACOtZVD 
  
  **Note**: After downloading, please move the *.mat files into the data folder.
   
### Use the pretrained model to reproduce the results

   After the install of toolbox, you can directly run the following codes. If you want to reproduce the results, please first download the adversary data.
   
     BN_MNIST.m
     BN_SVHN.m
     BN_GTSRB.m
     
### Train a network by yourself

   One can also train a network by simply uncommenting the lines in the above scripts, as follows:
   
     %===> Uncomment the following line for training
     % [bnet2, ll, engine2] = learn_params_em(engine,training,maxiter,epsilon);
     %===> or use a pretrained Bayesian network
     % load bnet2_mnist_m2_com
   
## Reclassification

   I think I have modified the reclassification.py to inlcude the following line 291
    
     if detected[:,i] == 1: 
    
   So as to make sure that only successfully detected samples can be used for reclassification.
   
   Also mention that line 234-237, only 1000 samples have been used in one run. (Don't remember why...)
   
     X_test_adv = X_test_adv0[0:1000,:,:]
     Y = Y0[0:1000,:]
     Y_adv = Y_adv0[0:1000,:]
     detected = detected0[:,0:1000]

# TODO

   We will attempt to convert the Bayesian network scripts in the python or write a middleware between the two modules;
   We will also consider how to improve the reclassfication function to make it much faster.
   
   * 攻击后数据保存是否可以直接matlab调用
   * Upload and Download the dataset
   * 验证attack和reclassification
