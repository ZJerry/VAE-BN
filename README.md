# VAE-BN

   This repository contains the codes for the paper 'Dual-Domain based Adversarial Defense with Conditional VAE and Bayesian Network', which has been submitted to IEEE Transactions on Indistrial Informatics. 
     
## First train CVAE models
   
     python mnist_cvae.py
     python svhn_cvae.py
     python GTSRB_cvae.py  
     
   >There is a preprocessing to generate the data for GTSRB.
    
   Combine model to concatenate the **classifier** and **encoder**:
   
     python model_combine.py -d=mnist
     
   We combine two models (encoder+classifier) as classifier for subsequent attacks


## Use craft_adv_cvae.py to attack

     python craft_adv_cvae.py -d=gtsrb -a=fgsm
   
## Use Bayesian Network (in Matlab) for adversary detection and diagnosis

   One should install the Bayesian network toolbox for Matlab, which have been provided in the repository. 
   One can refer to https://github.com/bayesnet/bnt for how to use the Bayes Net Toolbox.
   There are two ways to run our code, either use the pretrained models and reproduce the results in the paper, or train the network.
   
### Use the pretrained model to reproduce the results

   After the install of toolbox, you can directly run the following codes.
   
     BN_MNIST.m
     BN_SVHN.m
     BN_GTSRB.m
     
### Train a network by yourself

   One can also train a network by simply uncommenting the lines in the scripts, as follows:
   
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

   In this repo, the Bayesian network is realized in Matlab to demonstrate our idea, in the long run, we will attempts to write the scripts in the python;
   We will also consider how to improve the reclassfication function to make it much faster.
   
