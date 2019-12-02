# VAE-BN

## Descriptions

   This repository contains the codes for reproducing the paper **Dual-Domain based Adversarial Defense with Conditional VAE and Bayesian Network**, which has been submitted to IEEE Transactions on Indistrial Informatics. 
     
   The code has been organized in four parts:
   
   * The training of CVAE models 
   * Craft attacks 
   * Make the Bayesian network to detection and diagnosis 
   * Use the recovery module to make the reform and reclassificaion 
   
   The pretrained models and data that produce the results presented in the paper are available in Google Drive https://drive.google.com/drive/folders/17_ZCNZZpBbiGawmBt3nal3lSFACOtZVD.
https://drive.google.com/drive/folders/143I5UdaC9B1AMP1-u23NV9Rz_Nne-HkP.
   One can simply run the following shell scripts to automatically download related models and data from the Google Drive to the specified path, which could avoid the manually download.
     bash Pretrained_mnist.sh
     bash Pretrained_svhn.sh
     bash Pretrained_gtsrb.sh  
   
## Train CVAE models
   To train a CVAE model on specified dataset, one could run the following script in the scripts folder.
   
     python mnist_cvae.py
     python svhn_cvae.py
     python GTSRB_cvae.py  
     
   >**Note**: There is a preprocessing to generate the data for GTSRB. Before running GTSRB_cvae.py, one should run GTSRB.py in the data folder first.
    
   Combine two sub-modules of CVAE (encoder+classifier) as the classifier for subsequent attacks:
   
     python model_combine.py -d=mnist(/svhn/gtsrb)
     
   >**Note**: If you want to reproduce the same results in the paper or simply skip the above two steps, please download the models files ending with '.h5' from aforementioned Google Drive and save them in the models folder.


## Craft an attack to generate adversarial data
   To fool the trained model and generate adversarial samples:

     python craft_adv_cvae.py -d=mnist(/svhn/gtsrb) -a=fgsm(/cw/bim-a/bim-b)
     
   The classification accuracy after the attack and the average adversarial distortion will be outputed by this script. The generated adversarial data will be saved in the '. . /data/Adv_%dataset_%attack.mat'. One could also directly download the data we provide from the Google Drive to reproduce the results in our paper.
   
## Use Bayesian Network for adversary detection and diagnosis

   One should install the Bayesian network toolbox for Matlab, which have been provided in the following google drive. 
   One can refer to https://github.com/bayesnet/bnt for how to use the toolbox.
   There are two ways to run our code, either use the pretrained models and reproduce the results in the paper, or train the network.
   
  **Note**: All the data, Bayesian net toolbox and Bayesian network models used in this work can be downloaded at https://drive.google.com/drive/folders/17_ZCNZZpBbiGawmBt3nal3lSFACOtZVD 
  
  **Note**: Please install the Bayesian net toolbox provided in our Google Drive, as we have modified several files in the toolbox.
  
  **Note**: For manual downloading, please save the *.mat files into the data folder.
   
### Use the pretrained model to reproduce the results

   After the install of toolbox, you can directly run the following codes in folder Detection and Diagnosis. If you want to reproduce the results, please make sure that you have prepared the adversary data by previous steps.
   
     BN_MNIST.m
     BN_SVHN.m
     BN_GTSRB.m
     
### Train a network by yourself

   One can also train a network by simply uncommenting the lines in the above scripts, as follows:
   
     %===> Uncomment the following line for training
     % [bnet2, ll, engine2] = learn_params_em(engine,training,maxiter,epsilon);
     %===> or use a pretrained Bayesian network
     % load bnet2_mnist_m2_com
   
## CVAE-based Recovery
   The 'recovery' for detected adversarial samples can be implemented by running the following code in the scripts folder. The restored classification accuracy will be outputed and the reformed samples will be saved in '. . /data/Adv_%d_%a_r.mat'.
   
     python reclassification.py -d=mnist(/svhn/gtsrb) -a=fgsm(/cw/bim-a/bim-b) 
     
   >**Note**: The script supports breakpoint resume from the exsisting 'data/Adv_%d_%a_r.mat' file. As the procedure takes a while for all samples, one can directly download the pretrained models and the requried data we provide to the folder 'models' and 'data' to get the same restored ACC as in Table III of the paper.
    
   One can visualize the original, adversarial, decoded (adversarial) and rcovered representations of a specified sample as Fig.7 in the paper shows. '-i/--index' is the index of the sample to visualize. It will also ouput the ground-truth label, adversarial prediction and recovered prediction of the specified sample.
     
     python visual_decoded.py -d=gtsrb -a=cw -i=128
   

# Contact
If you have any questions, please feel free to contact us at peng0086@e.ntu.edu.sg
