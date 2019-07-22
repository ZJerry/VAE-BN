# VAE-BN
## First train models
   
     python mnist_cvae.py
     python svhn_cvae.py
     python GTSRB_cvae.py  
     
   >There is a preprocessing to generate the data for GTSRB.
    
   Combine model to concatenate the **classifier** and **encoder**:
   
     python model_combine.py -d=mnist
     
   We combine two models so as to:
   
   - Craft attacks easily
   - Make a lot of funs

## Use craft_adv_cvae.py to attack

     python craft_attack_cvae.py -d=gtsrb -a=fgsm
     
   Please mention that in line 154-155, I only use the first 5000 samples
   
     craft_one_type(sess, model, X_test[:5000], Y_test[:5000], args.dataset, args.attack,args.batch_size)

## Reclassification

   I think I have modified the reclassification.py to inlcude the following line 291
    
     if detected[:,i] == 1: 
    
   So as to make sure that only successfully detected samples can be used for reclassification.
   
   Also mention that line 234-237, only 1000 samples have been used in one run. (Don't remember why...)
   
     X_test_adv = X_test_adv0[0:1000,:,:]
     Y = Y0[0:1000,:]
     Y_adv = Y_adv0[0:1000,:]
     detected = detected0[:,0:1000]

   TODO
   
   |             |Attack          |Defense        |
   |-------------|Have fun?       |Not fun        |
   |Yes          |V               |  O            |
   |NO           |O               |  V            |
