# VAE-BN
(1) First train models

(2) Use craft_adv_cvae to attack

(3) Reclassification
   I think I have modified the reclassification.py to inlcude the following line 291
    
       if detected[:,i] == 1: 
    
   So as to make sure that only successfully detected samples can be used for reclassification.
