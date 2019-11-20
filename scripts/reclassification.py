import numpy as np 
import keras
from keras.models import load_model,Model
from keras.utils import to_categorical
from keras import backend as K
from keras.layers import Layer,Input
import os
from os.path import isfile
import argparse
import scipy.io as sio
import sys
sys.path.append("..")

from matplotlib import pyplot
from keras.optimizers import Adam

import tensorflow as tf

import matplotlib.pyplot as plt

import time

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

def import_model(args):
    assert args.dataset in ['mnist', 'svhn','gtsrb'], \
        "Dataset parameter must be either 'mnist', 'gtsrb' or 'svhn'"
    assert os.path.isfile('../models/%s_cvae_decoder.h5'% args.dataset), \
        'model file not found...'
    assert os.path.isfile('../models/%s_cvae_class_mean_estimator.h5'% args.dataset), \
        'model file not found...'
    if args.dataset == 'mnist' or args.dataset == 'svhn':
        num_classes = 10
    elif args.dataset == 'gtsrb':
        num_classes = 43

    # load the cvae classifier model
    decoder = load_model('../models/%s_cvae_decoder.h5'% args.dataset)

    for layer in decoder.layers:
        layer.trainable = False

    class_mean_estimator = load_model('../models/%s_cvae_class_mean_estimator.h5'% args.dataset)
    means = class_mean_estimator.predict(to_categorical(range(num_classes), num_classes))

    return decoder, means

def import_classifier(args):
    assert args.dataset in ['mnist', 'svhn','gtsrb'], \
        "Dataset parameter must be either 'mnist', 'gtsrb' or 'svhn'"
    assert os.path.isfile('../models/%s_cvae_classifier.h5'% args.dataset), \
        'model file not found...'
    # load the cvae classifier model
    classifier = load_model('../models/%s_cvae_classifier.h5'% args.dataset)

    for layer in classifier.layers:
        layer.trainable = False
    return classifier

def import_encoder(args):
    assert args.dataset in ['mnist', 'svhn','gtsrb'], \
        "Dataset parameter must be either 'mnist', 'gtsrb' or 'svhn'"
    assert os.path.isfile('../models/%s_cvae_encoder.h5'% args.dataset), \
        'model file not found...'
    # load the cvae classifier model
    encoder = load_model('../models/%s_cvae_encoder.h5'% args.dataset)

    for layer in encoder.layers:
        layer.trainable = False

    return encoder

class Regressor_z(Layer):
    """A layer that relcassifies adv image 
    """
    def __init__(self, decoder, latent_dim, **kwargs):
        self.decoder = decoder
        self.latent_dim = latent_dim
        super(Regressor_z, self).__init__(**kwargs)
    def build(self,input_shape):
        self.z_star = self.add_weight(name='z_star',
                                    shape=(1,self.latent_dim),
                                    initializer='zeros',
                                    trainable=True)
        super(Regressor_z, self).build(input_shape) 

    def call(self, inputs):
        #z = K.expand_dims(self.z_star, 0)
        z = self.decoder(self.z_star)
        residual = inputs - z
        return residual
    def reinitial(self,z_reinit):
        self.set_weights([z_reinit])
    def reinitial_to_zeros(self):
        self.set_weights([np.zeros(self.z_star.shape)])

def build_regressor(regressor_model, img_dim, channel_dim):
    x = Input(shape=(img_dim, img_dim, channel_dim))
    #print (x.shape)
    #r=x
    r = regressor_model(x)
    regressor = Model(x, r, name='regressor') 
    regressor.summary()
    # plot_model(regressor, to_file='regressor.png', show_shapes=True)
    re_loss = K.sum(r**2)
    regressor.add_loss(re_loss)
    opt = Adam(lr=0.01,beta_1=0.9,beta_2=0.999,epsilon=1e-08,decay=0.0)
    regressor.compile(optimizer=opt)
    return regressor

def regressor_fit(args,regressor,input_x,epochs=50):
    #input should be in shape of img_dim,img_dim,channel_num
    input_x = K.expand_dims(input_x,0)
    #print(input_x.shape)
    checkpointer = keras.callbacks.ModelCheckpoint(
                filepath = './models/checkpoints/checkpoint.h5', 
                monitor='val_loss', 
                verbose=1, 
                save_best_only=True, 
                save_weights_only=False, 
                mode='min', 
                period=1
    )
    lrate = keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', 
                factor=0.2, 
                patience=2, 
                verbose=0, 
                mode='min', 
                epsilon=0.0001, 
                cooldown=0, 
                min_lr=0
    )

    history = regressor.fit(input_x,
                epochs=epochs,
                #batch_size=1,
                verbose=0,                      #if u wanna display progress bar then uncomment this line
                steps_per_epoch=1
                #callbacks=[checkpointer, lrate]
                )

    # list all data in history
    #print(history.history.keys())
    # if args.visual:
    #     pyplot.plot(history.history['loss'])
    #     pyplot.title('model loss')
    #     pyplot.ylabel('loss')
    #     pyplot.xlabel('epoch')
    #     pyplot.show()

    return regressor.layers[-1].z_star,history.history['loss'][-1]

def reclassification(x_reg, means):
    #reclassify by the E_distance between z_star and each class center
    z = x_reg-means
    z = np.sum(z**2,1)
    z = np.argmin(z,0)
    return z

def reclassification1(x_reg,classifier):
    #reclassify by result of classifier on regressed z_star
    z = classifier(x_reg)
    #print(z.shape)
    z = K.argmax(z,1)
    return z

def reclassification2(x_reg,encoder,decoder,classifier):
    #reclassify by result of classifier on regressed z_star after 1 postprocess
    z = decoder(x_reg)
    z,_ = encoder(z)
    z = classifier(z)
    z = K.argmax(z,1)
    return z

def reclassification3(x_reg,encoder,decoder,classifier):
    #reclassify by result of classifier on regressed z_star after 5 postprocess
    z = x_reg
    for i in range(5):
        z = decoder(x_reg)
        z,_ = encoder(z)
    z = classifier(z)
    z = K.argmax(z,1)
    return z

def visualize_x_adv_reg(X,X_adv,X_adv_reg,X_reg):
    #X_reg = decoder(X_reg)

    X = X.squeeze()
    X_adv = X_adv.squeeze()
    X_adv_reg = X_adv_reg.squeeze()
    X_reg = X_reg.squeeze()

    plt.figure(figsize=(8, 4))
    # plt.suptitle('X,X_adv,X_adv_decoded, X_reg')

    plt.subplot(1,4,1),plt.title('Original')
    plt.imshow(X)

    plt.subplot(1,4,2),plt.title('Adversarial')
    plt.imshow(X_adv)

    plt.subplot(1,4,3),plt.title('Decoded')
    plt.imshow(X_adv_reg)

    plt.subplot(1,4,4),plt.title('Decoded(Recovered)')
    plt.imshow(X_reg)

    plt.savefig('visualization/sample')
    plt.show()

def cal_acc(y_list):
    y_l_p = np.transpose(y_list)
    table = (y_l_p == y_l_p[-2])
    acc = table.sum(1)/float(table.shape[1])
    return acc
     
def main(args):
    if args.dataset == 'mnist':
        num_classes = 10
        latent_dim = 20
    elif args.dataset == 'svhn':
        num_classes = 10
        latent_dim = 20      
    elif args.dataset == 'gtsrb':
        num_classes = 43
        latent_dim = 32

    assert os.path.isfile('../data/Adv_%s_%s.mat' %
                          (args.dataset, args.attack)), \
        'adversarial sample file not found... must first craft adversarial ' \
        'samples using craft_adv_cave.py'
    assert os.path.isfile('../data/det_%s_%s_all.mat' %
                          (args.dataset, args.attack)), \
        'detected label file not found... must implement detection first ' \
        'samples using craft_adv_cave.py'

    # Load adversarial samples
    data = sio.loadmat('../data/Adv_%s_%s.mat' % (args.dataset, args.attack))
    data_det = sio.loadmat('../data/det_%s_%s_all.mat' % (args.dataset, args.attack))

    X_test =  data['X']
    X_test_adv =  data['X_adv']
    Y = data['Y']
    Y_adv = data['Y_preds']
    print(data_det.keys())
    detected = data_det['detected']

    if args.cover_exsisting==False:
        exsisting_mat_name = '../data/Adv_'+args.dataset+'_'+args.attack+'_r.mat'
        if isfile(exsisting_mat_name):
            data_t = sio.loadmat(exsisting_mat_name)
            starting_idx = len(data_t['X_regr'])
        else:
            starting_idx = 0
    ###
    img_dim,channel_dim = X_test_adv.shape[-2:]
    img_num = min(X_test_adv.shape[0],detected.shape[1])

    decoder, means = import_model(args)
    classifier = import_classifier(args)
    encoder = import_encoder(args)
    z_regressor = Regressor_z(decoder,latent_dim)

    regressor = build_regressor(z_regressor,img_dim,channel_dim)

    sess = K.get_session()
    y_list=[]

    idx_batch_t = 0
    for i in range(img_num):
        if i < starting_idx:
            continue
        print('=> The {} of {} images is under processing'.format(i+1,img_num))
        start = time.time()
        input_x = X_test_adv[i]
        if args.zeros_start == True:
            regressor.layers[-1].reinitial_to_zeros()
            # regressor.layers[-1].reinitial(means[0][np.newaxis,:])
            z_star,_ = regressor_fit(args,regressor,input_x,epochs=1000)
        else:
            loss_f = 10000
            for j in range(len(means)):
                regressor.layers[-1].reinitial(means[j][np.newaxis,:])
                z_star_t,loss_t = regressor_fit(args,regressor,input_x,epochs=1000)
                if loss_t < loss_f:
                    z_star = z_star_t

        #restore the regressed z_star
        z_star1 = sess.run(z_star)
        if i%args.batch_size == 0:
            z_regr = z_star1
            y_list=[]
        else:
            z_regr = np.concatenate([z_regr,z_star1],0)

        #reclassification
        y_reg = reclassification(z_star1,means)

        y_reg1 = reclassification1(z_star,classifier)
        y_reg1 = sess.run(y_reg1)[0]

        y_reg2 = reclassification2(z_star,encoder,decoder,classifier)
        y_reg2 = sess.run(y_reg2)[0]

        y_reg3 = reclassification3(z_star,encoder,decoder,classifier)
        y_reg3 = sess.run(y_reg3)[0]
        
        if detected[:,i] == 1:
            y_list.append([y_reg,y_reg1,y_reg2,y_reg3, Y[i].argmax(0),Y_adv[i].argmax(0)])
        end = time.time()
        print('time cost of this iter is:{}'.format(str(end-start)))

        if (i+1)%args.batch_size == 0 or i+1 == img_num:
            if i//args.batch_size ==0:
                data['X_regr'] = z_regr
                y_list = np.array(y_list)
                data['y_list'] = y_list
                sio.savemat('../data/Adv_%s_%s_r.mat'%(args.dataset,args.attack),data)
                print('{} batch of regressed hidden_variables added to mat file and saved to data/ subfolder.'.format(i//args.batch_size+1)) 
                del data
            else:     
                data = sio.loadmat('../data/Adv_%s_%s_r.mat'%(args.dataset,args.attack))
                z_regr_t = data['X_regr']
                y_list_t = data['y_list']
                z_regr = np.concatenate([z_regr_t,z_regr],0)
                y_list = np.concatenate([y_list_t,np.array(y_list)],0)
                data['X_regr'] = z_regr
                data['y_list'] = y_list
                sio.savemat('../data/Adv_%s_%s_r.mat'%(args.dataset,args.attack),data)
                print('{} batch of regressed hidden_variables added to mat file and saved to data/ subfolder.'.format(i//args.batch_size+1)) 
                del z_regr_t,y_list_t,data,z_regr, y_list            

    if args.visual==True:
        input_x = X_test_adv[args.index]
        if args.zeros_start == True:
            regressor.layers[-1].reinitial_to_zeros()
            # regressor.layers[-1].reinitial(means[0][np.newaxis,:])
            z_star,_ = regressor_fit(args,regressor,input_x,epochs=1000)
        else:
            loss_f = 10000
            for j in range(len(means)):
                regressor.layers[-1].reinitial(means[j][np.newaxis,:])
                z_star_t,loss_t = regressor_fit(args,regressor,input_x,epochs=1000)
                if loss_t < loss_f:
                    z_star = z_star_t

        X_reg = decoder(z_star)[0]
        X_reg = sess.run(X_reg)

        input_x=tf.convert_to_tensor(input_x[np.newaxis,:])
        input_x = tf.cast(input_x, 'float32')
        X_adv_h,_ = encoder(input_x)
        X_adv_reg = decoder(X_adv_h)[0]
        X_adv_reg = sess.run(X_adv_reg)

        #if args.dataset == 'svhn' or args.dataset == 'gtsrb' or args.dataset == 'cw':
        visualize_x_adv_reg(X_test[args.index],X_test_adv[args.index],X_adv_reg,X_reg)
        print (np.sum((X_test[args.index]-X_test_adv[args.index]).reshape(-1)**2,0))
        print (np.sum((X_test[args.index]-X_reg).reshape(-1)**2,0))
        print (np.sum((X_test_adv[args.index]-X_reg).reshape(-1)**2,0))

    print('regressed hidden_variables added to mat file and saved to data/ subfolder.')
    # acc = cal_acc2(np.array(y_list),detected)
    y_list = sio.loadmat('../data/Adv_%s_%s_r.mat'%(args.dataset,args.attack))['y_list']
    print("%s adversarial samples are detected and recovered"%(len(y_list)))
    acc = cal_acc(y_list)
    print("ACC of each strategy is:")
    print("ACC_null(w/o recvoery): %s"%(acc[5]))
    print("ACC_R0: %s"%(acc[0]))
    print("ACC_R1: %s"%(acc[1]))
    print("ACC_R2: %s"%(acc[2]))
    print("ACC_R3: %s"%(acc[3]))
    # print("acc of each reclassification method is  %s"%(acc))   
    #pyplot.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--dataset',
        help="Dataset to use; either 'mnist', 'cifar','gtsrb' or 'svhn'",
        required=False, type=str
    )
    parser.add_argument(
        '-a', '--attack',
        help="Attack to use; either 'fgsm', 'bim-a', 'bim-b', 'jsma' 'cw' "
             "or 'all'",
        required=False, type=str
    )
    parser.add_argument(
        '-z', '--zeros_start',
        help="initial z_star to each class center or zeros",
        required=False, type=bool
    )
    parser.add_argument(
        '-b', '--batch_size',
        help="batch size of operating",
        required=False, type=int
    )
    parser.add_argument(
        '-v', '--visual',
        help="visualizaiton of specific sample and tarining plot",
        required=False, type=bool
    )
    parser.add_argument(
        '-i', '--index',
        help="the index of the sample user wanna visualize",
        required=False, type=int
    )
    parser.add_argument('-c','--cover_exsisting', action='store_true', help='set true if u wanna cover existing mat file and redo the experiment')
    parser.set_defaults(dataset='gtsrb')
    parser.set_defaults(attack='fgsm')
    parser.set_defaults(zeros_start=True)
    parser.set_defaults(batch_size=500)
    parser.set_defaults(visual=False)
    parser.set_defaults(index=0)
    args = parser.parse_args()
    main(args)

