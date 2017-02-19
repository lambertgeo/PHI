'''
to avoid problems with keras versions download Keras from Chollet's:
sudo pip install --upgrade git+https://github.com/fchollet/keras
'''
import os
import sys
import h5py
import numpy as np
np.set_printoptions(threshold=np.inf)
np.random.seed(1337) # for reproducibility
from keras.models import Sequential, Model, load_model 
from keras.layers import Flatten, Dense, Dropout, Reshape, Permute, Activation, \
    Input, merge
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras import backend as K
import scipy as sp
from scipy.misc import imread, imresize, imsave
from keras.layers.core import  Lambda, Merge
from keras import backend as K
from keras.engine import Layer
from os import listdir
from os.path import isfile, join, dirname
from scipy.io import loadmat
from scipy import ndimage
from scipy import misc
from skimage.exposure import rescale_intensity
from skimage import novice
import matplotlib.pyplot as plt
import theano

np.set_printoptions(threshold=np.inf)


imagepath = ('goldenpride1.jpg')
picture = novice.open(imagepath)
width = picture.width
height = picture.height
weights_path = ('feb13_try.h5')
phi = misc.imread(imagepath)
im_converted = rescale_intensity(phi)
im = im_converted.transpose((2,0,1))
im = np.expand_dims(im, axis=0)
'''
plt.imshow(phi)
plt.show()
'''
def rotate_weights(model,layer_name):
        layer = model.get_layer(layer_name)
        W,b = layer.get_weights()
        n_filter,previous_filter,ax1,ax2 = layer.get_weights()[0].shape
        new_W = W.reshape((previous_filter,ax1,ax2,n_filter))
        new_W = new_W.transpose((3,0,1,2))
        new_W = new_W[:,:,::-1,::-1]
        layer.set_weights([new_W,b])


def VGG_16T(weights_path=None):
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(3,None,None)))
    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_2'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_3'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
     
    model.add(Convolution2D(512,7,7,activation='relu',name='dense_1'))
    model.add(Convolution2D(1,1,1,activation='relu',name='dense_2'))
	    
    if weights_path:
        model.load_weights(weights_path, by_name = True)
        rotate_weights(model,'dense_1')
        rotate_weights(model,'dense_2')       

    for layer in model.layers[:32]:
        layer.trainable = False

    return model
		        
     
model = VGG_16T(weights_path)
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='binary_crossentropy')

out = model.predict(im)

#HYPERCOLUMN
def extract_hypercolumn(model, layer_indexes, instance):
    layers = [model.layers[li].output for li in layer_indexes]
    get_feature = K.function([model.layers[0].input],layers)
    assert instance.shape == (1,3,height,width)
    feature_maps = get_feature([instance])
    hypercolumns = []
    for convmap in feature_maps:
        for fmap in convmap[0]:
            upscaled = sp.misc.imresize(fmap, size=(height,width),
                                        mode="F", interp='bilinear')
            hypercolumns.append(upscaled)

    return np.asarray(hypercolumns)


#layers_extract = [15, 16, 26, 28, 30,31,32]
layers_extract = [32]
hc = extract_hypercolumn(model, layers_extract, im)
ave = np.average(hc.transpose(1, 2, 0), axis=2)
plt.imshow(ave)
plt.show()

