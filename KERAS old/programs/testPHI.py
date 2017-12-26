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

weights_path = ('feb13_try.h5')
imagepath = ('richat.jpg')

phi = misc.imread(imagepath)
im_original = misc.imresize(phi,(224, 224),interp='bicubic')
im_converted = rescale_intensity(im_original)
im = im_converted.transpose((2,0,1))
im = np.expand_dims(im, axis=0)

plt.imshow(phi)
plt.show()

# build the VGG16 network
model = Sequential()
model.add(ZeroPadding2D((1, 1), input_shape=(3, 224, 224)))

model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_2'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_3'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(Flatten())
model.add(Dense(512, activation='relu', name = 'dense_1'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid', name = 'dense_2'))

# load weights
model.load_weights(weights_path, by_name=True)
print("Loaded model from disk")

# compile the model with a SGD/momentum optimizer
# and a very slow learning rate.

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='binary_crossentropy')

out = model.predict(im)
print(out)

