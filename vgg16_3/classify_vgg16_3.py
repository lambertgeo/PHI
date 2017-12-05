import sys
import os
import csv
import glob
import tensorflow as tf
from keras import utils
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as k
from keras import models
from keras.models import load_model
from keras import layers
from keras.applications import VGG16
import shutil
import numpy as np
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as k

test_data_dir = '/home/dede/Bureau/test/'
results_dir = '/home/dede/Bureau/'
results_name = 'predictions.csv'

img_width, img_height = 224, 224  #
batch_size = 20  # try 4, 8, 16, 32, 64, 128, 256 dependent on CPU/GPU memory capacity (powers of 2 values).

model = load_model('/home/dede/Bureau/vgg_16_3.h5')

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-5),
              metrics=['acc'])

# Read Data
test_datagen = ImageDataGenerator(rescale=1. / 255)
test_generator = test_datagen.flow_from_directory(test_data_dir,
                                                  target_size=(img_width, img_height),
                                                  batch_size=batch_size,
                                                  shuffle=False)
# Calculate class posteriors probabilities
y_probabilities = model.predict_generator(test_generator,
					  steps=40)
                                              
# Calculate class labels
filenames = [filename.split('/')[1] for filename in test_generator.filenames]
ids = [filename.split('.')[0] for filename in filenames]
# save results as a csv file in the specified results directory
with open(os.path.join(results_path, results_name), 'w') as file:
    writer = csv.writer(file)
    writer.writerow(('id', 'class0_prob'))
    writer.writerows(zip(ids, y_probabilities[:, 0]))
   
# release memory
k.clear_session()
