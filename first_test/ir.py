#Reference:https://www.kaggle.com/bmarcos/image-recognition-gender-detection-inceptionv3


import pandas as pd
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score

from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.utils import np_utils
from keras.optimizers import SGD

from IPython.core.display import display, HTML
from PIL import Image
from io import BytesIO
import base64
import os
#from ir_methods import load_reshape_img

import tensorflow as tf
print(tf.__version__)
plt.style.use('ggplot')


#methods to generate dataframe

def load_reshape_img(fname):
    img = load_img(fname)
    x = img_to_array(img)/255.
    x = x.reshape((1,) + x.shape)

    return x


def generate_df(partition, attr, num_samples):
    '''
    partition
        0 -> train
        1 -> validation
        2 -> test

    '''
    df_ = df_par_attr[(df_par_attr['partition'] == partition) & (df_par_attr[attr] == 0)].sample(int(num_samples/2))
    df_ = pd.concat([df_,
                      df_par_attr[(df_par_attr['partition'] == partition) & (df_par_attr[attr] == 1)].sample(int(num_samples/2),replace=True)])

    # for Train and Validation
    if partition != 2:
        x_ = np.array([load_reshape_img(images_folder + fname) for fname in df_.index])
        x_ = x_.reshape(x_.shape[0], 218, 178, 3)
        y_ = np_utils.to_categorical(df_[attr],2)
    # for Test
    else:
        x_ = []
        y_ = []

        for index, target in df_.iterrows():
            im = cv2.imread(images_folder + index)
            im = cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), (IMG_WIDTH, IMG_HEIGHT)).astype(np.float32) / 255.0
            im = np.expand_dims(im, axis =0)
            x_.append(im)
            y_.append(target[attr])

    return x_, y_







#========================================SECTION 1: Set-up========================================
#change Directory
my_path = os.getcwd()
main_folder = my_path
images_folder = main_folder + '/dataset/img_align_celeba/'

EXAMPLE_PIC = images_folder + '000005.jpg'

n = 202599
training_ratio = 0.8

TRAINING_SAMPLES = 8000       #approx 80%
VALIDATION_SAMPLES = 1000      #approx 10%
TEST_SAMPLES = 1000            #approx 10%
IMG_WIDTH = 178
IMG_HEIGHT = 218
BATCH_SIZE = 16
NUM_EPOCHS = 20

# import the data set that include the attribute for each picture
df_attr = pd.read_csv(main_folder + '/dataset/list_attr_celeba.txt', sep = "\s+")
df_attr.set_index('image_id', inplace=True)
df_attr.replace(to_replace=-1, value=0, inplace=True) #replace -1 by 0
df_attr.shape


# Example of reading in images
"""
img = load_img(EXAMPLE_PIC)
plt.grid(False)
plt.imshow(img)
print(df_attr.loc[EXAMPLE_PIC.split('/')[-1]][['Smiling','Male','Young']]) #some attributes
"""


# ===============================SECTION 2: Train-Validation-Test Split================================
df_partition = pd.read_csv(main_folder+'/dataset/list_eval_partition.txt',sep = "\s+",names = ["image_id", "partition"])

df_partition['partition'].value_counts().sort_index()
df_partition.set_index('image_id', inplace=True)
df_par_attr = df_partition.join(df_attr['Male'], how='inner')


# ===============================SECTION 3: Data Augment: Pre-processing================================

#just for tryout
"""
# Generate image generator for data augmentation
datagen =  ImageDataGenerator(
  #preprocessing_function=preprocess_input,
  rotation_range=30,
  width_shift_range=0.2,
  height_shift_range=0.2,
  shear_range=0.2,
  zoom_range=0.2,
  horizontal_flip=True
)

#testing of data augmentation with example image

# load one image and reshape
img = load_img(EXAMPLE_PIC)
x = img_to_array(img)/255.
x = x.reshape((1,) + x.shape)

# plot 10 augmented images of the loaded iamge
plt.figure(figsize=(20,10))
plt.suptitle('Data Augmentation', fontsize=28)

i = 0
for batch in datagen.flow(x, batch_size=1):
    plt.subplot(3, 5, i+1)
    plt.grid(False)
    plt.imshow( batch.reshape(218, 178, 3))

    if i == 9:
        break
    i += 1

plt.show()
"""

# Make Train data_df
x_train, y_train = generate_df(0, 'Male', TRAINING_SAMPLES)

# Train - Data Preparation - Data Augmentation with generators
train_datagen =  ImageDataGenerator(
  preprocessing_function=preprocess_input,
  rotation_range=30,
  width_shift_range=0.2,
  height_shift_range=0.2,
  shear_range=0.2,
  zoom_range=0.2,
  horizontal_flip=True,
)

train_datagen.fit(x_train)

train_generator = train_datagen.flow(
x_train, y_train,
batch_size=BATCH_SIZE,
)

x_valid, y_valid = generate_df(1, 'Male', VALIDATION_SAMPLES)

# ===============================SECTION 4: Building model================================
# Import InceptionV3 Model
inc_model = InceptionV3(weights='imagenet',     #../input/inceptionv3/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5 (weight file?)
                        include_top=False,
                        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))

print("number of layers:", len(inc_model.layers))
#inc_model.summary()
