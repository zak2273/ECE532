import os
import numpy as np
import tensorflow as tf

from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt

# dimensions of our images.
img_width, img_height = 8, 8

# path variables we will need later
train_data_dir = '/homes/n/nabizakr/ece532/python/data/train'
validation_data_dir = '/homes/n/nabizakr/ece532/python/data/validation'
test_data_dir = '/homes/n/nabizakr/ece532/python/data/test'

# training parameters
num_epochs = 10
batch_size = 2

# dataset metrics
num_glass_tr = len(os.listdir(train_data_dir + '/glass'))
num_metal_tr = len(os.listdir(train_data_dir + '/metal'))
num_paper_tr = len(os.listdir(train_data_dir + '/paper'))
num_plastic_tr = len(os.listdir(train_data_dir + '/plastic'))

num_glass_val = len(os.listdir(validation_data_dir + '/glass'))
num_metal_val = len(os.listdir(validation_data_dir + '/metal'))
num_paper_val = len(os.listdir(validation_data_dir + '/paper'))
num_plastic_val = len(os.listdir(validation_data_dir + '/plastic'))

total_train = num_glass_tr + num_metal_tr + num_paper_tr + num_plastic_tr
total_val = num_glass_val + num_metal_val + num_paper_val + num_plastic_val

# augmentation configuration for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    vertical_flip=False,
    horizontal_flip=False
)

# augmentation configuration for validation
valid_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    vertical_flip=False,
    horizontal_flip=False
)

# augmentation configuration for testing
test_datagen = ImageDataGenerator(
    vertical_flip=False,
    horizontal_flip=False
)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    color_mode='grayscale',
    class_mode='sparse',
    shuffle=True,
    seed=123
)

validation_generator = valid_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    color_mode='grayscale',
    class_mode='sparse',
    shuffle=True,
    seed=123
)

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=1,
    color_mode='grayscale',
    class_mode='sparse',
    shuffle=False
)

'''
# Display some images to ensure the generator works correctly
def display_image(image, test):
    if not test:
        image = np.squeeze(image[0])
        image = np.delete(image, 1, axis=0)
        image = np.transpose(image, (2, 1, 0))

        print(np.shape(image))
    else:
        image = np.squeeze(image[0])
        image = np.transpose(image, (1, 0))
    plt.imshow(image, cmap='gray')
    plt.show()
    
train_image = train_generator[0]
val_image = validation_generator[0]
test_image = test_generator[0]

display_image(train_image, False)
display_image(val_image, False)
display_image(test_image, True)
'''

#print(train_generator[0][0].shape)
#print(train_generator[0][1])
#exit()

# create tensorflow model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Input(shape=(img_width, img_height, 1)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(30, activation=tf.nn.sigmoid))
model.add(tf.keras.layers.Dense(30, activation=tf.nn.sigmoid))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.sigmoid))
model.add(tf.keras.layers.Dense(4, activation=tf.nn.softmax))

# compile and train the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# train the model
model.fit(
    train_generator,
    epochs=num_epochs,
    steps_per_epoch=total_train // batch_size,
    validation_data=validation_generator,
    validation_steps=total_val // batch_size
)

model.summary()

model.save('/homes/n/nabizakr/ece532/python/my_model.keras')

# evaluate on test data
scores = model.evaluate(test_generator)
