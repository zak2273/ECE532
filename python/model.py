import os
import numpy as np
import tensorflow as tf

from tf_keras.preprocessing.image import ImageDataGenerator
from tf_keras.models import Sequential
from tf_keras.layers import Input, Flatten, Dense, Activation, BatchNormalization
from tf_keras.optimizers import Adam
from tf_keras.regularizers import l1
from tf_keras.models import load_model

from matplotlib import pyplot as plt

# dimensions of our images.
img_width, img_height = 120, 160

# path variables we will need later
path = 'C:/ECE532'
train_data_dir = path + '/data/train'
validation_data_dir = path + '/data/validation'
test_data_dir = path + '/data/test'

# training parameters
num_epochs = 10
batch_size = 1

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

display = False
if display:
    # Display some images to ensure the generator works correctly
    def display_image(image, batch):
        if not batch:
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

# create tensorflow model
model = Sequential()
model.add(Input(shape=(img_width, img_height, 1)))
model.add(Flatten())
model.add(Dense(64, activation='sigmoid'))
model.add(Dense(32, activation='sigmoid'))
model.add(Dense(32, activation='sigmoid'))
model.add(Dense(4, activation='softmax'))

train = False
if train:
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
        batch_size=batch_size,
        validation_data=validation_generator,
    )
    # save the model
    model.summary()
    model.save('my_model.keras')
    # evaluate on test data
    scores = model.evaluate(test_generator)
else:
    model = load_model('my_model.keras')
    
import hls4ml
import plotting
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

config = hls4ml.utils.config_from_keras_model(model, granularity='name')
print(config)

hls_model = hls4ml.converters.convert_from_keras_model(
    model, 
    hls_config=config, 
    output_dir='model_1/hls4ml_prj', 
    part='xc7a100tcsg324-1'
)

hls_model.compile()

y_keras = model.predict(test_generator)
y_hls = hls_model.predict(test_generator)
print("Accuracy: {}".format(accuracy_score(np.argmax(test_generator.classes, axis=1), np.argmax(y_keras, axis=1))))
plt.figure(figsize=(9, 9))
_ = plotting.makeRoc(test_generator.classes, y_keras, test_generator.class_indices)