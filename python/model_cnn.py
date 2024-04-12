import os
import time
import numpy as np
import tensorflow.compat.v2 as tf
import matplotlib.pyplot as plt

from keras.preprocessing.image import ImageDataGenerator

# dimensions of our images.
img_width, img_height = 120, 160

# path variables we will need later
train_data_dir = '/homes/n/nabizakr/ece532/python/data/train'
validation_data_dir = '/homes/n/nabizakr/ece532/python/data/validation'
test_data_dir = '/homes/n/nabizakr/ece532/python/data/test'

# training parameters
num_epochs = 50
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
    class_mode='categorical',
    shuffle=True,
    seed=123
)

validation_generator = valid_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    color_mode='grayscale',
    class_mode='categorical',
    shuffle=True,
    seed=123
)

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=1,
    color_mode='grayscale',
    class_mode='categorical',
    shuffle=False
)

display = False
if display:
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

# create cnn model
from keras.layers import Input
from keras.layers import BatchNormalization
from keras.layers import MaxPooling2D
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Dense

from keras.regularizers import l1

from keras.models import Model

from qkeras import QActivation
from qkeras import QDense, QConv2DBatchnorm

n_classes = 4
filters_per_conv_layer = [4, 4, 6]
neurons_per_dense_layer = [16]

x = x_in = Input((img_width, img_height, 1))

# add conv layers
for i, f in enumerate(filters_per_conv_layer):
    print(('Adding fused QConv+BN block {} with N={} filters').format(i, f))
    x = QConv2DBatchnorm(
        int(f),
        kernel_size=(3, 3),
        strides=(2, 2),
        kernel_quantizer="quantized_bits(6,0,alpha=1)",
        bias_quantizer="quantized_bits(6,0,alpha=1)",
        kernel_initializer='lecun_uniform',
        kernel_regularizer=l1(0.0001),
        use_bias=True,
        name='fused_convbn_{}'.format(i),
    )(x)
    x = QActivation('quantized_relu(6)', name='conv_act_%i' % i)(x)
    x = MaxPooling2D(pool_size=(2, 2), name='pool_{}'.format(i))(x)
x = Flatten()(x)

# add dense layers
for i, n in enumerate(neurons_per_dense_layer):
    print(('Adding QDense block {} with N={} neurons').format(i, n))
    x = QDense(
        n,
        kernel_quantizer="quantized_bits(6,0,alpha=1)",
        kernel_initializer='lecun_uniform',
        kernel_regularizer=l1(0.0001),
        name='dense_%i' % i,
        use_bias=False,
    )(x)
    x = BatchNormalization(name='bn_dense_{}'.format(i))(x)
    x = QActivation('quantized_relu(6)', name='dense_act_%i' % i)(x)
x = Dense(int(n_classes), name='output_dense')(x)
x_out = Activation('softmax', name='output_softmax')(x)

# see model params
qmodel = Model(inputs=[x_in], outputs=[x_out], name='qkeras')
qmodel.summary()

# check num_params per layer is less than 4096 (vivado limit)
for layer in qmodel.layers:
    if layer.__class__.__name__ in ['Conv2D', 'Dense']:
        w = layer.get_weights()[0]
        layersize = np.prod(w.shape)
        print("{}: {}".format(layer.name, layersize))  # 0 = weights, 1 = biases
        if layersize > 4096:  # assuming that shape[0] is batch, i.e., 'None'
            print("Layer {} is too large ({}), are you sure you want to train?".format(layer.name, layersize))

# prune dense and convolutional layers
from keras.models import clone_model
import tensorflow_model_optimization as tfmot
from tensorflow_model_optimization.sparsity import keras as sparsity
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_callbacks

NSTEPS = total_train // batch_size
print('Number of training steps per epoch is {}'.format(NSTEPS))

# Prune all convolutional and dense layers gradually from 0 to 50% sparsity every 2 epochs,
# ending by the 10th epoch
def pruneFunction(layer):
    pruning_params = {
        'pruning_schedule': sparsity.PolynomialDecay(
            initial_sparsity=0.0, final_sparsity=0.50, begin_step=NSTEPS * 2, end_step=NSTEPS * 10, frequency=NSTEPS
        )
    }
    if isinstance(layer, tf.keras.layers.Conv2D):
        return tfmot.sparsity.keras.prune_low_magnitude(layer, **pruning_params)
    if isinstance(layer, tf.keras.layers.Dense) and layer.name != 'output_dense':
        return tfmot.sparsity.keras.prune_low_magnitude(layer, **pruning_params)
    return layer

qmodel_pruned = clone_model(qmodel, clone_function=pruneFunction)

# train the model
train = False
if train:
    LOSS = 'categorical_crossentropy'
    OPTIMIZER = 'adam'

    qmodel_pruned.compile(loss=LOSS, optimizer=OPTIMIZER, metrics=["accuracy"])
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=10, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1),
        pruning_callbacks.UpdatePruningStep(),
    ]
    
    start = time.time()
    qmodel_pruned.fit(train_generator, 
                     epochs=num_epochs, 
                     validation_data=validation_generator, 
                     callbacks=callbacks)
    end = time.time()
    
    print('It took {} minutes to train Keras model'.format((end - start) / 60.0))
    
    # save the model
    qmodel_pruned.save('/homes/n/nabizakr/ece532/python/cnn_model_reduced.h5')
else:
    from qkeras.utils import _add_supported_quantized_objects
    from tensorflow_model_optimization.sparsity.keras import strip_pruning
    from tensorflow_model_optimization.python.core.sparsity.keras import pruning_wrapper
    from keras.models import load_model
    
    co = {}
    _add_supported_quantized_objects(co)
    co['PruneLowMagnitude'] = pruning_wrapper.PruneLowMagnitude
    
    qmodel_pruned = load_model('/homes/n/nabizakr/ece532/python/cnn_model_reduced.h5', custom_objects=co)
    qmodel_pruned = strip_pruning(qmodel_pruned)

test = True
if test:
    predict_qkeras = qmodel_pruned.predict(test_generator[0][0])
    print(predict_qkeras)
    print(np.argmax(predict_qkeras, axis=1))

run_hls = True
if run_hls:
    import hls4ml
    import plotting

    # configure hls model
    hls_config = hls4ml.utils.config_from_keras_model(qmodel_pruned, granularity='name')
    hls_config['Model']['ReuseFactor'] = 8
    hls_config['LayerName']['output_softmax']['Strategy'] = 'Stable'
    for layer in ['dense_0', 'output_dense']:
        hls_config['LayerName'][layer]['ReuseFactor'] = 64

    '''
    cfg = hls4ml.converters.create_config(part='xc7a100t-csg324-1')
    cfg['HLSConfig'] = hls_config
    cfg['IOType'] = 'io_stream'
    cfg['Backend'] = 'Pynq'
    cfg['Interface'] = 'm_axi'
    cfg['AxiWidth'] = 8
    cfg['ApplyPatches'] = 1
    cfg['Implementation'] = 'serial'
    cfg['ClockPeriod'] = 10
    cfg['KerasModel'] = qmodel_pruned
    cfg['OutputDir'] = '/homes/n/nabizakr/ece532/inference/hls/nexys4_ddr_axi_8_serial_prj'
    cfg['ProjectName'] = 'matnet'
    
    print("-----------------------------------")
    print_dict(cfg)
    print("-----------------------------------")
    
    # profiling / testing
    hls_model = hls4ml.converters.keras_to_hls(cfg)
    '''
    
    hls_model = hls4ml.converters.convert_from_keras_model(
        qmodel_pruned, 
        hls_config=hls_config, 
        output_dir='/homes/n/nabizakr/ece532/inference/hls/nexys4_ddr_prj',
        io_type='io_stream',
        clock_period='10',
        #backend='VivadoAccelerator',
        #board='zcu102',
        backend='Vivado',
        part='xc7a100tcsg324-1',
        project_name='materials_net'
    )

    hls_model.compile()
    predict_hls = hls_model.predict(test_generator[0][0])
    print(predict_hls)
    print(np.argmax(predict_hls))

    hls4ml.model.profiling.numerical(model=qmodel_pruned, hls_model=hls_model)
    hls4ml.utils.plot_model(
        hls_model, 
        show_shapes=True, 
        show_precision=True, 
        to_file='/homes/n/nabizakr/ece532/inference/hls/nexys4_ddr_prj/plot.png')
    
    os.environ['PATH'] += os.pathsep + '/cad1/Xilinx/Vivado/2018.3/bin'

    synth = True
    if synth:
        hls_model.build(reset=True, csim=False, synth=True, export=True)
        #hls_model_q.build(csim=False, export=True, bitfile=True)
