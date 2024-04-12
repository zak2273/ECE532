import os
import platform
import hls4ml
import tensorflow as tf
from tf_keras.models import load_model
import matplotlib.pyplot as plt

os.environ['PATH'] += os.pathsep + 'D:/Xilinx/Vivado/2018.3/bin'

model = load_model('my_model.keras')

config = hls4ml.utils.config_from_keras_model(model, granularity='name')
print(config)

hls_model = hls4ml.converters.convert_from_keras_model(
    model, 
    hls_config=config, 
    output_dir='model_1/hls4ml_prj', 
    part='xc7a100tcsg324-1'
)
print(platform.system())
#hls4ml.utils.plot_model(hls_model, show_shapes=True, show_precision=True, to_file=None)
hls_model.compile()
#hls_model.build()