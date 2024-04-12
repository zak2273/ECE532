import os
import hls4ml
from keras.models import load_model

os.environ['PATH'] += os.pathsep + '/cad1/Xilinx/Vivado/2018.3/bin'

model = load_model('/homes/n/nabizakr/ece532/python/my_model.keras')

config = hls4ml.utils.config_from_keras_model(model, granularity='name')
print(config)

hls_model = hls4ml.converters.convert_from_keras_model(
    model, 
    hls_config=config, 
    output_dir='/homes/n/nabizakr/ece532/python/model_1/hls4ml_prj', 
    part='xc7a100tcsg324-1'
)

hls4ml.utils.plot_model(hls_model, show_shapes=True, show_precision=True, to_file=None)

hls_model.compile()
hls_model.build(csim=False, export=True, bitfile=True)
hls4ml.report.read_vivado_report('/homes/n/nabizakr/ece532/python/model_1/hls4ml_prj/')
