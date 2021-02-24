"""from EQTransformer.core.EqT_utils import f1, SeqSelfAttention, FeedForward, LayerNormalization
from EQTransformer.core.mseed_predictor import mseed_predictor, _mseed2nparry, PreLoadGeneratorTest
import keras
from keras.models import load_model
import keras2onnx"""
import platform
from os import listdir
from os.path import join

# original
# detection.ipynb
"""model = load_model('EqT_model.h5',
                    custom_objects={
                       'SeqSelfAttention': SeqSelfAttention, 
                       'FeedForward': FeedForward,
                       'LayerNormalization': LayerNormalization, 
                   'f1': f1})
model.compile(loss = ['binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy'],
              loss_weights = [0.02, 0.40, 0.58],           
              optimizer = Adam(lr = 0.001),
              metrics = [f1])

time_slots, comp_types = [], []

params_pred = {'batch_size': 500,
               'norm_mode': 'std'}"""

args = {'input_dir': 'downloads_mseeds',
        'stations_json': 'station_list.json',
        'overlap': 0.3}

if platform.system() == 'Windows':
    station_list = [ev.split(".")[0] for ev in listdir(args['input_dir']) if ev.split("\\")[-1] != ".DS_Store"]
else:
    station_list = [ev.split(".")[0] for ev in listdir(args['input_dir']) if ev.split("/")[-1] != ".DS_Store"]

station_list = sorted(set(station_list))

for ct, st in enumerate(station_list):
    if platform.system() == 'Windows':
        file_list = [join(st, ev) for ev in listdir(args["input_dir"]+"\\"+st) if ev.split("\\")[-1].split(".")[-1].lower() == "mseed"]
    else:
        file_list = [join(st, ev) for ev in listdir(args["input_dir"]+"/"+st) if ev.split("/")[-1].split(".")[-1].lower() == "mseed"]

    mon = [ev.split('__')[1]+'__'+ev.split('__')[2] for ev in file_list]
    uni_list = list(set(mon))
    uni_list.sort()

    time_slots, comp_types = [], []

    for _, month in enumerate(uni_list):
        matching = [s for s in file_list if month in s]
        print(matching)
