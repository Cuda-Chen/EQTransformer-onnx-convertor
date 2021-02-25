from EQTransformer.core.EqT_utils import f1, SeqSelfAttention, FeedForward, LayerNormalization
from EQTransformer.core.mseed_predictor import mseed_predictor, _mseed2nparry, PreLoadGeneratorTest
import keras
from keras.models import load_model
from keras.optimizers import Adam
from keras.engine.training_utils import iter_sequence_infinite
import keras2onnx
import platform
from os import listdir
from os.path import join
import pprint as pp
import numpy as np

import onnxruntime

params_pred = {'batch_size': 500,
               'norm_mode': 'std'}

args = {'input_dir': 'downloads_mseeds',
        'stations_json': 'station_list.json',
        'overlap': 0.3}

# original
# detection.ipynb
print('Keras:')
model = load_model('EqT_model.h5',
                    custom_objects={
                       'SeqSelfAttention': SeqSelfAttention, 
                       'FeedForward': FeedForward,
                       'LayerNormalization': LayerNormalization, 
                   'f1': f1})
model.compile(loss = ['binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy'],
              loss_weights = [0.02, 0.40, 0.58],           
              optimizer = Adam(lr = 0.001),
              metrics = [f1])

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
        print(f'{month}')
        meta, time_slots, comp_types, data_set = _mseed2nparry(args, matching, time_slots, comp_types, st)

        pred_generator = PreLoadGeneratorTest(meta["trace_start_time"], data_set, **params_pred)

        predD, predP, predS = model.predict_generator(pred_generator)

# ONNX port
print('ONNX:')
sess_options = onnxruntime.SessionOptions()
sess = onnxruntime.InferenceSession('eqt_model.onnx', sess_options)

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
        print(f'{month}')
        meta, time_slots, comp_types, data_set = _mseed2nparry(args, matching, time_slots, comp_types, st)

        pred_generator = PreLoadGeneratorTest(meta["trace_start_time"], data_set, **params_pred)

        out_pred_generator = iter_sequence_infinite(pred_generator)
        steps_done = 0
        steps = len(pred_generator)

        # ONNX model predict_generator monkey typing
        while steps_done < steps:
            generator_output = next(out_pred_generator)

            x = generator_output
            x_test = list(x.values())[0].astype(np.float32)
            res = sess.run(None, input_feed={'input': x_test})

            steps_done += 1
