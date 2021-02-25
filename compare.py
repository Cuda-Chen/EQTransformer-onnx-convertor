from EQTransformer.core.EqT_utils import f1, SeqSelfAttention, FeedForward, LayerNormalization
from EQTransformer.core.mseed_predictor import (
    mseed_predictor, 
    _mseed2nparry, 
    PreLoadGeneratorTest,
    _picker,
    _get_snr,
    _output_writter_prediction,
    _plotter_prediction,
)
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
import sys
import os
import csv
import shutil

params_pred = {'batch_size': 500,
               'norm_mode': 'std'}

args = {'input_dir': 'downloads_mseeds',   
         'input_model': 'EqT_model.h5',
         'stations_json': 'station_list.json',
         'output_dir': 'detections2',
         'loss_weights': [0.02, 0.40, 0.58],          
         'detection_threshold': 0.3,                
         'P_threshold': 0.1,
         'S_threshold': 0.1, 
         'number_of_plots': 10,
         'plot_mode': 'time_frequency',
         'normalization_mode': 'std',
         'batch_size': 500,
         'overlap': 0.3,
         'gpuid': None,
         'gpu_limit': None}
overwrite = False
"""
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
        
        print(f'{len(predD)},{len(predP)},{len(predS)}')
"""
# ONNX port
print('ONNX:')
sess_options = onnxruntime.SessionOptions()
sess = onnxruntime.InferenceSession('eqt_model.onnx', sess_options)

out_dir = os.path.join(os.getcwd(), str(args['output_dir']))
if os.path.isdir(out_dir):
    # print('============================================================================')        
    # print(f' *** {out_dir} already exists!')
    print(f"*** {out_dir} already exists!")
    if overwrite == True:
        inp = "y"
        print("Overwriting your previous results")
    else:
        inp = input(" --> Type (Yes or y) to create a new empty directory! This will erase your previous results so make a copy if you want them.")
    if inp.lower() == "yes" or inp.lower() == "y":
        shutil.rmtree(out_dir)  
        os.makedirs(out_dir) 
    else:
        print("Okay.")
        sys.exit(1)

if platform.system() == 'Windows':
    station_list = [ev.split(".")[0] for ev in listdir(args['input_dir']) if ev.split("\\")[-1] != ".DS_Store"]
else:
    station_list = [ev.split(".")[0] for ev in listdir(args['input_dir']) if ev.split("/")[-1] != ".DS_Store"]

station_list = sorted(set(station_list))

for ct, st in enumerate(station_list):
    # create output directories
    save_dir = os.path.join(out_dir, str(st)+'_outputs')
    save_figs = os.path.join(save_dir, 'figures') 
    if os.path.isdir(save_dir):
        shutil.rmtree(save_dir)  
    os.makedirs(save_dir) 
    if args['number_of_plots']:
        os.makedirs(save_figs)
        
    plt_n = 0            
    csvPr_gen = open(os.path.join(save_dir,'X_prediction_results.csv'), 'w')          
    predict_writer = csv.writer(csvPr_gen, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    predict_writer.writerow(['file_name', 
                             'network',
                             'station',
                             'instrument_type',
                             'station_lat',
                             'station_lon',
                             'station_elv',
                             'event_start_time',
                             'event_end_time',
                             'detection_probability',
                             'detection_uncertainty', 
                             'p_arrival_time',
                             'p_probability',
                             'p_uncertainty',
                             'p_snr',
                             's_arrival_time',
                             's_probability',
                             's_uncertainty',
                             's_snr'
                                 ])  
    csvPr_gen.flush()

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

        all_outs = []

        # ONNX model predict_generator monkey typing
        out_pred_generator = iter_sequence_infinite(pred_generator)
        steps_done = 0
        steps = len(pred_generator)
        while steps_done < steps:
            generator_output = next(out_pred_generator)

            x = generator_output
            x_test = list(x.values())[0].astype(np.float32)
            outs = sess.run(None, input_feed={'input': x_test})
        
            if not all_outs:
                for out in outs:
                    all_outs.append([])

            for i, out in enumerate(outs):
                all_outs[i].append(out)

            steps_done += 1

        results = [np.concatenate(out) for out in all_outs]
        #print(f'{len(results[0])},{len(results[1])},{len(results[2])}')
        predD, predP, predS  = results[0], results[1], results[2]
        detection_memory = []
        for ix in range(len(predD)):
            matches, pick_errors, yh3 =  _picker(args, predD[ix][:, 0], predP[ix][:, 0], predS[ix][:, 0])        
            if (len(matches) >= 1) and ((matches[list(matches)[0]][3] or matches[list(matches)[0]][6])):
                snr = [_get_snr(data_set[meta["trace_start_time"][ix]], matches[list(matches)[0]][3], window = 100), _get_snr(data_set[meta["trace_start_time"][ix]], matches[list(matches)[0]][6], window = 100)]
                pre_write = len(detection_memory)
                detection_memory=_output_writter_prediction(meta, predict_writer, csvPr_gen, matches, snr, detection_memory, ix)
                post_write = len(detection_memory)
                if plt_n < args['number_of_plots'] and post_write > pre_write:
                    _plotter_prediction(data_set[meta["trace_start_time"][ix]], args, save_figs, predD[ix][:, 0], predP[ix][:, 0], predS[ix][:, 0], meta["trace_start_time"][ix], matches)
                    plt_n += 1  
