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
import time
import pandas as pd
import json
import obspy
from obspy import read

"""params_pred = {'batch_size': 500,
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
         'gpu_limit': None}"""

params_pred = {'batch_size': 1,
               'norm_mode': 'std'}
args = {'input_dir': 'test_dataset',   
         'input_model': 'EqT_model.h5',
         'stations_json': 'test_dataset/test_dataset.json',
         'output_dir': 'test_detections',
         'loss_weights': [0.02, 0.40, 0.58],          
         'detection_threshold': 0.3,                
         'P_threshold': 0.1,
         'S_threshold': 0.1, 
         'number_of_plots': 10,
         'plot_mode': 'time_frequency',
         'normalization_mode': 'std',
         'batch_size': 1,
         'overlap': 0.3,
         'gpuid': None,
         'gpu_limit': None}
overwrite = False

# ONNX model predict_generator monkey typing
def onnx_predict_generator(pred_generator):
    all_outs = []
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
    return results[0], results[1], results[2]

def mseed2nparry_one_minute(args, matching, time_slots, comp_types, st_name):
    ' read miniseed files and from a list of string names and returns 3 dictionaries of numpy arrays, meta data, and time slice info'

    json_file = open(args['stations_json'])
    stations_ = json.load(json_file)

    st = obspy.core.Stream()
    tsw = False
    for m in matching:
        temp_st = read(os.path.join(str(args['input_dir']), m),debug_headers=True)
        if tsw == False and temp_st:
            tsw = True
            for tr in temp_st:
                time_slots.append((tr.stats.starttime, tr.stats.endtime))
        try:
            temp_st.merge(fill_value=0)
        except Exception:
            temp_st =_resampling(temp_st)
            temp_st.merge(fill_value=0)
        temp_st.detrend('demean')
        st += temp_st

    st.filter(type='bandpass', freqmin = 1.0, freqmax = 45, corners=2, zerophase=True)
    st.taper(max_percentage=0.001, type='cosine', max_length=2)
    if len([tr for tr in st if tr.stats.sampling_rate != 100.0]) != 0:
        try:
            st.interpolate(100, method="linear")
        except Exception:
            st=_resampling(st)

    st.trim(min([tr.stats.starttime for tr in st]), max([tr.stats.endtime for tr in st]), pad=True, fill_value=0)

    start_time = st[0].stats.starttime
    end_time = st[0].stats.endtime

    meta = {"start_time":start_time,
            "end_time": end_time,
            "trace_name":m
             }

    chanL = [tr.stats.channel[-1] for tr in st]
    comp_types.append(len(chanL))
    tim_shift = int(60-(args['overlap']*60))
    next_slice = start_time+60

    data_set={}

    sl = 0; st_times = []
    #while next_slice <= end_time:
    npz_data = np.zeros([6000, 3])
    st_times.append(str(start_time).replace('T', ' ').replace('Z', ''))
    w = st.slice(start_time, next_slice)
    if 'Z' in chanL:
        npz_data[:,2] = w[chanL.index('Z')].data[:6000]
    if ('E' in chanL) or ('1' in chanL):
        try:
            npz_data[:,0] = w[chanL.index('E')].data[:6000]
        except Exception:
            npz_data[:,0] = w[chanL.index('1')].data[:6000]
    if ('N' in chanL) or ('2' in chanL):
        try:
            npz_data[:,1] = w[chanL.index('N')].data[:6000]
        except Exception:
            npz_data[:,1] = w[chanL.index('2')].data[:6000]

    data_set.update( {str(start_time).replace('T', ' ').replace('Z', '') : npz_data})

    start_time = start_time+tim_shift
    next_slice = next_slice+tim_shift
    sl += 1

    meta["trace_start_time"] = st_times

    try:
        meta["receiver_code"]=st[0].stats.station
        meta["instrument_type"]=st[0].stats.channel[:2]
        meta["network_code"]=stations_[st[0].stats.station]['network']
        meta["receiver_latitude"]=stations_[st[0].stats.station]['coords'][0]
        meta["receiver_longitude"]=stations_[st[0].stats.station]['coords'][1]
        meta["receiver_elevation_m"]=stations_[st[0].stats.station]['coords'][2]
    except Exception:
        meta["receiver_code"]=st_name
        meta["instrument_type"]=stations_[st_name]['channels'][0][:2]
        meta["network_code"]=stations_[st_name]['network']
        meta["receiver_latitude"]=stations_[st_name]['coords'][0]
        meta["receiver_longitude"]=stations_[st_name]['coords'][1]
        meta["receiver_elevation_m"]=stations_[st_name]['coords'][2]

    return meta, time_slots, comp_types, data_set

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
#sess = onnxruntime.InferenceSession('eqt_optimized.onnx', sess_options)
#args['output_dir'] = 'detections_onnx_optimized'

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
    print(f"Started working on {st}, {ct+1} out of {len(station_list)} ...", flush=True)

    start_Predicting = time.time()

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
        print(f'{month}', flush=True)
        meta, time_slots, comp_types, data_set = mseed2nparry_one_minute(args, matching, time_slots, comp_types, st)

        pred_generator = PreLoadGeneratorTest(meta["trace_start_time"], data_set, **params_pred)

        """all_outs = []

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
        predD, predP, predS  = results[0], results[1], results[2]"""
        predD, predP, predS = onnx_predict_generator(pred_generator)
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

    end_Predicting = time.time()
    delta = (end_Predicting - start_Predicting)
    hour = int(delta / 3600)
    delta -= hour * 3600
    minute = int(delta / 60)
    delta -= minute * 60
    seconds = delta

    dd = pd.read_csv(os.path.join(save_dir,'X_prediction_results.csv'))
    print(f"Finished the prediction in: {hour} hours and {minute} minutes and {round(seconds, 2)} seconds.", flush=True)
    print(f'*** Detected: '+str(len(dd))+' events.', flush=True)
    print(' *** Wrote the results into --> " ' + str(save_dir)+' "', flush=True)
