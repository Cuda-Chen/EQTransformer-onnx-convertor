import keras
from keras import Model
from keras.models import load_model

from EQTransformer.core.EqT_utils import f1, SeqSelfAttention, FeedForward, LayerNormalization
import h5py
import numpy as np
import matplotlib.pyplot as plt

def show_activation_map(model_name, file_name, trace_name):
    dtfl = h5py.File(file_name, 'r')
    dataset = dtfl.get('data/' + trace_name)
    data = np.array(dataset)

    # pre-processing of trace here in the future

    # load model
    model = load_model('EqT_model.h5',
                        custom_objects={
                            'SeqSelfAttention': SeqSelfAttention, 
                            'FeedForward': FeedForward,
                            'LayerNormalization': LayerNormalization, 
                            'f1': f1})
    #model = Model(inputs=model.input,outputs=model.get_layer(name='attentionS').output)
                  #outputs=[model.output, 
                     #model.get_layer(name='layer_normalization_4').output,
                     #model.get_layer(name='attentionP').output,
                     #model.get_layer(name='attentionS').output])

    # make prediction
    data = np.expand_dims(data, axis=0)
    #print(data.shape)
    outputs = model.predict(data)
    print(outputs.shape)

if __name__ == '__main__':
    model_name = "EqT_model.h5"
    file_name = "/home/cudachen/Documents/stead_datasets/chunk5/chunk5.hdf5"
    trace_name = "MOD.BK_20141110123806_EV"
    show_activation_map(model_name, file_name, trace_name)
