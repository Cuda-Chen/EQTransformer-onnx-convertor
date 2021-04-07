import h5py
import numpy as np
import keras
import matplotlib
import matplotlib.pyplot as plt
from keras import Model
from keras.models import load_model
from EQTransformer.core.EqT_utils import f1, SeqSelfAttention, FeedForward, LayerNormalization

# Override matplotlib backend setting to Tkagg
# as EQTRansformer sets the backend to agg,
# which will not show the image 
matplotlib.use("Tkagg")

def show_activation_map(model_name, file_name, trace_name):
    dtfl = h5py.File(file_name, 'r')
    dataset = dtfl.get('data/' + trace_name)
    data = np.array(dataset)

    # pre-processing of trace here in the future

    # load model
    """model = load_model('EqT_model.h5',
                        custom_objects={
                            'SeqSelfAttention': SeqSelfAttention, 
                            'FeedForward': FeedForward,
                            'LayerNormalization': LayerNormalization, 
                            'f1': f1})"""
    #model = Model(inputs=model.input,outputs=model.get_layer(name='attentionS').output)
                  #outputs=[model.output, 
                     #model.get_layer(name='layer_normalization_4').output,
                     #model.get_layer(name='attentionP').output,
                     #model.get_layer(name='attentionS').output])

    # make prediction
    #input_data = np.expand_dims(data, axis=0)
    #print(data.shape)
    #outputs = model.predict(data)
    #print(outputs.shape)

    # plot the waveform and the activation heatmap
    fig = plt.figure()
    ax = fig.add_subplot(311)
    plt.plot(data[:,0], 'k')
    plt.rcParams["figure.figsize"] = (8, 5)
    plt.tight_layout()
    ymin, ymax = ax.get_ylim()
    plt.ylabel('Amplitude counts', fontsize=12)
    ax.set_xticklabels([])   

    ax = fig.add_subplot(312)
    plt.plot(data[:,1], 'k')
    plt.rcParams["figure.figsize"] = (8, 5)
    plt.tight_layout()
    ymin, ymax = ax.get_ylim()
    plt.ylabel('Amplitude counts', fontsize=12)
    ax.set_xticklabels([])

    ax = fig.add_subplot(313)
    plt.plot(data[:,2], 'k')
    plt.rcParams["figure.figsize"] = (8, 5)
    plt.tight_layout()
    ymin, ymax = ax.get_ylim()
    plt.ylabel('Amplitude counts', fontsize=12)
    ax.set_xticklabels([])
    plt.show()

if __name__ == '__main__':
    model_name = "EqT_model.h5"
    file_name = "/home/cudachen/Documents/stead_datasets/chunk5/chunk5.hdf5"
    trace_name = "MOD.BK_20141110123806_EV"
    show_activation_map(model_name, file_name, trace_name)
