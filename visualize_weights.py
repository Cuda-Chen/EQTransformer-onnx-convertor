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
    model = load_model('EqT_model.h5',
                        custom_objects={
                            'SeqSelfAttention': SeqSelfAttention, 
                            'FeedForward': FeedForward,
                            'LayerNormalization': LayerNormalization, 
                            'f1': f1})
    #model.summary()
    #model = Model(inputs=model.input,outputs=model.get_layer(name='attentionS').output)
                  #outputs=[model.output, 
                     #model.get_layer(name='layer_normalization_4').output,
                     #model.get_layer(name='attentionP').output,
                     #model.get_layer(name='attentionS').output])

    # make prediction
    input_data = np.expand_dims(data, axis=0)
    outputs = model.predict(input_data)

    # plot the waveform and the activation heatmap
    fig = plt.figure()
    ax = fig.add_subplot(311)
    plt.plot(data[:,0]/abs(data[:,0]).max(), 'gray')
    plt.plot(outputs[0][0], '--', color='b')
    plt.rcParams["figure.figsize"] = (8, 8)
    plt.tight_layout()
    plt.ylabel('Normalized\n amplitude', fontsize=12)
    ax.set_xticklabels([])
    ax.set_ylim([-1.1, 1.1])

    ax = fig.add_subplot(312)
    plt.plot(data[:,1]/abs(data[:,1]).max(), 'gray')
    plt.plot(outputs[1][0], '--', color='g')
    plt.rcParams["figure.figsize"] = (8, 8)
    plt.tight_layout()
    plt.ylabel('Normalized\n amplitude', fontsize=12)
    ax.set_xticklabels([])    
    ax.set_ylim([-1.1, 1.1])

    ax = fig.add_subplot(313)
    plt.plot(data[:,2]/abs(data[:,2]).max(), 'gray')
    plt.plot(outputs[2][0], '--', color='r')
    plt.rcParams["figure.figsize"] = (8, 8)
    plt.tight_layout()
    plt.ylabel('Normalized\n amplitude', fontsize=12)
    ax.set_xticklabels([])    
    ax.set_ylim([-1.1, 1.1])
    
    plt.show()

if __name__ == '__main__':
    model_name = "EqT_model.h5"
    file_name = "/home/cudachen/Documents/stead_datasets/chunk5/chunk5.hdf5"
    trace_name = "MOD.BK_20141110123806_EV"
    show_activation_map(model_name, file_name, trace_name)
