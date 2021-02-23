from EQTransformer.core.EqT_utils import f1, SeqSelfAttention, FeedForward, LayerNormalization
from EQTransformer.core.mseed_predictor import mseed_predictor, _mseed2nparry, PreLoadGeneratorTest
import keras
from keras.models import load_model
import keras2onnx

# original
# detection.ipynb
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
