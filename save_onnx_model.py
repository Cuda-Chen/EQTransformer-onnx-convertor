import keras
from keras.models import load_model
from EQTransformer.core.EqT_utils import f1, SeqSelfAttention, FeedForward, LayerNormalization
import keras2onnx

def convert2ONNX(input_model_name, output_model_name):
    keras.backend.set_learning_phase(0)
    model = load_model(input_model_name,
                       custom_objects={
                            'SeqSelfAttention': SeqSelfAttention, 
                            'FeedForward': FeedForward,
                            'LayerNormalization': LayerNormalization, 
                            'f1': f1})

    onnx_model = keras2onnx.convert_keras(model, debug_mode=1)
    keras2onnx.save_model(onnx_model, output_model_name)

if __name__ == '__main__':
    convert2ONNX('EqT_model.h5', 'eqt_model.onnx')
