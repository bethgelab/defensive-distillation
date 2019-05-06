import tensorflow as tf
from model import make_model
import foolbox
import inspect
import os

TEMPERATURE = 100

def create():
    sess = tf.Session().__enter__()
    _input = tf.placeholder(tf.float32, (None, 28, 28, 1))
    
    local_path = os.path.dirname(os.path.realpath(inspect.stack()[0][1]))
    MODEL_PATH = os.path.join(local_path, 'models/distilled')
    model = make_model(MODEL_PATH)
    _logits = model(_input)/TEMPERATURE

    # create Foolbox model
    fmodel = foolbox.models.TensorFlowModel(_input, _logits, bounds=(0, 1))
    
    return fmodel
