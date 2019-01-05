import tensorflow as tf
from model import make_model
import foolbox

TEMPERATURE = 100
MODEL_PATH = 'models/distilled'

def create():
    sess = tf.Session().__enter__()
    _input = tf.placeholder(tf.float32, (1, 28, 28, 1))

    model = make_model(MODEL_PATH)
    _logits = model(_input)/TEMPERATURE

    # create Foolbox model
    fmodel = foolbox.models.TensorFlowModel(_input, _logits, bounds=(0, 1))
    
    return fmodel