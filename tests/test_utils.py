from modelman.io import *
from modelman.utils import *


def assert_float_equal(a, b):
    threshold = 0.0001
    return True if abs(a-b) < threshold else False


def test_sigmoid():
    inputs, labels = load_training_data()
    #print(sigmoid(inputs))
    pass
