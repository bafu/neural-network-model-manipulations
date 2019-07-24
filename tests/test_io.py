import numpy as np

from modelman.io import *


def assert_float_equal(a, b):
    threshold = 0.0001
    return True if abs(a-b) < threshold else False


def test_mnist_data():
    inputs_manualsum = 262678.26015968103
    labels_manualsum = 27500
    inputs, labels = load_training_data()

    print('sum of inputs: %f' % np.sum(inputs))
    assert_float_equal(np.sum(inputs), inputs_manualsum)

    print('sum of lables: %d' % np.sum(labels))
    assert np.sum(labels) == labels_manualsum


def test_mnist_model():
    theta1_manualsum = 9.2426439281200246
    theta2_manualsum = -100.08344384930396
    theta1, theta2 = load_weights()
    assert_float_equal(np.sum(theta1), theta1_manualsum)
    assert_float_equal(np.sum(theta2), theta2_manualsum)
