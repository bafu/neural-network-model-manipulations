import numpy as np

from modelman.io import *
from modelman.utils import *


def _predict(model, inputs):
    theta1, theta2 = model
    a1 = np.insert(inputs, 0, 1, axis=1)  # add bias, (5000,401)
    a2 = np.dot(a1, theta1.T)  # (5000,401) x (401,25)
    a2 = sigmoid(a2)
    a2 = np.insert(a2, 0, 1, axis=1)  # add bias, (5000,26)
    a3 = np.dot(a2, theta2.T)  # (5000,26) x (26,10)
    a3 = sigmoid(a3)  # (5000,10)
    return [i.argmax()+1 for i in a3]


def test_predict_default_model():
    inputs, labels = load_training_data()
    theta1, theta2 = load_weights()
    model = (theta1, theta2)

    outputs = _predict(model, inputs)

    correct_prediction = 0
    for i, predict in enumerate(outputs):
        if predict == labels[i]:
           correct_prediction += 1
    precision = float(correct_prediction) / len(labels)
    # 0.9752
    print('precision: {}'.format(precision))
