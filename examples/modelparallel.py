#!/usr/bin/python
#
# MNIST digit recognizer in distributed GPU mode.
#
# This program is based on the prior MNIST digit recognizer, and change
# the matrix computing codes from using local CPU to distributed GPUs.
#
# To distribute data to worknodes, MPI (mpi4py) is used.  To execute
# matrix operations, Theano is used.
#
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import functools
import numpy as np
import math
import os
import time

from mpi4py import MPI

from .io import *
from .utils import *


if os.getenv('MNISTNN_PARALLEL') == 'yes':
    Distributed = True
    print('Parallelism: yes')
else:
    Distributed = False
    print('Parallelism: no')


# Init MPI
comm = MPI.COMM_WORLD

# Structure of the 3-layer neural network.
Input_layer_size = 400
Hidden_layer_size = 25
Output_layer_size = 10


def cost_function(theta1, theta2, input_layer_size, hidden_layer_size, output_layer_size, inputs, labels, regular=0):
    '''Calculate cost and gradient by model parallelism.

    Note: theta1, theta2, inputs, labels are numpy arrays:

        theta1: (25, 401)
        theta2: (10, 26)
        inputs: (5000, 400)
        labels: (5000, 1)

    return: cost and gradients
    '''
    # Back propagation: calculate gradiants
    theta1_grad = np.zeros_like(theta1)  # 25x401
    theta2_grad = np.zeros_like(theta2)  # 10x26

    for index, training_data in enumerate(inputs):
        # Forward propagation: get h_theta(x).
        input_layer_result = np.insert(  # add bias, 1x401 
            # reshape from list to matrix
            np.reshape(training_data, (1, len(training_data))),
            0, 1, axis=1)

        hidden_layer_result = Matrix_dot(input_layer_result, theta1.T)
        hidden_layer_result = sigmoid(hidden_layer_result)
        hidden_layer_result = np.insert(hidden_layer_result, 0, 1, axis=1)  # add bias, 1x26

        output_layer_result = Matrix_dot(hidden_layer_result, theta2.T)
        output_layer_result = sigmoid(output_layer_result)  # 1x10

        # Change label to a 1x10 matrix.
        # Label value is from 1 to 10, and 10 represents zero.
        label_index = labels[index] - 1
        outputs = np.zeros((1, output_layer_size))  # 1x10
        outputs[0][label_index] = 1

        # calculate delta3
        delta3 = (output_layer_result - outputs).T  # 10x1

        # calculate delta2
        z2 = Matrix_dot(theta1, input_layer_result.T)  # (25,401) x (401,1)
        z2 = np.insert(z2, 0, 1, axis=0)  # add bias, (26,1)
        delta2 = np.multiply(
            Matrix_dot(theta2.T, delta3),  # (26,10) x (10,1)
            sigmoid_gradient(z2)  # (26,1)
        )
        delta2 = delta2[1:]  # (25,1)

        # calculate gradients of theta1 and theta2
        # (25,401) = (25,1) x (1,401)
        theta1_grad += Matrix_dot(delta2, input_layer_result)
        # (10,26) = (10,1) x (1,26)
        theta2_grad += Matrix_dot(delta3, hidden_layer_result)

    theta1_grad /= len(inputs)
    theta2_grad /= len(inputs)

    # Calculate cost.
    input_layer = np.insert(inputs, 0, 1, axis=1)  # add bias, 5000x401

    hidden_layer = Matrix_dot(input_layer, theta1.T)
    hidden_layer = sigmoid(hidden_layer)
    hidden_layer = np.insert(hidden_layer, 0, 1, axis=1)  # add bias, 5000x26

    output_layer = Matrix_dot(hidden_layer, theta2.T)  # 5000x10
    output_layer = sigmoid(output_layer)

    cost = 0.0
    for training_index in xrange(len(inputs)):
        # transform label y[i] from a number to a vector.
        #
        # Note:
        #   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        #    1  2  3  4  5  6  7  8  9 10
        #
        #   if y[i] is 0 -> [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        #   if y[i] is 1 -> [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        outputs = [0] * output_layer_size
        outputs[labels[training_index]-1] = 1

        for k in xrange(output_layer_size):
            error = -outputs[k] * math.log(output_layer[training_index][k]) - (1 - outputs[k]) * math.log(1 - output_layer[training_index][k])
            cost += error
    cost /= len(inputs)

    return cost, (theta1_grad, theta2_grad)


def gradient_descent(inputs, labels, learningrate=0.8, iteration=50):
    '''
    @return cost and trained model (weights).
    '''
    if Distributed is True:
        if comm.rank == 0:
            theta1 = rand_init_weights(Input_layer_size, Hidden_layer_size)
            theta2 = rand_init_weights(Hidden_layer_size, Output_layer_size)
        else:
            theta1 = np.zeros((Hidden_layer_size, Input_layer_size + 1))
            theta2 = np.zeros((Output_layer_size, Hidden_layer_size + 1))
        comm.Barrier()
        if comm.rank == 0:
            time_bcast_start = time.time()
        comm.Bcast([theta1, MPI.DOUBLE])
        comm.Barrier()
        comm.Bcast([theta2, MPI.DOUBLE])
        if comm.rank == 0:
            time_bcast_end = time.time()
            print('\tBcast theta1 and theta2 uses {} secs.'.format(time_bcast_end - time_bcast_start))
    else:
        theta1 = rand_init_weights(Input_layer_size, Hidden_layer_size)
        theta2 = rand_init_weights(Hidden_layer_size, Output_layer_size)

    cost = 0.0
    for i in xrange(iteration):
        time_iter_start = time.time()

        if Distributed is True:
            # Scatter training data and labels.
            sliced_inputs = np.asarray(np.split(inputs, comm.size))
            sliced_labels = np.asarray(np.split(labels, comm.size))
            inputs_buf = np.zeros((len(inputs)/comm.size, Input_layer_size))
            labels_buf = np.zeros((len(labels)/comm.size), dtype='uint8')

            comm.Barrier()
            if comm.rank == 0:
                time_scatter_start = time.time()
            comm.Scatter(sliced_inputs, inputs_buf)
            if comm.rank == 0:
                time_scatter_end = time.time()
                print('\tScatter inputs uses {} secs.'.format(time_scatter_end - time_scatter_start))

            comm.Barrier()
            if comm.rank == 0:
                time_scatter_start = time.time()
            comm.Scatter(sliced_labels, labels_buf)
            if comm.rank == 0:
                time_scatter_end = time.time()
                print('\tScatter labels uses {} secs.'.format(time_scatter_end - time_scatter_start))

            # Calculate distributed costs and gradients of this iteration
            # by cost function.
            comm.Barrier()
            cost, (theta1_grad, theta2_grad) = cost_function(theta1, theta2,
                Input_layer_size, Hidden_layer_size, Output_layer_size,
                inputs_buf, labels_buf, regular=0)

            # Gather distributed costs and gradients.
            comm.Barrier()
            cost_buf = [0] * comm.size
            try:
                cost_buf = comm.gather(cost)
                cost = sum(cost_buf) / len(cost_buf)
            except TypeError as e:
                print('[{0}] {1}'.format(comm.rank, e))

            theta1_grad_buf = np.asarray([np.zeros_like(theta1_grad)] * comm.size)
            comm.Barrier()
            if comm.rank == 0:
                time_gather_start = time.time()
            comm.Gather(theta1_grad, theta1_grad_buf)
            if comm.rank == 0:
                time_gather_end = time.time()
                print('\tGather theta1 uses {} secs.'.format(time_gather_end - time_gather_start))
            comm.Barrier()
            theta1_grad = functools.reduce(np.add, theta1_grad_buf) / comm.size

            theta2_grad_buf = np.asarray([np.zeros_like(theta2_grad)] * comm.size)
            comm.Barrier()
            if comm.rank == 0:
                time_gather_start = time.time()
            comm.Gather(theta2_grad, theta2_grad_buf)
            if comm.rank == 0:
                time_gather_end = time.time()
                print('\tGather theta2 uses {} secs.'.format(time_gather_end - time_gather_start))
            comm.Barrier()
            theta2_grad = functools.reduce(np.add, theta2_grad_buf) / comm.size
        else:
            cost, (theta1_grad, theta2_grad) = cost_function(theta1, theta2,
                Input_layer_size, Hidden_layer_size, Output_layer_size,
                inputs, labels, regular=0)

        theta1 -= learningrate * theta1_grad
        theta2 -= learningrate * theta2_grad

        if Distributed is True:
           # Sync-up weights for distributed worknodes.
           comm.Bcast([theta1, MPI.DOUBLE])
           comm.Bcast([theta2, MPI.DOUBLE])
           comm.Barrier()

        time_iter_end = time.time()
        if comm.rank == 0:
            print('Iteration {0} (learning rate {2}, iteration {3}), cost: {1}, time: {4}'.format(
                i+1, cost, learningrate, iteration, time_iter_end - time_iter_start)
            )
    return cost, (theta1, theta2)


def train(inputs, labels, learningrate=0.8, iteration=50):
    cost, model = gradient_descent(inputs, labels, learningrate, iteration)
    return model


def predict(model, inputs):
    theta1, theta2 = model
    a1 = np.insert(inputs, 0, 1, axis=1)  # add bias, (5000,401)
    a2 = np.dot(a1, theta1.T)  # (5000,401) x (401,25)
    a2 = sigmoid(a2)
    a2 = np.insert(a2, 0, 1, axis=1)  # add bias, (5000,26)
    a3 = np.dot(a2, theta2.T)  # (5000,26) x (26,10)
    a3 = sigmoid(a3)  # (5000,10)
    return [i.argmax()+1 for i in a3]


if __name__ == '__main__':
    # Note: There are 10 units which present the digits [1-9, 0]
    # (in order) in the output layer.
    inputs, labels = load_training_data()

    # train the model from scratch and predict based on it
    model = train(inputs, labels, learningrate=0.1, iteration=60)

    # Load pretrained weights for debugging precision.
    # The precision will be around 97% (0.9756).
    #weights = load_weights()
    #theta1 = weights[0]  # size: 25 entries, each has 401 numbers
    #theta2 = weights[1]  # size: 10 entries, each has  26 numbers
    #model = (theta1, theta2)
    #cost, (theta1_grad, theta2_grad) = cost_function(theta1, theta2, Input_layer_size, Hidden_layer_size, Output_layer_size, inputs, labels, regular=0)
    #print('cost:', cost)

    outputs = predict(model, inputs)

    correct_prediction = 0
    for i, predict in enumerate(outputs):
        if predict == labels[i]:
            correct_prediction += 1
    precision = float(correct_prediction) / len(labels)
    print('precision: {}'.format(precision))
