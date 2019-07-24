#!/usr/bin/python
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

import numpy as np
import scipy.io as sio


def convert_memory_ordering_f2c(array):
    if np.isfortran(array) is True:
        return np.ascontiguousarray(array)
    else:
        return array


def load_training_data(training_file='mnistdata.mat'):
    '''Load training data (mnistdata.mat) and return (inputs, labels).

    inputs: numpy array with size (5000, 400).
    labels: numpy array with size (5000, 1).

    The training data is from Andrew Ng's exercise of the Coursera
    machine learning course (ex4data1.mat).
    '''
    # FIXME: Endian issue
    #   This issue leads an exception "KeyError: '<d'" in execution.
    #
    #   Workaround: indicate type of numpy array explicitly.
    #
    #   Reference:
    #   https://groups.google.com/forum/#!searchin/mpi4py/%22%3Cd%22$20scipy/mpi4py/8gOVvT4ObvU/9gHKOl-jy88J
 
    # FIXME: Memory alignment of input matrices returned by
    #        scipy.io.loadmat is false.
    #
    #   This issue leads Theano to complain that "The numpy.ndarray
    #   object is not aligned.  Theano C code does not support that."
    #
    #   Workaround: ensure the numpy array to be aligned.
    #
    #   Reference:
    #   http://stackoverflow.com/questions/36321400/strange-typeerror-with-theano/36323861
 
    # FIXME: Memory ordering of input matrices returned by
    #        scipy.io.loadmat is Fortran-ordering.
    #
    #   This leads the potential issue that matrix operations might
    #   return unexpected results.
    #   
    #   Current solution is to ensure the loaded external data to use
    #   C-ordering, aka convert its ordering manually.
    training_data = sio.loadmat(training_file)
    inputs = training_data['X'].astype('f8')
    inputs = convert_memory_ordering_f2c(inputs)
    labels = training_data['y'].reshape(training_data['y'].shape[0])
    labels = convert_memory_ordering_f2c(labels)
    return (inputs, labels)


def load_weights(weight_file='mnistweights.mat'):
    '''Load training data (mnistdata.mat) and return (inputs, labels).

    The weights file is from Andrew Ng's exercise of the Coursera
    machine learning course (ex4weights.mat).
    '''
    weights = sio.loadmat(weight_file)
    theta1 = convert_memory_ordering_f2c(weights['Theta1'].astype('f8'))  # size: 25 entries, each has 401 numbers
    theta2 = convert_memory_ordering_f2c(weights['Theta2'].astype('f8'))  # size: 10 entries, each has  26 numbers
    return (theta1, theta2)


def rand_init_weights(size_in, size_out):
    epsilon_init = 0.12
    return np.random.rand(size_out, 1 + size_in) * 2 * epsilon_init - epsilon_init
