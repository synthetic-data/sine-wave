# -*- coding: utf-8 -*-
"""Some code generating a dataset for LSTM training
"""
from typing import List
from math import pi
from random import uniform
import numpy as np
import tensorflow as tf


def random_phases(batch_size: int, phase: List = None) -> List:
    phases = [uniform(phase[0], phase[1]) for n in range(batch_size)]
    return phases


def tensorflow_sine_wave_data(n_half_periods: int, npoints: int, batch_size: int, phases: List = None):
    """
    A sine wave dataset.
    :param n_half_periods: number of half periods in order to imitate the different phase at the end;
    :param npoints: number of points in sequence;
    :param batch_size: batch size after which the weights will be updated;
    :param phase: list of 'from' 'until' phases in radian, if None (default) random between 1 and 2pi;
    :return: Dataset
    """

    # phases for the individual sequences in the batch
    phases_for_batch = random_phases(batch_size, phases)

    layer = []
    for phase in phases_for_batch:
        duration = n_half_periods * pi
        t = np.arange(start=phase,
                         stop=phase + duration,
                         step=duration / (npoints - 1),
                         dtype=np.float)
        x1 = np.sin(t)
        x2 = t
        if layer:
            pass
        else:
            layer = t
        print('ok')
        # rold = np.roll(a=data, shift=-1, axis=0)
    # seri = Series(data)
    # print(series)
    # # prepare data for normalization

    # shif = seri.shift(1)
    # npsh = shif.to_numpy()
    # values = series.values
    # values = values.reshape((len(values), 1))
    # curve = np.sin(data[:, np.newaxis])
    # sin = np.sin(data)
    # rols = np.sin(rold)

    # arr = np.array(data)
    # val = arr.reshape((len(arr), 1))
    #
    # a = np.arange(6, 10)
    # b = np.arange(12, 17)
    #
    # table = a[:, np.newaxis] * b
    # table = np.column_stack((data, sin, rols)) # PyTorch dataset time first
    # table_tf = np.swapaxes(table, 0, 1)
    # newest_table = np.row_stack((table, b))

    # train_dataset = tf.data.Dataset.from_tensor_slices((train_examples, train_labels))
    # test_dataset = tf.data.Dataset.from_tensor_slices((test_examples, test_labels))

    print('ok')
    return


def pytorch_sine_wave_data():
    dataset = {}
    return dataset

def test_sequence():
    sequence = []
    return sequence


if __name__ == '__main__':
    # the library that we are preparing data for
    library = 'TensorFlow' # "PyTorch"

    if library.startswith('TensorFlow'):
        tensorflow_sine_wave_data(n_half_periods=3, npoints=16, batch_size=8, phases=[0, pi/2])
    elif library.startswith('PyTorch'):
        pytorch_sine_wave_data()
    else:
        print('I do not know the library that you are talking about! ')
    print('\ndone')
