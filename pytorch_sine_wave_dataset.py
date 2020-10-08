# -*- coding: utf-8 -*-
"""Some code generating a dataset for LSTM training
"""
from __future__ import print_function, division
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

from typing import List
from math import pi, sin
from random import uniform


class SineWaveDataset(Dataset):
    """Sine wave dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the traces.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.classification_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.classification_frame)

    def __getitem__(self, indx):
        if torch.is_tensor(indx):
            indx = indx.tolist()

        trace_file_name = os.path.join(self.root_dir,
                                self.classification_frame.iloc[indx, 0])
        trace = pd.read_csv(trace_file_name)
        classvector = self.classification_frame.iloc[indx, 1:]
        classvector = np.array([classvector])
        classvector = classvector.astype('float').reshape(-1, 2)
        sample = {'player_trace': trace, 'classvector': classvector}

        if self.transform:
            sample = self.transform(sample)

        return sample


def random_phases(batch_size: int, phase: List = None) -> List:
    """ list of equidistributed random phases for sine waves
    :param batch_size:
    :param phase:
    :return:
    """
    phases = [uniform(phase[0], phase[1]) for n in range(batch_size)]
    return phases


def tensorflow_sine_wave_data(n_half_periods: int, npoints: int, batch_size: int, phases: List = None):
    """ A sine wave dataset and labels
    :param n_half_periods: number of half periods in order to imitate the different phase at the end;
    :param npoints: number of points in sequence;
    :param batch_size: batch size after which the weights will be updated;
    :param phase: list of 'from' 'until' phases in radian, if None (default) random between 1 and 2pi;
    :return: Dataset
    """

    # phases for the individual sequences in the batch
    phases_for_batch = random_phases(batch_size, phases)

    data_stack  = []
    label_stack = []
    for phase in phases_for_batch:
        layer = []
        duration = n_half_periods * pi
        t = np.arange(start=phase,
                      stop=phase + duration,
                      step=duration / npoints,
                      dtype=np.float)
        x1 = np.sin(t)
        next = sin(phase+duration)
        phase_list = [phase for i in range(npoints)]
        x2 = np.array(phase_list, dtype=np.float)
        layer.append(x1)
        layer.append(x2)
        # rolled_t = np.roll(a=t, shift=-1, axis=0)
        data_stack.append(layer)
        label_stack.append(next)
    examples = np.asarray(data_stack)
    labels = np.asarray(label_stack)
    return examples, labels


def pytorch_sine_wave_data(n_half_periods: int, npoints: int, batch_size: int, phases: List = None):
    """ A sine wave dataset and labels
    :param n_half_periods: number of half periods in order to imitate the different phase at the end;
    :param npoints: number of points in sequence;
    :param batch_size: batch size after which the weights will be updated;
    :param phase: list of 'from' 'until' phases in radian, if None (default) random between 1 and 2pi;
    :return: Dataset, Labels list
    """

    # phases for the individual sequences in the batch
    phases_for_batch = random_phases(batch_size, phases)

    data_stack  = []
    label_stack = []
    for phase in phases_for_batch:
        layer = []
        duration = n_half_periods * pi
        t = np.arange(start=phase,
                      stop=phase + duration,
                      step=duration / npoints,
                      dtype=np.float)
        x1 = np.sin(t)
        next = sin(phase+duration)
        phase_list = [phase for i in range(npoints)]
        x2 = np.array(phase_list, dtype=np.float)
        layer.append(x1)
        layer.append(x2)
        # rolled_t = np.roll(a=t, shift=-1, axis=0)
        data_stack.append(layer)
        label_stack.append(next)
    examples = np.asarray(data_stack)
    labels = np.asarray(label_stack)
    return examples, labels


if __name__ == '__main__':
    # the library that we are preparing data for
    library = 'TensorFlow' # "PyTorch"
    dataset_path = '/media/alxfed/data/sine_wave'
    number_of_half_periods = 2
    number_of_time_points = 16
    batch_size = 8
    phases_range = [0, pi/2]

    if library.startswith('TensorFlow'):
        train_examples, train_labels = tensorflow_sine_wave_data(n_half_periods=number_of_half_periods,
                                                                 npoints=number_of_time_points,
                                                                 batch_size=batch_size,
                                                                 phases=phases_range)
        train_dataset = tf.data.Dataset.from_tensor_slices((train_examples, train_labels))

        tf.data.experimental.save(train_dataset, dataset_path)
        # spec = train_dataset.element_spec
        # new_dataset = tf.data.experimental.load(dataset_path,
        #                                         element_spec=spec)
        # for elem in new_dataset:
        #     print(elem)

        test_examples, test_labels = tensorflow_sine_wave_data(n_half_periods=number_of_half_periods,
                                                                 npoints=number_of_time_points,
                                                                 batch_size=batch_size,
                                                                 phases=phases_range)
        test_dataset = tf.data.Dataset.from_tensor_slices((test_examples, test_labels))

    elif library.startswith('PyTorch'):
        train_examples, train_labels = pytorch_sine_wave_data(n_half_periods=number_of_half_periods,
                                                                 npoints=number_of_time_points,
                                                                 batch_size=batch_size,
                                                                 phases=phases_range)
    else:
        print('I do not know the library that you are talking about! ')
    print('\ndone')

'''
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
'''