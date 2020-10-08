# -*- coding: utf-8 -*-
"""https://www.tensorflow.org/api_docs/python/tf/data/experimental/make_csv_dataset
"""
import tensorflow as tf

if __name__ == '__main__':
    file_pattern = '/media/alxfed/data/datasets/*.csv'
    batch_size = 8
    dataset = tf.data.experimental.make_csv_dataset(
        file_pattern=file_pattern, batch_size=batch_size, column_names=None, column_defaults=None,
        label_name=None, select_columns=None, field_delim=',', use_quote_delim=True,
        na_value='', header=True, num_epochs=None, shuffle=True,
        shuffle_buffer_size=10000, shuffle_seed=None, prefetch_buffer_size=None,
        num_parallel_reads=None, sloppy=False, num_rows_for_inference=100,
        compression_type=None, ignore_errors=False)

    print('\ndone')