from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os
import numpy as np

flags = tf.app.flags
FLAGS = flags.FLAGS
SEED = 2017

def read_and_decode(filename_queue, batch_size, mode):
    image_size = 160
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features = {
            'index': tf.FixedLenFeature([], tf.int64),
            'image_raw': tf.FixedLenFeature([], tf.string)
        })

    # index
    index = tf.cast(features['index'], tf.int32)
    index = tf.reshape(index, [1])

    # Convert from a scalar string tensor (whose single string has
    # length ) to a uint8 tensor with shape [size*size*3].
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image = tf.reshape(image, [image_size, image_size, 3])
    # convert from [0, 255] to [0.0, 1.0] floats
    # image = tf.cast(image, tf.float32) * (1. / 255)

    if mode == 'train':
        index, image = tf.train.shuffle_batch(
            [index, image],
            batch_size=batch_size,
            capacity=5000,
            min_after_dequeue=2000)

    elif mode == 'feature':
        index, image = tf.train.batch(
            [index, image],
            batch_size=batch_size,
            capacity=5000)
    return index, image

def inputs(filename, batch_size=32, mode='feature'):
    """ Reads input data num_epochs times.
    Args:
        batch_size: Number of examples per returned batch.
        mode: 'train' for random or 'feature' for FIFO.
    """
    with tf.variable_scope(mode+'_input', reuse=True):
        filename_queue = tf.train.string_input_producer([filename], shuffle=False)
        index, image = read_and_decode(filename_queue, batch_size, mode)

    return index, image