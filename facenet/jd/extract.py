"""Training a face recognizer with TensorFlow using softmax cross entropy loss
"""
# MIT License
# 
# Copyright (c) 2016 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.append('../src/')
from datetime import datetime
import os.path
import time
import sys
import random
import tensorflow as tf
import numpy as np
import importlib
import argparse
import facenet
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops

def main(args):
    network = importlib.import_module(args.model_def)

    pretrained_model = os.path.expanduser(args.pretrained_model)
    print('Pre-trained model: %s' % pretrained_model)

    with tf.Graph().as_default():
        tf.set_random_seed(args.seed)
        global_step = tf.Varia, label_list = get_image_paths(args.data_file)
        assert len(image_list)>0, 'The dataset should not be empty'

        range_size = array_ops.shape(labels)[0]
        index_queue = tf.train.range_input_producer(range_size, num_epochs=None,
            shuffle=True, seed=None, capacity=32)
        index_dequeue_op = index_queue.dequeue_many(args.batch_size, 'index_dequeue')

        batch_size_placeholder = tf.placeholder(tf.int32, name='batch_size')
        
        phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')
        
        image_paths_placeholder = tf.placeholder(tf.string, shape=(None,1), name='image_paths')
     
        input_queue = data_flow_ops.FIFOQueue(capacity=100000,
                                    dtypes=[tf.string],
                                    shapes=[(1,)],
                                    shared_name=None, name=None)
        enqueue_op = input_queue.enqueue_many([image_paths_placeholder], name='enqueue_op')
        
        nrof_preprocess_threads = 4
        images_and_labels = []
        for _ in range(nrof_preprocess_threads):
            filenames, label = input_queue.dequeue()
            images = []
            for filename in tf.unstack(filenames):
                file_contents = tf.read_file(filename)
                image = tf.image.decode_image(file_contents, channels=3)

    
                #pylint: disable=no-member
                image.set_shape((args.image_size, args.image_size, 3))
                images.append(tf.image.per_image_standardization(image))

    
        image_batch= tf.train.batch_join(
            images_and_labels, batch_size=batch_size_placeholder, 
            shapes=[(args.image_size, args.image_size, 3), ()], enqueue_many=True,
            capacity=4 * nrof_preprocess_threads * args.batch_size,
            allow_smaller_final_batch=True)
        image_batch = tf.identity(image_batch, 'image_batch')
        image_batch = tf.identity(image_batch, 'input')
        
        print('Total number of examples: %d' % len(image_list))
        
        for i in range(2):
            with tf.device('/gpu:%d'%i):
                print('Building training graph')
                _, endpoints = network.inference(image_batch, 1.0, 
                    phase_train=phase_train_placeholder, bottleneck_layer_size=128, 
                    weight_decay=0.0)
                features = endpoints['AvgPool_1a_8x8']

        trainable_variables = tf.trainable_variables()

        # Start running operations on the Graph.
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
        coord = tf.train.Coordinator()
        tf.train.start_queue_runners(coord=coord, sess=sess)

        with sess.as_default():
            if pretrained_model:
                print('Restoring pretrained model: %s' % pretrained_model)
                saver.restore(sess, pretrained_model)

            print('Running extracting features')
            extract_feature(args, sess, image_list, index_dequeue_op, enqueue_op, image_paths_placeholder, 
                phase_train_placeholder, batch_size_placeholder)
def extract_feature(args, sess, image_list, index_dequeue_op, enqueue_op, image_paths_placeholder, 
    phase_train_placeholder, batch_size_placeholder):
    batch_number = 0

    index_epoch = sess.run(index_dequeue_op)
    image_epoch = np.array(image_list)[index_epoch]
    
    # Enqueue one epoch of image paths and labels
    image_paths_array = np.expand_dims(np.array(image_epoch),1)
    sess.run(enqueue_op, {image_paths_placeholder: image_paths_array})

    # Training loop
    total_batch_num = images_num / args.batch_size
    sw = open("feature.txt", 'w+')
    while batch_number < total_batch_num:
        start_time = time.time()
        feed_dict = {phase_train_placeholder:False, batch_size_placeholder:args.batch_size}
        fea = sess.run([features], feed_dict=feed_dict)
        sw.writelines(fea+'\n')
        duration = time.time() - start_time
        batch_number += 1

def get_image_paths(filename):
    image_paths_flat = []
    fw = open(filename, 'r')
    lines = fw.readlines()
    for i in range(len(lines)):
        image_path = lines[i].split()[0]
        image_paths_flat.append(image_paths)
    return image_paths_flat

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--pretrained_model', type=str,
        help='Load a pretrained model before training starts.')
    parser.add_argument('--model_def', type=str,
        help='Model definition. Points to a module containing the definition of the inference graph.', default='models.inception_resnet_v1')
    parser.add_argument('--batch_size', type=int,
        help='Number of images to process in a batch.', default=90)
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=160) 
    parser.add_argument('--data_file', type=str,
        help='File containing the image_paths.')
    parser.add_argument('--seed', type=int,
        help='Random seed.', default=666)
 

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))