from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import math
from scipy import misc
import finance
from sklearn import metrics
import os
import sys
sys.path.append('../src')
import facenet

def main(args):  
    with tf.Graph().as_default():
        with tf.Session() as sess:
            # Read the image to extract features
            image_size = args.image_size

            # Get the images and labels
            print('Start loading data...')
            paths, actual_issame = finance.get_data(os.path.expanduser(args.images_path), args.filename, args.file_ext)
            print('Finish loading data...')
            # Load the model
            facenet.load_model(args.model)
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            
            embedding_size = embeddings.get_shape()[1]

            # Run forward pass to extract features
            print('Runnning forward pass on images')
            batch_size = args.batch_size
            images_num = len(paths)
            batches_num = int(math.ceil(1.0*images_num / batch_size))
            print(batches_num)
            emb_array = np.zeros((images_num, embedding_size))
            for i in range(batches_num):
                start_index = i*batch_size
                end_index = min((i+1)*batch_size, images_num)
                paths_batch = paths[start_index:end_index]
                inputs = finance.load_data(paths_batch, False, False, args.image_size)
                feed_dict = { images_placeholder:inputs, phase_train_placeholder:False }
                emb_array[start_index:end_index,:] = sess.run(embeddings, feed_dict=feed_dict)
                print('Have loaded %d'%i)

            tpr, fpr, accuracy = finance.evaluate(emb_array, actual_issame) 
            print(np.mean(accuracy))
            auc = metrics.auc(fpr, tpr)
            print('Area Under Curve (AUC): %1.3f' % auc)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--batch_size', type=int,
        default=1)
    parser.add_argument('--images_path', type=str,
        help='The directory where images are in')
    parser.add_argument('--filename', type=str,
        help='The file containing the image')
    parser.add_argument('--file_ext', type=str,
        help='The file extension for the dataset.', default='png', choices=['jpg', 'png'])
    parser.add_argument('--model', type=str,
        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))