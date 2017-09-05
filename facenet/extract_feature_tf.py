from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import math
from scipy import misc

import os
import sys
sys.path.append('../src')
import facenet
import input_pipeline
tf.app.flags.DEFINE_string('mode', 'feature',
                           'train or feature')


def main(args):
    with tf.Graph().as_default():
        sw = open(args.feature_path, 'w+')
        image_size = args.image_size
        batch_size = args.batch_size
        # Read the image to extract features
        print ('Start loading data...')
        index, images = input_pipeline.inputs(args.filename, batch_size=batch_size, mode='feature')
        
        # Load the model
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            facenet.load_model(args.model)
            # batch_size_placeholder = tf.placeholder(tf.int32, batch_size)
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            features = tf.get_default_graph().get_tensor_by_name("InceptionResnetV1/Logits/AvgPool_1a_8x8/AvgPool:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            # checkpoint_file = args.model
            # net, endpoints = inception_resnet_v1.inference(images_placeholder, 0.8, phase_train=False)
            # features = endpoints['AvgPool_1a_8x8']
            feature_size = features.get_shape()[3]


            # saver = tf.train.Saver()
            # saver.restore(sess, checkpoint_file)
            # Run forward pass to extract features
            print('Runnning forward pass on images')
            
            image_num = args.image_num
            batch_num = int(math.ceil(1.0*image_num / batch_size))
            print(batch_num)
            for i in range(batch_num):
                start_index = i * batch_size
                end_index = min((i+1)*batch_size, image_num)
                batch_size = end_index-start_index
                IMG = sess.run(images)
                (feature) = sess.run(features[:,0,0,:], feed_dict={images_placeholder:IMG, phase_train_placeholder:False})
                # print(feature)
                for j in range(batch_size):
                    for k in range(1792):
                        sw.writelines(str(feature[j][k])+" ")
                    sw.writelines("\n")
                if i % 100 == 0:
                    print('Have loaded %d'%i)
            sw.close()
            coord.request_stop()
            coord.join(threads)
        sess.close()

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--batch_size', type=int,
        default=1)
    parser.add_argument('--image_num', type=int,
        help='Total number of images')
    parser.add_argument('--filename', type=str,
        help='The name of TFRecord')
    parser.add_argument('--model', type=str,
        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    parser.add_argument('--feature_path', type=str,
        help='The file to save features')
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))