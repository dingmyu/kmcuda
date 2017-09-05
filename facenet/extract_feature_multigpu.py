from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import math
from scipy import misc
import time

import os
import sys
sys.path.append('/home/face/face-recognition/facenet/src')
import facenet

def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1/std_adj)
    return y  

def crop(image, random_crop, image_size):
    if image.shape[1]>image_size:
        sz1 = int(image.shape[1]//2)
        sz2 = int(image_size//2)
        if random_crop:
            diff = sz1-sz2
            (h, v) = (np.random.randint(-diff, diff+1), np.random.randint(-diff, diff+1))
        else:
            (h, v) = (0,0)
        image = image[(sz1-sz2+v):(sz1+sz2+v),(sz1-sz2+h):(sz1+sz2+h),:]
    return image
  
def flip(image, random_flip):
    if random_flip and np.random.choice([True, False]):
        image = np.fliplr(image)
    return image

def to_rgb(img):
    w, h = img.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
    return ret

def load_data(filename, do_random_crop, do_random_flip, image_size, do_prewhiten=True):
    fw = open(filename, 'r')
    lines = fw.readlines()
    num = len(lines)
    images = np.zeros((num, image_size, image_size, 3))
    paths = []
    for i in range(num):
        image_path = lines[i].split()[0]
        img = misc.imread(image_path)
        if img.ndim == 2:
            img = to_rgb(img)
        if do_prewhiten:
            img = prewhiten(img)
        img = crop(img, do_random_crop, image_size)
        img = flip(img, do_random_flip)
        images[i,:,:,:] = img
        if i % 1000 == 0:
            print ('Have loaded %d'%i)
    return images

def main(args): 
    start_time = time.time()
    sw = open(args.feature_path, 'w+')
    with tf.device('/cpu'):
        image_size = args.image_size
        # Read the image to extract features
        print ('Start loading data...')
        images = load_data(args.filename, False, False, image_size)

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Graph().as_default(), tf.Session(config=config).as_default() as sess:
        
        # Load the model
        facenet.load_model(args.model)
        batch_size = args.batch_size
        image_num = len(images)
        gpu_num = args.gpus_num
        batch_num = int(math.ceil(1.0*image_num / (batch_size * gpu_num)))
        print ('batch num %d'%batch_num)
        print('Runnning forward pass on images')
        for i in range(batch_num):
            for d in range(gpu_num):
                with tf.device('/gpu:%d'%d):
                    images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                    features = tf.get_default_graph().get_tensor_by_name("InceptionResnetV1/Logits/AvgPool_1a_8x8/AvgPool:0")
                    phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
                
                    feature_size = features.get_shape()[3]

                     
                    fetures_arr = np.zeros((batch_size, feature_size))
                    start_index = (i * gpu_num + d)* batch_size
                    end_index = min((i * gpu_num + d + 1)*batch_size, image_num)
                    inputs = images[start_index:end_index]
                    feed_dict = { images_placeholder:inputs, phase_train_placeholder:False }
                    features_arr = sess.run(features[0:end_index-start_index,0,0,:], feed_dict=feed_dict)

                    for j in range(end_index-start_index):
                        for k in range(feature_size):
                            sw.writelines(str(features_arr[j][k])+" ")
                        sw.writelines("\n")
            if i % 1 == 0:
                print ('Have loaded %d batches'%(i)) 

    sw.close()
    duration = time.time() - start_time
    print('Total time is %.3f seconds'%duration)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus_num', type=int,
        help='Total number of gpus.', default=2)
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--batch_size', type=int,
        default=1)
    parser.add_argument('--filename', type=str,
        help='The file containing the image path')
    parser.add_argument('--model', type=str,
        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    parser.add_argument('--feature_path', type=str,
        help='The file to save features')
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))