'''
Usage:
extract_feature_multi(gpus_num, image_size, batch_size, filename, model, feature_path)
gpus_num: Total number of gpus e.g. 2, type int
image_size: Image size (height, width) in pixels. e.g. 160, type int
batch_size: e.g. 1024, type int
filename: The file containing the full image path. type str
model: Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file
        e.g. '/home/face/face-master/data/20170512-110547', type str
feature_path: The file saving the features, type str
'''


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import multiprocessing
import os, time

import tensorflow as tf
import numpy as np
import argparse
import math
from scipy import misc

import os
import sys
sys.path.append('/home/face/facenet-master/src')
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

def extract_feature(gpu_num, image_size, batch_size, model, data_path, feature_path): 
    print("ok")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_num
    sw = open(feature_path, 'w+')
    images = load_data(data_path, False, False, image_size)
    with tf.Graph().as_default():
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
        # Load the model
            facenet.load_model(model)
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            features = tf.get_default_graph().get_tensor_by_name("InceptionResnetV1/Logits/AvgPool_1a_8x8/AvgPool:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        
            feature_size = features.get_shape()[3]
        # Run forward pass to extract features
            print('Runnning forward pass on images')
            image_num = len(images)
            batch_num = int(math.ceil(1.0*image_num / batch_size))
        # features_arr = np.zeros((image_num, feature_size))
            print ('batch num %d'%(batch_num))
            for i in range(batch_num):
                fetures_arr = np.zeros((batch_size, feature_size))
                start_index = i * batch_size
                end_index = min((i+1)*batch_size, image_num)
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

def split(filename, indice):
    output_file = []
    fw  = open(filename, 'r')
    lines = fw.readlines()
    line_num = len(lines)
    num_per_file = int(line_num / indice) + 1
    for i in range(indice):
        start_index = i * num_per_file
        end_index = min((i + 1) * num_per_file, line_num)
        print(start_index, end_index)
        name = 'temp_'+str(i)+'.txt'
        sw = open(name, 'w+')
        for j in range(start_index, end_index):
            sw.writelines(lines[j])
        sw.close()
        output_file.append(name)
    fw.close()
    return output_file

def merge(file_names, output_file):
    sw = open(output_file, 'w+')
    for name in file_names:
        if os.path.isfile(name):
            fw = open(name, 'r')
            lines = fw.readlines()
            for line in lines:
                sw.writelines(line)
            fw.close()
    sw.close()

def extract_feature_multi(gpus_num, image_size, batch_size, filename, model, feature_path):
    start_time = time.time()
    gpus_num = gpus_num

    file_names = split(filename, gpus_num)
    save_names = []
    p = multiprocessing.Pool(processes=gpus_num)
    for i in range(gpus_num):
        save_name = feature_path+'f_'+str(i)+'.txt'
        save_names.append(save_name)
        p.apply_async(extract_feature, args=(str(i), image_size, batch_size, 
              model, file_names[i], save_name))
    p.close()
    p.join()
    merge(save_names, feature_path+'feature_'+str(time.time())+'.txt')
    for file_name in file_names:
        os.remove(file_name)
    for file_name in save_names:
        os.remove(file_name)
    duration = time.time() - start_time
    print('Total time is %.3f seconds'%duration)

