from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import argparse
import math
from scipy import misc
import numpy as np
import os, sys
import PIL.Image

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

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

def load_image(image, do_random_crop, do_random_flip, image_size, do_prewhiten=True):
    try:
        img = PIL.Image.open(image)
        img = np.asarray(img)
        if img.ndim == 2:
            img = to_rgb(img)
        if do_prewhiten:
            img = prewhiten(img)
        img = crop(img, do_random_crop, image_size)
        img = flip(img, do_random_flip)
        returned_img = PIL.Image.fromarray(img, 'RGB')
        return returned_img
    except IOError:
        print(image)

def convert_to_tf(writer, image, index):
    image_raw = image.tobytes()
    example = tf.train.Example(features=tf.train.Features(feature={
        'index': _int64_feature([index]),
        'image_raw': _bytes_feature([image_raw])
        }))
    writer.write(example.SerializeToString())

def convert_TFRecords(filename, image_dir, save_name, image_size):
    print('Start writing %s'%save_name)
    writer = tf.python_io.TFRecordWriter(save_name)
    fw = open(filename, 'r')
    lines = fw.readlines()
    for i in range(len(lines)):
        img_name = image_dir+lines[i].split()[0]
        if os.path.exists(img_name):
            image = load_image(img_name, False, False, image_size)
            if image is None:
                continue
            else:
                convert_to_tf(writer,image,i)
        else:
            print (img_name)
        if i % 1000 == 0:
            print('Have loaded %d'%i)


def main(args):
    image_size = args.image_size
    save_name = args.save_name
    filename = args.filename
    image_dir = args.image_dir
    convert_TFRecords(filename=filename, image_dir=image_dir, save_name=save_name, image_size=image_size)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--save_name', type=str,
        help='Directory to write the converted result')
    parser.add_argument('--filename', type=str,
        help='File which stores images')
    parser.add_argument('--image_dir', type=str,
        help='Directory which stores images')
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
