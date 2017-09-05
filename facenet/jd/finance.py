from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from scipy import misc

import os
import sys
sys.path.append('../src')
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
  
def load_data(image_paths, do_random_crop, do_random_flip, image_size, do_prewhiten=True):
    nrof_samples = len(image_paths)
    images = np.zeros((nrof_samples, image_size, image_size, 3))
    for i in range(nrof_samples):
        img = misc.imread(image_paths[i])
        if img.ndim == 2:
            img = to_rgb(img)
        if do_prewhiten:
            img = prewhiten(img)
        img = crop(img, do_random_crop, image_size)
        img = flip(img, do_random_flip)
        images[i,:,:,:] = img
    return images

def evaluate(embeddings, actual_issame):
    # Calculate evaluation metrics
    best = 0.0
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    # thresholds = np.arange(0, 4, 0.01)
    # diff = np.subtract(embeddings1, embeddings2)
    # dist = np.sum(np.square(diff),1)

    # for threshold in thresholds:
    #     tpr, fpr, accuracy = facenet.calculate_accuracy(threshold, dist, np.asarray(actual_issame))
    #     if accuracy > best:
    #         best = accuracy
    # return best
    nrof_folds = 5
    thresholds = np.arange(0, 4, 0.001)
    tpr, fpr, accuracy = facenet.calculate_roc(thresholds, embeddings1, embeddings2,
        np.asarray(actual_issame), nrof_folds=nrof_folds)
    return tpr, fpr, accuracy

def get_data(dirname, filename, file_ext):
    skipped_pairs_num = 0
    path_list = []
    issame_list = []
    fw = open(filename)
    lines = fw.readlines()
    for i in range(len(lines)):
        line = lines[i]
        words = line.split()
        if len(words) == 3:
            path0 = os.path.join(dirname, words[0], words[1] + '.' + file_ext)
            path1 = os.path.join(dirname, words[0], words[1] + '.' + file_ext)
            issame = True
        elif len(words) == 4:
            path0 = os.path.join(dirname, words[0], words[1] + '.' + file_ext)
            path1 = os.path.join(dirname, words[2], words[3] + '.' + file_ext)
            issame = False
        else:
            break
        if os.path.exists(path0) and os.path.exists(path1):    # Only add the pair if both paths exist
            path_list += (path0,path1)
            issame_list.append(issame)
        else:
            skipped_pairs_num += 1
        if i % 100 == 0:
            print('Have loading %d'%i)
    if skipped_pairs_num > 0:
        print('Skipped %d image pairs' % skipped_pairs_num)
    # print (path_list, issame_list)
    return path_list, issame_list