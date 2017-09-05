#!/bin/bash
python extract.py --batch_size=256 --pretrained_model='/home/face/face-recognition/facenet/data/20170512-110547' --data_file='v.txt' --model_def=models.inception_resnet_v1