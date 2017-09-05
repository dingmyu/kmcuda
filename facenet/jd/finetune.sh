#!/bin/bash
python finetune_v1_centerloss.py --logs_base_dir='../logs/shangcheng' --models_base_dir='../logs/shangcheng' --pretrained_model='../data/20170512-110547/model-20170512-110547.ckpt-250000' --data_dir='../../datasets/shangcheng_final/' --batch_size=128 --learning_rate=0.001 --center_loss_factor=0.02