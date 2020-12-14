#!/bin/bash


PYTHONHASHSEED=0 python train.py --train-shot 10 --train-query 20 --val-episode 50 --train-way 2 --test-way 5 --dataset CIFAR_FS --network ResNet --save-path './experiments/OC_MetaOptNet_K_10_CIFAR_FS' --gpu 0 --head 'OC-SVM' --val-shot-ocsvm 10 --val-query-ocsvm 100 --seed 1

