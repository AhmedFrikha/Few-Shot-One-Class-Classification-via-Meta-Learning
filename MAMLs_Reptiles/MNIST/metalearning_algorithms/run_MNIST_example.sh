#!/bin/bash

# to run the (original) class-balanced version of any algorithm set cir_inner_loop to 0.5 in the configuration file 
# to run another task-combination modify test_task_idx in the configuration file to determine the task that should be used as a test task
PYTHONHASHSEED=0 python -u main.py -config_file=config_MNIST_ocmaml_K_10_BN_seed_1.json


