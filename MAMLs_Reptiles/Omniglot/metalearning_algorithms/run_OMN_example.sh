#!/bin/bash


# to run the (original) class-balanced version of any algorithm set cir_inner_loop to 0.5 in the configuration file 
PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES=3 python -u main.py -config_file=config_OMN_ocmaml_K_10_BN_seed_1.json

