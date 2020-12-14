#!/bin/bash


# to run the (original) class-balanced version of any algorithm set cir_inner_loop to 0.5 in the configuration file 
PYTHONHASHSEED=0 python -u main.py -config_file=config_MIN_ocmaml_K_10_BN_seed_1.json

