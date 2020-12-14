#!/bin/bash

# to run the (original) class-balanced version of any algorithm set cir_inner_loop to 0.5 in the configuration file 
PYTHONHASHSEED=0 python -u main.py -config_file=config_STS_Sine_ocmaml_K_10_noBN_seed_1.json

