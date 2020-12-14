#!/bin/bash


# to run the (original) class-balanced version of any algorithm set cir_inner_loop to 0.5 in the configuration file 
PYTHONHASHSEED=0 python main.py -config_file=config_FC100_oc_metasgd_K_10_BN_seed_1.json

