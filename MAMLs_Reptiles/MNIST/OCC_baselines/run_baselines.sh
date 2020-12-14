#!/bin/bash

cd FB

echo "exp 1 - FB test 1"
PYTHONHASHSEED=0 python -u fb_main.py -config_file=config_val9_test1_fb.json



cd ../FB_IF

echo "exp 2 - FB-IF test 1"
PYTHONHASHSEED=0 python -u fb_if_main.py -config_file=config_val9_test1_fb_if.json



cd ../FB_OCSVM

echo "exp 3 - FB-OCSVM test 1"
PYTHONHASHSEED=0 python -u fb_ocsvm_main.py -config_file=config_val9_test1_fb_ocsvm.json

cd ../IF

echo "exp 4 IF - K=2 test 1"
PYTHONHASHSEED=0 python -u if_main.py -config_file=config_val9_test1_K_2_cir_100_IF.json

echo "exp 5 IF - K=10 test 1"
PYTHONHASHSEED=0 python -u if_main.py -config_file=config_val9_test1_K_10_cir_100_IF.json



cd ../MTL_IF

echo "exp 7 - MTL-IF test 1"
PYTHONHASHSEED=0 python -u mtl_if_main.py -config_file=config_val9_test1_mtl_if.json



cd ../MTL_OCSVM

echo "exp 8 - MTL-OCSVM test 1"
PYTHONHASHSEED=0 python -u mtl_ocsvm_main.py -config_file=config_val9_test1_mtl_ocsvm.json



cd ../OCSVM


echo "exp 9 OCSVM - K=2 test 1"
PYTHONHASHSEED=0 python -u ocsvm_main.py -config_file=config_val9_test1_K_2_cir_100_OCSVM.json

echo "exp 10 OCSVM - K=10 test 1"
PYTHONHASHSEED=0 python -u ocsvm_main.py -config_file=config_val9_test1_K_10_cir_100_OCSVM.json


