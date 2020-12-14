#!/bin/bash

cd FB

echo "exp 1 - FB"
PYTHONHASHSEED=0  python -u fb_main.py -config_file=config_fb.json



cd ../FB_OCSVM

echo "exp 3 - FB-OCSVM"
PYTHONHASHSEED=0 python -u fb_ocsvm_main.py -config_file=config_fb_ocsvm.json


cd ../FB_IF

echo "exp 2 - FB-IF"
PYTHONHASHSEED=0 python -u fb_if_main.py -config_file=config_fb_if.json



cd ../IF

echo "exp 4 IF - K=2"
PYTHONHASHSEED=0 python -u if_main.py -config_file=config_K_2_cir_100_IF.json

echo "exp 5 IF - K=10"
PYTHONHASHSEED=0  python -u if_main.py -config_file=config_K_10_cir_100_IF.json


cd ../OCSVM


echo "exp 9 OCSVM - K=2"
PYTHONHASHSEED=0 python -u ocsvm_main.py -config_file=config_K_2_cir_100_OCSVM.json

echo "exp 10 OCSVM - K=10"
PYTHONHASHSEED=0 python -u ocsvm_main.py -config_file=config_K_10_cir_100_OCSVM.json



cd ../MTL_OCSVM

echo "exp 8 - MTL-OCSVM"
PYTHONHASHSEED=0 python -u mtl_ocsvm_main.py -config_file=config_mtl_ocsvm.json

cd ../MTL_IF

echo "exp 7 - MTL-IF"
PYTHONHASHSEED=0 python -u mtl_if_main.py -config_file=config_mtl_if.json


