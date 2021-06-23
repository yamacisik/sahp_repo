#!/bin/bash

python  main_func.py  --save_model True -t mimic --lambda-l2 0.0  --early-stop-threshold 0.0001 --lr 0.001
python  main_func.py  --save_model True -t stackOverflow --lambda-l2 0.0  --early-stop-threshold 0.01 --lr 0.0001
python  mainbash_func.py  --save_model True -t retweet --lambda-l2 0.0  --early-stop-threshold 0.001 --lr 0.001