#!/bin/bash

for lr in 0.001 0.0001
do
python  main_func.py  -e 1000 -t retweet --lr $lr  --early-stop-threshold 0.001 --lambda-l2 0.0
done
