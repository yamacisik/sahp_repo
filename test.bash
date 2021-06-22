#!/bin/bash

for lr in 0.01 0.001 0.0001
do
for es in 0.01 0.001
do
python python main_func.py  -e 1000 -t mimic --lr $lr  --early_stop_threshold $es --lambda_l2 0.0
done
done

python python main_func.py  -e 1000 -t synthetic

for lr in 0.001 0.0001
do
for es in 0.01 0.001
do
python python main_func.py  -e 1000 -t stackOverflow --lr $lr  --early_stop_threshold $es --lambda_l2 0.0
done
done


for lr in 0.001 0.0001
do
for es in 0.01 0.001
do
python python main_func.py  -e 1000 -t retweet --lr $lr  --early_stop_threshold $es --lambda_l2 0.0
done
done