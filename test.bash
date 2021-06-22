#!/bin/bash

for lr in 0.01 0.001 0.0001
do
for es in 0.01 0.001
do
python  main_func.py  -e 1000 -t mimic --lr $lr  --early-stop-threshold $es --lambda-l2 0.0
done
done

python  main_func.py  -e 1000 -t synthetic

for lr in 0.001 0.0001
do
for es in 0.01 0.001
do
python  main_func.py  -e 1000 -t stackOverflow --lr $lr  --early-stop-threshold $es --lambda-l2 0.0
done
done


for lr in 0.001 0.0001
do
for es in 0.01 0.001
do
python  main_func.py  -e 1000 -t retweet --lr $lr  --early-stop-threshold $es --lambda-l2 0.0
done
done