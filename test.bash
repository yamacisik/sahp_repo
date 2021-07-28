#!/bin/bash


for st in 0 1 2 3
do
python  main_func.py  --save_model True -t synthetic -e 1000 -st $st
python  main_func.py  --save_model True -t synthetic -e 1000 -st $st --atten-heads 1 --nLayers 1
done