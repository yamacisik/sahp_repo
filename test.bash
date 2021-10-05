#!/bin/bash

#python3  main_func.py  --save_model True -t synthetic -e 1000 -st $st
for st in 0 1 2 3
do
python  main_func.py  --save_model True -t synthetic -e 1000 -st $st
done