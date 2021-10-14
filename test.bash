#!/bin/bash

#python3  main_func.py  --save_model True -t synthetic -e 1000 -st $st
#for st in 0
#do
#python  main_func.py  --save_model True -t synthetic -e 500 -st $st
#done
#python  main_func.py  --save_model True -t mimic -e 2 --lr 0.0001 -es  0.001

python  main_func.py  --save_model True -t stackOverflow -e 500 --lr 0.0001 -es  0.001 -b 32
