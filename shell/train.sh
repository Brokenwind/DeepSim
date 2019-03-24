#!/bin/bash

cd ../src
nohup python3 train_model.py > ../logs/train_model_gensim_new.log 2>&1 &