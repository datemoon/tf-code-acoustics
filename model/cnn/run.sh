#!/bin/bash

#~/git/kaldi/src/nnetbin/nnet-forward cnn_pooltest ark:input ark,t:-

CUDA_VISIBLE_DEVICES=false python3 cnn2_test.py
CUDA_VISIBLE_DEVICES=false python3 cnn_test.py
