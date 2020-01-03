import gzip
import os
import sys, re
import glob
import struct

import numpy
import random
import logging
import threading
import multiprocessing
import ctypes
import time

sys.path.extend(["../","./"])
import tensorflow as tf
from feat_process.feature_transform import FeatureTransform
from io_func.kaldi_io_parallel import KaldiDataReadParallel


class KaldiDataset(tf.data.Dataset):
    def _generator():
        while True:
            inputs = self.io_read.GetInput()
            indexs, in_labels, weights, statesinfo, num_states = inputs[3]
            if inputs[0] is not None:
                yield(inputs[0],inputs[1],inputs[2],indexs, in_labels, weights, statesinfo, num_states)

    def __init__(cls):
        return tf.data.Dataset.from_generator(
                cls._generator,
                output_types=(tf.float32,tf.int32,tf.int32,tf.int32,tf.int32,tf.float32,tf.int32,tf.int32),
                output_shapes=None,
                args=None
                )


if __name__ == '__main__':
    path = '/search/speech/hubo/git/tf-code-acoustics/chain_source_7300/8-cegs-scp/'
    path = '/search/speech/hubo/git/tf-code-acoustics/chain_source_7300/'
    conf_dict = { 'batch_size' :64,
            'skip_offset': 0,
            'skip_frame':3,
            'shuffle': False,
            'queue_cache':2,
            'io_thread_num':5}
    feat_trans_file = '../conf/final.feature_transform'
    feat_trans = FeatureTransform()
    feat_trans.LoadTransform(feat_trans_file)
    io_read = KaldiDataReadParallel()
    #io_read.Initialize(conf_dict, scp_file=path+'cegs.all.scp_0',
    io_read.Initialize(conf_dict, scp_file=path+'cegs.1.scp',
            feature_transform = feat_trans, criterion = 'chain')

    io_read.Reset(shuffle = False)

    def Gen():
        while True:
            inputs = io_read.GetInput()
            if inputs[0] is not None:
                indexs, in_labels, weights, statesinfo, num_states = inputs[3]
                yield(inputs[0],inputs[1],inputs[2],indexs, in_labels, weights, statesinfo, num_states)
            else:
                print("-----end io----")
                break


    #dataset = KaldiDataset(io_read)
    dataset = tf.data.Dataset.from_generator(Gen, output_types=(tf.float32,tf.float32,tf.int32,tf.int32,tf.int32,tf.float32,tf.int32,tf.int32),output_shapes=None,args=None)
    
    #start_time = time.perf_counter()
    start_time = time.time()
    batch_num = 0 
    for inputs in dataset:
        print('feature shape:',inputs[0].shape)
        batch_num += 1
    print("batch_num : %d Execution time: %f" % (batch_num, time.time() - start_time))
    io_read.JoinInput()
    print("*********end************")



