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

sys.path.extend(["../","./","../../"])

from io_func import smart_open, skip_frame, sparse_tuple_from
from feat_process.feature_transform import FeatureTransform
from io_func.matio import read_next_utt
from fst import *
from io_func.kaldi_io_egs import NnetChainExample,ProcessEgsFeat
from io_func.kaldi_io_parallel import KaldiDataReadParallel

from tf_chain_py_api import chainloss,chainxentloss
from fst import Fst2SparseMatrix

import tensorflow as tf


if __name__ == '__main__':
    #path = '/search/speech/hubo/git/tf-code-acoustics/chain_source_7300/8-cegs-scp/'
    path = './'
    conf_dict = { 'batch_size' :64,
            'skip_offset': 0,
            'shuffle': False,
            'queue_cache':2,
            'io_thread_num':2}
    feat_trans_file = '../../conf/final.feature_transform'
    feat_trans = FeatureTransform()
    feat_trans.LoadTransform(feat_trans_file)
    logging.basicConfig(filename = 'test.log')
    logging.getLogger().setLevel('INFO')
    io_read = KaldiDataReadParallel()
    #io_read.Initialize(conf_dict, scp_file=path+'cegs.1.scp',
    io_read.Initialize(conf_dict, scp_file=path+'scp',
            feature_transform = feat_trans, criterion = 'chain')

    batch_info = 10
    start = time.time()
    io_read.Reset(shuffle = False)
    batch_num = 0


    label_dim = 3766
    with tf.device(0):
        self_X = tf.placeholder(tf.float32, [None, conf_dict['batch_size'], label_dim],
                name='feature')
        self_Y = tf.placeholder(tf.float32, [conf_dict['batch_size'], None], name="labels")
        self_indexs = tf.placeholder(tf.int32, [conf_dict['batch_size'], None, 2], name="indexs")
        self_in_labels = tf.placeholder(tf.int32, [conf_dict['batch_size'], None], name="in_labels")
        self_weights = tf.placeholder(tf.float32, [conf_dict['batch_size'], None], name="weights")
        self_statesinfo = tf.placeholder(tf.int32, [conf_dict['batch_size'], None, 2], name="statesinfo")
        self_num_states = tf.placeholder(tf.int32, [conf_dict['batch_size']], name="num_states")
        self_length = tf.placeholder(tf.int32, [None], name="length")
        
        leaky_hmm_coefficient = 0.1
        l2_regularize = 0.00005
        xent_regularize = 0.0
        delete_laststatesuperfinal = True
        den_fst = '../../chain_source/den.fst'
        den_indexs, den_in_labels, den_weights, den_statesinfo, den_num_states, den_start_state, laststatesuperfinal = Fst2SparseMatrix(den_fst)
        den_indexs = np.reshape(den_indexs,[-1]).tolist()
        den_in_labels = np.reshape(den_in_labels , [-1]).tolist()
        den_weights = np.reshape(den_weights, [-1]).tolist()
        den_statesinfo = np.reshape(den_statesinfo, [-1]).tolist()
    
        chain_loss = chainloss(self_X, self_Y,
                self_indexs, self_in_labels, self_weights, self_statesinfo, self_num_states,
                label_dim,
                den_indexs, den_in_labels, den_weights, den_statesinfo, den_num_states,den_start_state, delete_laststatesuperfinal,
                l2_regularize, leaky_hmm_coefficient, xent_regularize, time_major=True)
    
        #outputs = np.random.rand(50* 64*3766).reshape(50, 64, 3766)
        
    
        sess =  tf.Session() 
        while True:
            start1 = time.time()
            feat_mat, label, length, lat_list = io_read.GetInput()
            end1 = time.time()
            if feat_mat is None:
                break
            logging.info('batch number: '+str(batch_num) + ' ' + str(numpy.shape(feat_mat)))
            logging.info("time:"+str(end1-start1))
            batch_num += 1
            outputs = np.random.rand(34* 64*3766).reshape(34, 64, 3766)
            feat = feat_mat.reshape(-1,355)
            outputs = np.hstack((feat,feat,feat,feat,feat,feat,feat,feat,feat,feat,feat)).reshape((46,64,-1))
            outputs = outputs[0:] = outputs[0:length,:,0:label_dim]
            
            deriv_weights = label
            indexs, in_labels, weights, statesinfo, num_states = lat_list
            feed_dict = {self_X : outputs, self_Y : label, self_length : [length],
                        self_indexs : lat_list[0], self_in_labels : lat_list[1], self_weights : lat_list[2],
                        self_statesinfo : lat_list[3], self_num_states : lat_list[4]}
            start3 = time.time()
            loss = sess.run(chain_loss, feed_dict)
            end3 = time.time()
            print("------------------chain_loss time:",end3-start3)
    
        print('******end*****')
        io_read.Join()
    
    
    
    
