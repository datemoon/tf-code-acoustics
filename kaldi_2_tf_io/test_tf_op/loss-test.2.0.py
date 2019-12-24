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

@tf.function
def chainloss_fn(outputs, deriv_weights,
        indexs, in_labels, weights, statesinfo, num_states,
        label_dim,
        den_indexs, den_in_labels, den_weights, den_statesinfo, den_num_states,
        den_start_state, delete_laststatesuperfinal,
        l2_regularize, leaky_hmm_coefficient, xent_regularize,
        time_major = True):
    chain_loss = chainloss(outputs, deriv_weights,
            indexs, in_labels, weights, statesinfo, num_states,
            label_dim,
            den_indexs, den_in_labels, den_weights, den_statesinfo, den_num_states,
            den_start_state, delete_laststatesuperfinal,
            l2_regularize, leaky_hmm_coefficient, xent_regularize,
            time_major = True)
    return chain_loss

if __name__ == '__main__':
    path = '/search/speech/hubo/git/tf-code-acoustics/chain_source_7300/'
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
    io_read.Initialize(conf_dict, scp_file=path+'cegs.1.scp',
            feature_transform = feat_trans, criterion = 'chain')

    start = time.time()
    io_read.Reset(shuffle = False)
    batch_num = 0
    den_fst = '../../chain_source/den.fst'
    den_indexs, den_in_labels, den_weights, den_statesinfo, den_num_states, den_start_state, laststatesuperfinal = Fst2SparseMatrix(den_fst)
    leaky_hmm_coefficient = 0.1
    l2_regularize = 0.00005
    xent_regularize = 0.0
    delete_laststatesuperfinal = True
    label_dim = 3766
    den_indexs = np.reshape(den_indexs,[-1]).tolist()
    den_in_labels = np.reshape(den_in_labels , [-1]).tolist()
    den_weights = np.reshape(den_weights, [-1]).tolist()
    den_statesinfo = np.reshape(den_statesinfo, [-1]).tolist()

    while True:
        start1 = time.time()
        feat_mat, label, length, lat_list = io_read.GetInput()
        end1 = time.time()
        if feat_mat is None:
            break
        logging.info('batch number: '+str(batch_num) + ' ' + str(numpy.shape(feat_mat)))
        logging.info("time:"+str(end1-start1))
        batch_num += 1
        
        deriv_weights = label
        indexs, in_labels, weights, statesinfo, num_states = lat_list
        outputs = np.random.rand(50* 64*3766).reshape(50, 64, 3766)
        start3 = time.time()
        chain_loss = chainloss(outputs, deriv_weights,
                    indexs, in_labels, weights, statesinfo, num_states,
                    label_dim,
                    den_indexs, den_in_labels, den_weights, den_statesinfo, den_num_states,
                    den_start_state, delete_laststatesuperfinal,
                    l2_regularize, leaky_hmm_coefficient, xent_regularize,
                    time_major = True)
        end3 = time.time()
        print("chain_loss time:",end3-start3)
        chain_mean_loss = chain_loss[0]

    print("-------------end-----------------")




