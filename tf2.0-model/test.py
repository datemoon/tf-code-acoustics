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

from io_func import smart_open, skip_frame, sparse_tuple_from
from feat_process.feature_transform import FeatureTransform
from io_func.matio import read_next_utt
from fst import *
from io_func.kaldi_io_egs import NnetChainExample,ProcessEgsFeat
from io_func.kaldi_io_parallel import KaldiDataReadParallel

from common_model import *
from nnet_conf import nnet_conf
from tf_chain_py_api import chainloss,chainxentloss
from fst import Fst2SparseMatrix

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
    path = '/search/speech/hubo/git/tf-code-acoustics/chain_source_7300/8-cegs-scp/'
    conf_dict = { 'batch_size' :64,
            'skip_offset': 0,
            'shuffle': False,
            'queue_cache':2,
            'io_thread_num':2}
    feat_trans_file = '../conf/final.feature_transform'
    feat_trans = FeatureTransform()
    feat_trans.LoadTransform(feat_trans_file)
    logging.basicConfig(filename = 'test.log')
    logging.getLogger().setLevel('INFO')
    io_read = KaldiDataReadParallel()
    #io_read.Initialize(conf_dict, scp_file=path+'cegs.1.scp',
    io_read.Initialize(conf_dict, scp_file=path+'cegs.all.scp_0',
            feature_transform = feat_trans, criterion = 'chain')

    batch_info = 10
    start = time.time()
    io_read.Reset(shuffle = False)
    batch_num = 0
    model = CommonModel(nnet_conf)
    
    # Instantiate an optimizer.
    optimizer = tf.keras.optimizers.SGD(learning_rate=1e-1)
    # Instantiate a loss function.
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    den_fst = '../chain_source/den.fst'
    den_indexs, den_in_labels, den_weights, den_statesinfo, den_num_states, den_start_state, laststatesuperfinal = Fst2SparseMatrix(den_fst)
    leaky_hmm_coefficient = 0.1
    l2_regularize = 0.00005
    xent_regularize = 0.0
    delete_laststatesuperfinal = True
    label_dim = 3766
    #den_indexs = tf.make_tensor_proto(den_indexs)
    #den_in_labels = tf.make_tensor_proto(den_in_labels)
    #den_weights = tf.make_tensor_proto(den_weights)
    #den_statesinfo = tf.make_tensor_proto(den_statesinfo)
    #den_indexs = tf.convert_to_tensor(den_indexs)
    #den_in_labels = tf.convert_to_tensor(den_in_labels)
    #den_weights = tf.convert_to_tensor(den_weights)
    #den_statesinfo = tf.convert_to_tensor(den_statesinfo)
    den_indexs = np.reshape(den_indexs,[-1]).tolist()
    den_in_labels = np.reshape(den_in_labels , [-1]).tolist()
    den_weights = np.reshape(den_weights, [-1]).tolist()
    den_statesinfo = np.reshape(den_statesinfo, [-1]).tolist()

    loss_value = 0.0

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
        with tf.GradientTape() as tape:
            start2 = time.time()
            outputs = model(feat_mat)
            end2 = time.time()
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
            #model.summary()
            # Compute the loss value for this minibatch.
            #loss_value = loss_fn(label, outputs)

        # Use the gradient tape to automatically retrieve
        # the gradients of the trainable variables with respect to the loss.
        start4 = time.time()
        grads = tape.gradient(chain_mean_loss, model.trainable_weights)
        end4 = time.time()

        grads, gradient_norms = tf.clip_by_global_norm(grads, 5.0, use_norm=None)
        
        # Run one step of gradient descent by updating
        # the value of the variables to minimize the loss.
        start5 = time.time()
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        end5 = time.time()
        
        print("io time:%f, nnet time:%f, loss time:%f, grad time:%f, apply_gradients time:%f" % (end1-start1,end2-start2,end3-start3,end4-start4,end5-start5))
        print(batch_num,chain_mean_loss)
        loss_value += chain_mean_loss
        # Log every 200 batches.
        if batch_num % batch_info == 0:
            print('Training loss (for one batch) at step %s: %s' % (batch_num, float(loss_value/batch_info)))
            #print('Seen so far: %s samples' % ((batch_num + 1) * 64))
            loss_value = 0.0

    #model.save('tmp.model.h5', save_format='tf')
    #new_model = keras.models.load_model('tmp.model.h5')
    print('******end*****')




