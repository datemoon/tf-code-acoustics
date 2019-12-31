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

class Train():
    def __init__(self, strategy):
        self.model = CommonModel(nnet_conf)
        self.strategy = strategy
        # Instantiate an optimizer.
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
        den_fst = '../chain_source/den.fst'
        den_indexs, den_in_labels, den_weights, den_statesinfo, den_num_states, den_start_state, laststatesuperfinal = Fst2SparseMatrix(den_fst)
        self.leaky_hmm_coefficient = 0.1
        self.l2_regularize = 0.00005
        self.xent_regularize = 0.0
        self.delete_laststatesuperfinal = True
        self.label_dim = 3766
        self.den_indexs = np.reshape(den_indexs,[-1]).tolist()
        self.den_in_labels = np.reshape(den_in_labels , [-1]).tolist()
        self.den_weights = np.reshape(den_weights, [-1]).tolist()
        self.den_statesinfo = np.reshape(den_statesinfo, [-1]).tolist()
        self.den_num_states = den_num_states
        self.den_start_state = den_start_state
        self.laststatesuperfinal = laststatesuperfinal

    def TrainStep(self, inputs):
        #feat_mat, label, length, lat_list = inputs
        start1 = time.time()
        feat_mat, label, length, indexs, in_labels, weights, statesinfo, num_states = inputs
        print("feat_mat:",feat_mat.shape)
        end1 = time.time()
        logging.info("time:"+str(end1-start1))
        
        deriv_weights = label
        #indexs, in_labels, weights, statesinfo, num_states = lat_list
        with tf.GradientTape() as tape:
            start2 = time.time()
            print("-----------model start------------------")
            outputs = self.model(feat_mat)
            print("----------end outputs:",outputs.shape)
            end2 = time.time()
            start3 = time.time()
            chain_loss = chainloss(outputs, deriv_weights,
                    indexs, in_labels, weights, statesinfo, num_states,
                    self.label_dim,
                    self.den_indexs, self.den_in_labels, self.den_weights, self.den_statesinfo, self.den_num_states,
                    self.den_start_state, self.delete_laststatesuperfinal,
                    self.l2_regularize, self.leaky_hmm_coefficient, self.xent_regularize,
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
        grads = tape.gradient(chain_mean_loss, self.model.trainable_weights)
        end4 = time.time()
    
        grads, gradient_norms = tf.clip_by_global_norm(grads, 5.0, use_norm=None)
        
        # Run one step of gradient descent by updating
        # the value of the variables to minimize the loss.
        start5 = time.time()
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        end5 = time.time()
        
        print("nnet time:%f, loss time:%f, grad time:%f, apply_gradients time:%f" % (end2-start2,end3-start3,end4-start4,end5-start5))
        return chain_mean_loss

    def RunStep(self, inputs):
        per_replica_loss = self.strategy.experimental_run_v2(
                train.TrainStep, args=(inputs,))
        return per_replica_loss

if __name__ == '__main__':
    #path = '/search/speech/hubo/git/tf-code-acoustics/chain_source_7300/8-cegs-scp/'
    path = '/search/speech/hubo/git/tf-code-acoustics/chain_source_7300/'
    conf_dict = { 'batch_size' :64,
            'skip_offset': 0,
            'skip_frame':3,
            'shuffle': False,
            'queue_cache':10,
            'io_thread_num':3}
    feat_trans_file = '../conf/final.feature_transform'
    feat_trans = FeatureTransform()
    feat_trans.LoadTransform(feat_trans_file)
    logging.basicConfig(filename = 'test.log')
    logging.getLogger().setLevel('INFO')
    io_read = KaldiDataReadParallel()
    io_read.Initialize(conf_dict, scp_file=path+'cegs.1.scp',
    #io_read.Initialize(conf_dict, scp_file=path+'cegs.all.scp_0',
            feature_transform = feat_trans, criterion = 'chain')

    batch_info = 200
    start = time.time()
    io_read.Reset(shuffle = False)
    batch_num = 0
    
    # Instantiate a loss function.
    #loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    loss_value = 0.0
    
    #ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=model)
    #manager = tf.train.CheckpointManager(ckpt, './model', max_to_keep=10)
    #ckpt.restore(manager.latest_checkpoint)

    #if manager.latest_checkpoint:
    #    print("Restored from {}".format(manager.latest_checkpoint))
    #else:
    #    print("Initializing from scratch.")

    def Gen():
        while True:
            inputs = io_read.GetInput()
            if inputs[0] is None:
                io_read.JoinInput()
                io_read.Reset(shuffle = True)
                print("-inputs once ok----end once io----")
                continue
            print("Gen input:",inputs[0].shape)
            indexs, in_labels, weights, statesinfo, num_states = inputs[3]
            yield(inputs[0],inputs[1],inputs[2],indexs, in_labels, weights, statesinfo, num_states)
            #yield(1,2,3,4,5,6,7,8)
    
    #dataset = tf.data.Dataset.from_generator(Gen, output_types=(tf.float32,tf.int32,tf.int32,tf.int32,tf.int32,tf.float32,tf.int32,tf.int32),output_shapes=None,args=None)
    
    num_gpu=1
    devices = ['/device:GPU:{}'.format(i) for i in range(num_gpu)]
    strategy = tf.distribute.MirroredStrategy(devices)
    input_context = tf.distribute.InputContext(num_input_pipelines=num_gpu,
            input_pipeline_id=0,
            num_replicas_in_sync=num_gpu)

    with strategy.scope():
        train = Train(strategy)
        def dataset_fn(input_context):
            dataset = tf.data.Dataset.from_generator(Gen, output_types=(tf.float32,tf.float32,tf.int32,tf.int32,tf.int32,tf.float32,tf.int32,tf.int32),output_shapes=None,args=None)
            return dataset.shard(
                    input_context.num_input_pipelines, input_context.input_pipeline_id)
        
        def distributed_train(model, one_batch):
            per_replica_loss = strategy.experimental_run_v2(
                    train.TrainStep, args=(one_batch))
            return per_replica_loss
        
        distributed_train = tf.function(distributed_train)



        train_dist_dataset = strategy.experimental_distribute_datasets_from_function(dataset_fn)

        print('Training...')
        total_loss = 0.0
        num_train_batches = 0.0
        batch_loss = 0.0

        for one_batch  in train_dist_dataset:
            print('*******************start************************')
            print('****one batch*******',one_batch[2],num_train_batches)
            #time.sleep(1)
            #per_replica_loss = strategy.experimental_run_v2(
            #        train.TrainStep, args=(one_batch))
            per_replica_loss = train.RunStep(one_batch)
            #per_replica_loss = distributed_train(model, one_batch)
            print('num_train_batches: %f per_replica_loss: %f'%(num_train_batches,per_replica_loss))
            batch_loss += per_replica_loss
            total_loss += strategy.reduce(
                    tf.distribute.ReduceOp.MEAN, per_replica_loss, axis=None)
                    #tf.distribute.ReduceOp.SUM, per_replica_loss, axis=None)

            num_train_batches += 1
            if num_train_batches % 20 == 0:
                print("num_train_batches:%d,train loss:%f, batch loss:%f" % (num_train_batches, total_loss/num_train_batches,batch_loss/20))
                batch_loss = 0.0
                train.model.summary()
                #time.sleep(5)



        #config = model.get_config()
        #weights = model.get_weights()
        #new_model.set_weights(weights)
        #model.save_weights('path_to_my_weights.h5')
        train.model.summary()
        #model.SaveModelWeights('model/path_to_my_weights.h5-1')
        #model.SaveModelWeights('model/path_to_my_weights.h5', save_format='h5')
        del train.model
        #new_model = CommonModel.ReStoreModel(nnet_conf, 'model/path_to_my_weights.h5-1')
        #new_model.summary()
        #new_model.SaveModelWeights('model/path_to_my_weights.h5-2')
        #del new_model
    
        #new_model = keras.models.load_model('tmp.model.h5')
        print('******end*****')


    io_read.JoinInput()


