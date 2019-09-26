#!/usr/bin/env python
from __future__ import absolute_import, division, print_function, unicode_literals
import os, sys, shutil, time
import random
import threading
try:
    import queue as Queue
except ImportError:
    import Queue

import numpy as np
import time
import logging

from io_func import sparse_tuple_from
from io_func.kaldi_io_parallel import KaldiDataReadParallel
from feat_process.feature_transform import FeatureTransform
from parse_args import parse_args
from model.lstm_model_new import LstmModel
from util.tensor_io import print_trainable_variables

import tensorflow as tf
from tensorflow.python import debug as tf_debug
from fst import Fst2SparseMatrix

strset=('criterion', 'feature_transfile', 'checkpoint_dir', 'optimizer')
class TrainClass(object):
    '''
    '''
    def __init__(self, conf_dict):
        # configure paramtere
        self.conf_dict = conf_dict
        self.print_trainable_variables_cf = False
        self.use_normal_cf = False
        self.use_sgd_cf = True
        self.optimizer_cf = 'GD'
        self.use_sync_cf = False
        self.use_clip_cf = False
        self.restore_training_cf = True
        self.checkpoint_dir_cf = None
        self.num_threads_cf = 1
        self.queue_cache_cf = 100
        self.task_index_cf = -1
        self.grad_clip_cf = 5.0
        self.feature_transfile_cf = None
        self.learning_rate_cf = 0.1
        self.learning_rate_decay_steps_cf = 100000
        self.learning_rate_decay_rate_cf = 0.96
        self.batch_size_cf = 16
        self.num_frames_batch_cf = 20
        self.reset_global_step_cf = True

        self.steps_per_checkpoint_cf = 1000

        self.criterion_cf = 'ctc'
        self.silence_phones = []
        # initial configuration parameter
        for attr in self.__dict__:
            if len(attr.split('_cf')) != 2:
                continue;
            key = attr.split('_cf')[0]
            if key in conf_dict.keys():
                if key in strset or type(conf_dict[key]) is not str:
                    self.__dict__[attr] = conf_dict[key]
                else:
                    print('***************',key)
                    self.__dict__[attr] = eval(conf_dict[key])

        if self.feature_transfile_cf == None:
            logging.info('No feature_transfile,it must have.')
            sys.exit(1)
        feat_trans = FeatureTransform()
        feat_trans.LoadTransform(self.feature_transfile_cf)

        # init train file
        self.kaldi_io_nstream_train = KaldiDataReadParallel()
        self.input_dim = self.kaldi_io_nstream_train.Initialize(self.conf_dict,
                scp_file = conf_dict['tr_scp'], label = conf_dict['tr_label'],
                lat_scp_file = conf_dict['lat_scp_file'],
                feature_transform = feat_trans, criterion = self.criterion_cf)
        # init cv file
        self.kaldi_io_nstream_cv = KaldiDataReadParallel()
        #self.kaldi_io_nstream_cv.Initialize(conf_dict,
        #        scp_file = conf_dict['cv_scp'], label = conf_dict['cv_label'],
        #        feature_transform = feat_trans, criterion = 'ctc')

        self.num_batch_total = 0
        self.tot_lab_err_rate = 0.0
        self.tot_num_batch = 0.0

        logging.info(self.kaldi_io_nstream_train.__repr__())
        #logging.info(self.kaldi_io_nstream_cv.__repr__())
        
        # Initial input queue.
        #self.input_queue = Queue.Queue(self.queue_cache_cf)

        self.acc_label_error_rate = []
        self.all_lab_err_rate = []
        self.num_save = 0
        for i in range(5):
            self.all_lab_err_rate.append(1.1)
        self.num_batch = []
        for i in range(self.num_threads_cf):
            self.acc_label_error_rate.append(1.0)
            self.num_batch.append(0)

        return
    
    # multi computers construct train graph
    def ConstructGraph(self, device, server):
        with tf.device(device):
            if 'cnn' in self.criterion_cf:
                self.X = tf.placeholder(tf.float32, [None, self.input_dim[0], self.input_dim[1], 1],
                        name='feature')
            else:
                self.X = tf.placeholder(tf.float32, [None, self.batch_size_cf, self.input_dim], 
                        name='feature')
            
            if 'ctc' in self.criterion_cf:
                self.Y = tf.sparse_placeholder(tf.int32, name="labels")
            elif 'whole' in self.criterion_cf:
                self.Y = tf.placeholder(tf.int32, [self.batch_size_cf, None], name="labels")
            elif 'ce' in self.criterion_cf:
                self.Y = tf.placeholder(tf.int32, [self.batch_size_cf, self.num_frames_batch_cf], name="labels")

            if 'mmi' in self.criterion_cf or 'smbr' in self.criterion_cf or 'mpfe' in self.criterion_cf:
                self.indexs = tf.placeholder(tf.int32, [self.batch_size_cf, None, 2], name="indexs")
                self.pdf_values = tf.placeholder(tf.int32, [self.batch_size_cf, None], name="pdf_values")
                self.lm_ws = tf.placeholder(tf.float32, [self.batch_size_cf, None], name="lm_ws")
                self.am_ws = tf.placeholder(tf.float32, [self.batch_size_cf, None], name="am_ws")
                self.statesinfo = tf.placeholder(tf.int32, [self.batch_size_cf, None, 2], name="statesinfo")
                self.num_states = tf.placeholder(tf.int32, [self.batch_size_cf], name="num_states")
                self.lattice = [self.indexs, self.pdf_values, self.lm_ws, self.am_ws, self.statesinfo, self.num_states]
            elif 'chain' in self.criterion_cf:
                self.indexs = tf.placeholder(tf.int32, [self.batch_size_cf, None, 2], name="indexs")
                self.in_labels = tf.placeholder(tf.int32, [self.batch_size_cf, None], name="in_labels")
                self.weights = tf.placeholder(tf.float32, [self.batch_size_cf, None], name="weights")
                self.statesinfo = tf.placeholder(tf.int32, [self.batch_size_cf, None, 2], name="statesinfo")
                self.num_states = tf.placeholder(tf.int32, [self.batch_size_cf], name="num_states")
                self.length = tf.placeholder(tf.int32, [None], name="length")
                self.fst = [self.indexs, self.in_labels, self.weights, self.statesinfo, self.num_states]

            self.seq_len = tf.placeholder(tf.int32,[None], name = 'seq_len')
            
            #self.learning_rate_var_tf = tf.Variable(float(self.learning_rate_cf), 
            #        trainable=False, name='learning_rate')
            # init global_step and learning rate decay criterion
            #self.global_step=tf.train.get_or_create_global_step()
            self.global_step=tf.Variable(0, trainable=False, name = 'global_step',dtype=tf.int64)
            exponential_decay = True
            if exponential_decay == True:
                self.learning_rate_var_tf = tf.train.exponential_decay(
                        float(self.learning_rate_cf), self.global_step,
                        self.learning_rate_decay_steps_cf,
                        self.learning_rate_decay_rate_cf,
                        staircase=True, 
                        name = 'learning_rate_exponential_decay')
            elif piecewise_constant == True:
                boundaries = [100000, 110000]
                values = [1.0, 0.5, 0.1]
                self.learning_rate_var_tf = tf.train.piecewise_constant(
                        self.global_step, boundaries, values)
            elif inverse_time_decay == True:
                # decayed_learning_rate = learning_rate / (1 + decay_rate * floor(global_step / decay_step))
                # decay_rate = 0.5 , decay_step = 100000
                self.learning_rate_var_tf =  tf.train.inverse_time_decay(
                        float(self.learning_rate_cf), self.global_step,
                        self.learning_rate_decay_steps_cf,
                        self.learning_rate_decay_rate_cf,
                        staircase=True,
                        name = 'learning_rate_inverse_time_decay')


            if self.optimizer_cf == 'GD':
                optimizer = tf.train.GradientDescentOptimizer(self.learning_rate_var_tf)
            elif self.optimizer_cf == 'Adam':
                optimizer = tf.train.AdamOptimizer(learning_rate=
                        self.learning_rate_var_tf, beta1=0.9, beta2=0.999, epsilon=1e-08)
            elif self.optimizer_cf == 'Adadelta':
                tf.train.AdadeltaOptimizer(learning_rate=self.learning_rate_var_tf,
                        rho=0.95,
                        epsilon=1e-08,
                        use_locking=False,
                        name='Adadelta')
            elif self.optimizer_cf == 'AdagradDA':
                tf.train.AdagradDAOptimizer(learning_rate=self.learning_rate_var_tf,
                        global_step = self.global_step,
                        initial_gradient_squared_accumulator_value=0.1,
                        l1_regularization_strength=0.0,
                        l2_regularization_strength=0.0,
                        use_locking=False,
                        name='AdagradDA')
            elif self.optimizer_cf == 'Adagrad':
                tf.train.AdagradOptimizer(learning_rate=self.learning_rate_var_tf,
                        initial_accumulator_value=0.1,
                        use_locking=False,
                        name='Adagrad')
            else:
                logging.error("no this opyimizer.")
                sys.exit(1)

            # sync train
            if self.use_sync_cf:
                optimizer = tf.train.SyncReplicasOptimizer(optimizer, replicas_to_aggregate=8,
                        total_num_replicas=8,
                        use_locking=True)
                sync_replicas_hook = [optimizer.make_session_run_hook(
                        is_chief = (self.task_index_cf==0))]
                logging.info("******use synchronization train******")
            else:
                sync_replicas_hook = None
                logging.info("******use asynchronization train******")
            nnet_model = LstmModel(self.conf_dict)

            mean_loss = None
            loss = None
            rnn_state_zero_op = None
            rnn_keep_state_op = None
            if 'ctc' in self.criterion_cf:
                ctc_mean_loss, ctc_loss , label_error_rate, _ = nnet_model.CtcLoss(self.X, self.Y, self.seq_len)
                mean_loss = ctc_mean_loss
                loss = ctc_loss
#            elif 'ce' in self.criterion_cf and 'cnn' in self.criterion_cf and 'whole' in self.criterion_cf:
#                ce_mean_loss, ce_loss , label_error_rate, rnn_keep_state_op, rnn_state_zero_op = nnet_model.CeCnnBlstmLoss(self.X, self.Y, self.seq_len)
#                mean_loss = ce_mean_loss
#                loss = ce_loss
            elif 'ce' in self.criterion_cf:
                ce_mean_loss, ce_loss , label_error_rate, rnn_keep_state_op, rnn_state_zero_op = nnet_model.CeLoss(self.X, self.Y, self.seq_len)
                mean_loss = ce_mean_loss
                loss = ce_loss
            elif 'mmi' in self.criterion_cf:
                mmi_mean_loss, mmi_loss, label_error_rate, rnn_keep_state_op, rnn_state_zero_op = nnet_model.MmiLoss(
                        self.X, self.Y, self.seq_len,
                        self.indexs, self.pdf_values, self.lm_ws, self.am_ws, self.statesinfo, self.num_states,
                        old_acoustic_scale = 0.0, acoustic_scale = 0.083, time_major = True, drop_frames= True)
                mean_loss = mmi_mean_loss
                loss = mmi_loss
            elif 'smbr' in self.criterion_cf or 'mpfe' in self.criterion_cf:
                if 'smbr' in self.criterion_cf:
                    criterion = 'smbr'
                else:
                    criterion = 'mpfe'
                pdf_to_phone = self.kaldi_io_nstream_train.pdf_to_phone
                log_priors = self.kaldi_io_nstream_train.pdf_prior

                mpe_mean_loss, mpe_loss, label_error_rate, rnn_keep_state_op, rnn_state_zero_op = nnet_model.MpeLoss(
                        self.X, self.Y, self.seq_len,
                        self.indexs, self.pdf_values, self.lm_ws, self.am_ws, self.statesinfo, self.num_states,
                        log_priors = log_priors,
                        silence_phones=self.silence_phones,
                        pdf_to_phone=pdf_to_phone,
                        one_silence_class = True,
                        criterion = criterion,
                        old_acoustic_scale = 0.0, acoustic_scale = 0.083,
                        time_major = True)
                mean_loss = mpe_mean_loss
                loss = mpe_loss
            elif 'chain' in self.criterion_cf:
                den_indexs, den_in_labels, den_weights, den_statesinfo, den_num_states, den_start_state, laststatesuperfinal = Fst2SparseMatrix(self.conf_dict['den_fst'])
                label_dim = self.conf_dict['label_dim']
                delete_laststatesuperfinal = True
                l2_regularize = 0.00005
                leaky_hmm_coefficient = 0.1
                #xent_regularize = 0.025
                xent_regularize = 0.0
                #den_indexs = tf.convert_to_tensor(den_indexs, name='den_indexs')
                #den_in_labels = tf.convert_to_tensor(den_in_labels, name='den_in_labels')
                #den_weights = tf.convert_to_tensor(den_weights, name='den_weights')
                #den_statesinfo = tf.convert_to_tensor(den_statesinfo, name='den_statesinfo')
                den_indexs = tf.make_tensor_proto(den_indexs)
                den_in_labels = tf.make_tensor_proto(den_in_labels)
                den_weights = tf.make_tensor_proto(den_weights)
                den_statesinfo = tf.make_tensor_proto(den_statesinfo)
                chain_mean_loss, chain_loss, label_error_rate, rnn_keep_state_op, rnn_state_zero_op = nnet_model.ChainLoss(
                        self.X, 
                        self.indexs, self.in_labels, self.weights, self.statesinfo, self.num_states, self.length,
                        label_dim,
                        den_indexs, den_in_labels, den_weights, den_statesinfo, den_num_states, 
                        den_start_state, delete_laststatesuperfinal,
                        l2_regularize, leaky_hmm_coefficient, xent_regularize)
                mean_loss = chain_mean_loss
                loss = chain_loss
            else:
                logging.info("no criterion.")
                sys.exit(1)

            if self.use_sgd_cf :
                tvars = tf.trainable_variables()
                if self.use_normal_cf :
                    l2_regu = tf.contrib.layers.l2_regularizer(0.5)
                    lstm_vars = [ var for var in tvars if 'lstm' in var.name ]
                    apply_l2_regu = tf.contrib.layers.apply_regularization(l2_regu, lstm_vars)

                #mean_loss = mean_loss + apply_l2_regu
                #grads_var = optimizer.compute_gradients(mean_loss+apply_l2_regu,
                #        var_list = tvars)
                #grads = [ g for g,_ in grads_var ]
                if self.use_clip_cf:
                    grads, gradient_norms = tf.clip_by_global_norm(tf.gradients(
                        mean_loss, tvars), self.grad_clip_cf,
                        use_norm=None)
                #grads, gradient_norms = tf.clip_by_global_norm(grads, 
                #        self.grad_clip_cf,
                #        use_norm=None)

                    train_op = optimizer.apply_gradients(
                            zip(grads, tvars),
                            global_step=self.global_step)
                else:
                    train_op = optimizer.minimize(mean_loss + apply_l2_regu,
                            global_step=self.global_step)


            else:
                train_op = optimizer.minimize(mean_loss,
                        global_step=self.global_step)

            # set run operation
            self.run_ops = {'train_op':train_op,
                    'mean_loss':mean_loss,
                    'loss':loss,
                    'label_error_rate':label_error_rate,
                    'rnn_keep_state_op':rnn_keep_state_op,
                    'rnn_state_zero_op':rnn_state_zero_op}

            # set initial parameter
            self.init_para = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())

            #tmp_variables = tf.trainable_variables()
            #self.saver = tf.train.Saver(tmp_variables, max_to_keep=100)

            self.total_variables = np.sum([np.prod(v.get_shape().as_list()) 
                for v in tf.trainable_variables()])
            logging.info('total parameters : %d' % self.total_variables)
            
            # set gpu option
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)

            # session config
            sess_config = tf.ConfigProto(intra_op_parallelism_threads=self.num_threads_cf,
                    inter_op_parallelism_threads=self.num_threads_cf,
                    allow_soft_placement=True,
                    log_device_placement=False,gpu_options=gpu_options)
            if self.reset_global_step_cf and self.task_index_cf == 0:
                non_train_variables = [self.global_step ] + tf.local_variables()
            else:
                non_train_variables = tf.local_variables()

            ready_for_local_init_op = tf.variables_initializer(non_train_variables)
            local_init_op = tf.report_uninitialized_variables(var_list=non_train_variables)

            # add saver hook
            self.saver = tf.train.Saver(max_to_keep=50, sharded=True, allow_empty=True)
            scaffold = tf.train.Scaffold(init_op = None,
                    init_feed_dict = None,
                    init_fn = None,
                    ready_op = None,
                    ready_for_local_init_op = ready_for_local_init_op,
                    local_init_op = local_init_op,
                    summary_op = None,
                    saver = self.saver,
                    copy_from_scaffold = None)

            self.sess = tf.train.MonitoredTrainingSession(
                    master = server.target,
                    is_chief = (self.task_index_cf==0),
                    checkpoint_dir = self.checkpoint_dir_cf,
                    scaffold= scaffold,
                    hooks=sync_replicas_hook,
                    chief_only_hooks=None,
                    save_checkpoint_secs=None,
                    save_summaries_steps=self.steps_per_checkpoint_cf,
                    save_summaries_secs=None,
                    config=sess_config,
                    stop_grace_period_secs=120,
                    log_step_count_steps=100,
                    max_wait_secs=7200,
                    save_checkpoint_steps=self.steps_per_checkpoint_cf)
          #          summary_dir = self.checkpoint_dir_cf + "_summary_dir")
            #self.sess = tf_debug.LocalCLIDebugWrapperSession(self.sess)

            '''
            sv = tf.train.Supervisor(is_chief=(self.task_index_cf==0),
                    global_step=global_step,
                    init_op = self.init_para,
                    logdir = self.checkpoint_dir_cf,
                    saver=self.saver,
                    save_model_secs=600,
                    checkpoint_basename='model.ckpt')

            self.sess = sv.prepare_or_wait_for_session(server.target, config=sess_config)
            '''
        return 

    def SaveTextModel(self):
        if self.print_trainable_variables_cf == True:
            ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir_cf)
            if ckpt and ckpt.model_checkpoint_path:
                print_trainable_variables(self.sess, ckpt.model_checkpoint_path+'.txt')

#    def InputFeat(self, input_lock):
#        while True:
#            input_lock.acquire()
#            '''
#            if 'ctc' in self.criterion_cf or 'whole' in self.criterion_cf:
#                if 'cnn' in self.criterion_cf:
#                    feat,label,length = self.kaldi_io_nstream.CnnLoadNextNstreams()
#                else:
#                    feat,label,length = self.kaldi_io_nstream.WholeLoadNextNstreams()
#                if length is None:
#                    break
#                print(np.shape(feat),np.shape(label), np.shape(length))
#                if len(label) != self.batch_size_cf:
#                    break
#                if 'ctc' in self.criterion_cf:
#                    sparse_label = sparse_tuple_from(label)
#                    self.input_queue.put((feat,sparse_label,length))
#                else:
#                    self.input_queue.put((feat,label,length))
#
#            elif 'ce' in self.criterion_cf:
#                if 'cnn' in self.criterion_cf:
#                    feat_array, label_array, length_array = self.kaldi_io_nstream.CnnSliceLoadNextNstreams()
#                else:
#                    feat_array, label_array, length_array = self.kaldi_io_nstream.SliceLoadNextNstreams()
#                if length_array is None:
#                    break
#                print(np.shape(feat_array),np.shape(label_array), np.shape(length_array))
#                if len(label_array[0]) != self.batch_size_cf:
#                    break
#                self.input_queue.put((feat_array, label_array, length_array))
#            '''
#            feat,label,length = self.kaldi_io_nstream.LoadBatch()
#            if length is None:
#                break
#            print(np.shape(feat),np.shape(label), np.shape(length))
#            sys.stdout.flush()
#            if 'ctc' in self.criterion_cf:
#                sparse_label = sparse_tuple_from(label)
#                self.input_queue.put((feat,sparse_label,length))
#            else:
#                self.input_queue.put((feat,label,length))
#            self.num_batch_total += 1
##            if self.num_batch_total % 3000 == 0:
##                self.SaveModel()
##                self.AdjustLearnRate()
#            print('total_batch_num**********',self.num_batch_total,'***********')
#            input_lock.release()
#        self.input_queue.put((None, None, None))
#
#    def ThreadInputFeatAndLab(self):
#        input_thread = []
#        input_lock = threading.Lock()
#        for i in range(1):
#            input_thread.append(threading.Thread(group=None, target=self.InputFeat,
#                args=(input_lock,),name='read_thread'+str(i)))
#
#        for thr in input_thread:
#            logging.info('ThreadInputFeatAndLab start')
#            thr.start()
#
#        return input_thread

#    def SaveModel(self):
#        while True:
#            time.sleep(1.0)
#            if self.input_queue.empty():
#                checkpoint_path = os.path.join(self.checkpoint_dir_cf, 
#                        str(self.num_batch_total)+'_model'+'.ckpt')
#                logging.info('save model: '+checkpoint_path+
#                        ' --- learn_rate: ' +
#                        str(self.sess.run(self.learning_rate_var_tf)))
#                self.saver.save(self.sess, checkpoint_path)
#                break

    # if current label error rate less then previous five
    def AdjustLearnRate(self):
        curr_lab_err_rate = self.GetAverageLabelErrorRate()
        logging.info("current label error rate : %f" % curr_lab_err_rate)
        all_lab_err_rate_len = len(self.all_lab_err_rate)
        for i in range(all_lab_err_rate_len):
            if curr_lab_err_rate < self.all_lab_err_rate[i]:
                break
            if i == len(self.all_lab_err_rate)-1:
                self.DecayLearningRate(0.8)
                logging.info('learn_rate decay to '+str(self.sess.run(self.learning_rate_var_tf)))
        self.all_lab_err_rate[self.num_save%all_lab_err_rate_len] = curr_lab_err_rate
        self.num_save += 1

    def DecayLearningRate(self, lr_decay_factor):
        learning_rate_decay_op = self.learning_rate_var_tf.assign(tf.multiply(self.learning_rate_var_tf, lr_decay_factor))
        self.sess.run(learning_rate_decay_op)
        logging.info('learn_rate decay to '+str(self.sess.run(self.learning_rate_var_tf)))
        logging.info('lr_decay_factor is '+str(lr_decay_factor))

    # get restore model number
    def GetNum(self,str):
        return int(str.split('/')[-1].split('_')[0])

    # train_loss is a open train or cv .
    def TrainLogic(self, device, shuffle = False, train_loss = True, skip_offset = 0):
        if train_loss == True:
            logging.info('TrainLogic train start.')
            logging.info('Start global step is %d---learn_rate is %f' % (self.sess.run(tf.train.get_or_create_global_step()), self.sess.run(self.learning_rate_var_tf)))
            self.kaldi_io_nstream = self.kaldi_io_nstream_train
            # set run operation
            if 'ctc' in self.criterion_cf or 'whole' in self.criterion_cf:
                run_op = {'train_op':self.run_ops['train_op'],
                        'label_error_rate': self.run_ops['label_error_rate'],
                        'mean_loss':self.run_ops['mean_loss'],
                        'loss':self.run_ops['loss']}
            elif 'ce' in self.criterion_cf:
                run_op = {'train_op':self.run_ops['train_op'],
                        'label_error_rate': self.run_ops['label_error_rate'],
                        'mean_loss':self.run_ops['mean_loss'],
                        'loss':self.run_ops['loss'],
                        'rnn_keep_state_op':self.run_ops['rnn_keep_state_op'],
                        'rnn_state_zero_op':self.run_ops['rnn_state_zero_op']}
            elif 'chain' in self.criterion_cf:
                run_op = {'train_op':self.run_ops['train_op'],
                        'mean_loss':self.run_ops['mean_loss'],
                        'loss':self.run_ops['loss']}
            else:
                assert 'No train criterion.'
        else:
            logging.info('TrainLogic cv start.')
            self.kaldi_io_nstream = self.kaldi_io_nstream_cv
            run_op = {'label_error_rate':self.run_ops['label_error_rate'],
                    'mean_loss':self.run_ops['mean_loss']}
        # reset io and start input thread
        self.kaldi_io_nstream.Reset(shuffle = shuffle, skip_offset = skip_offset)
        #threadinput = self.ThreadInputFeatAndLab()
        time.sleep(3)
        with tf.device(device):
            if 'ctc' in self.criterion_cf or 'whole' in self.criterion_cf or 'chain' in self.criterion_cf:
                self.WholeTrainFunction(0, run_op, 'train_ctc_thread_hubo')
            elif 'ce' in self.criterion_cf:
                self.SliceTrainFunction(0, run_op, 'train_ce_thread_hubo')
            else:
                logging.info('no criterion in train')
                sys.exit(1)
        
        tmp_label_error_rate = self.GetAverageLabelErrorRate()
        logging.info("current averagelabel error rate : %f" % tmp_label_error_rate)
        logging.info('learn_rate is '+str(self.sess.run(self.learning_rate_var_tf)))
        
        if train_loss == True:
            self.AdjustLearnRate()
            logging.info('TrainLogic train end.')
        else:
            logging.info('TrainLogic cv end.')

        # End input thread
        self.kaldi_io_nstream.JoinInput()
        #for  i in range(len(threadinput)):
        #    threadinput[i].join()

        self.ResetAccuracy()
        return tmp_label_error_rate

    def WholeTrainFunction(self, gpu_id, run_op, thread_name):
        logging.info('******start WholeTrainFunction******')
        total_curr_error_rate = 0.0
        num_batch = 0
        self.acc_label_error_rate[gpu_id] = 0.0
        self.num_batch[gpu_id] = 0
        total_curr_mean_loss = 0.0
        total_mean_loss = 0.0

        #print_trainable_variables(self.sess, 'save.model.txt')
        while True:
            time1=time.time()
            feat, label, length, lat_list = self.GetFeatAndLabel()
            if feat is None:
                logging.info('train thread end : %s' % thread_name)
                break
            time2=time.time()

            if 'mmi' in self.criterion_cf or 'smbr' in self.criterion_cf or 'mpfe' in self.criterion_cf:
                feed_dict = {self.X : feat, self.Y : label, self.seq_len : length,
                        self.indexs : lat_list[0], self.pdf_values : lat_list[1], self.lm_ws : lat_list[2],
                        self.am_ws : lat_list[3], self.statesinfo : lat_list[4], self.num_states : lat_list[5]}
            elif 'chain' in self.criterion_cf:
                # length is valid_length is int
                feed_dict = {self.X : feat, self.length : [length], 
                        self.indexs : lat_list[0], self.in_labels : lat_list[1], self.weights : lat_list[2],
                        self.statesinfo : lat_list[3], self.num_states : lat_list[4]}
            else:
                feed_dict = {self.X : feat, self.Y : label, self.seq_len : length}

            time3 = time.time()
            calculate_return = self.sess.run(run_op, feed_dict = feed_dict)
            time4 = time.time()


            print("thread_name: ", thread_name,  num_batch, " time:",time2-time1,time3-time2,time4-time3,time4-time1)
            if 'label_error_rate' in calculate_return.keys():
                print('label_error_rate:',calculate_return['label_error_rate'])
                total_curr_error_rate += calculate_return['label_error_rate']
                self.acc_label_error_rate[gpu_id] += calculate_return['label_error_rate']
            else:
                total_curr_error_rate += 0.0
                self.acc_label_error_rate[gpu_id] += 0.0
            print('mean_loss:',calculate_return['mean_loss'])

            total_curr_mean_loss += calculate_return['mean_loss']
            total_mean_loss += calculate_return['mean_loss']

            num_batch += 1

            self.num_batch[gpu_id] += 1
            if self.num_batch[gpu_id] % int(self.steps_per_checkpoint_cf/50) == 0:
                logging.info("Batch: %d current averagelabel error rate : %f, mean loss : %f" % (int(self.steps_per_checkpoint_cf/50), total_curr_error_rate / int(self.steps_per_checkpoint_cf/50),total_curr_mean_loss / int(self.steps_per_checkpoint_cf/50)))
                total_curr_error_rate = 0.0
                total_curr_mean_loss = 0.0
                logging.info("Batch: %d current total averagelabel error rate : %f,  mean loss : %f" % (self.num_batch[gpu_id], self.acc_label_error_rate[gpu_id] / self.num_batch[gpu_id], total_mean_loss/ self.num_batch[gpu_id]))
        logging.info('******end TrainFunction******')

    def SliceTrainFunction(self, gpu_id, run_op, thread_name):
        logging.info('******start SliceTrainFunction******')
        total_curr_error_rate = 0.0
        num_batch = 0
        self.acc_label_error_rate[gpu_id] = 0.0
        self.num_batch[gpu_id] = 0
        total_curr_mean_loss = 0.0
        total_mean_loss = 0.0

        num_sentence = 0
        while True:
            time1=time.time()
            feat, label, length, lat_list = self.GetFeatAndLabel()
            if feat is None:
                logging.info('train thread end : %s' % thread_name)
                break
            time2=time.time()
            self.sess.run(run_op['rnn_state_zero_op'])
            for i in range(len(feat)):
                print('************input info**********:',np.shape(feat[i]),np.shape(label[i]),length[i], flush=True)
                time3 = time.time()
                feed_dict = {self.X : feat[i], self.Y : label[i], self.seq_len : length[i]}
                time4 = time.time()
                run_need_op = {'train_op':run_op['train_op'],
                        'mean_loss':run_op['mean_loss'],
                        'loss':run_op['loss'],
                        'rnn_keep_state_op':run_op['rnn_keep_state_op'],
                        'label_error_rate':run_op['label_error_rate']}
                calculate_return = self.sess.run(run_need_op, feed_dict = feed_dict)
                time5 = time.time()
                print("thread_name: ", thread_name,  num_batch, " time:",time4-time3,time5-time4)
                print('label_error_rate:',calculate_return['label_error_rate'])
                print('mean_loss:',calculate_return['mean_loss'])

                total_curr_mean_loss += calculate_return['mean_loss']
                total_mean_loss += calculate_return['mean_loss']

                num_batch += 1
                total_curr_error_rate += calculate_return['label_error_rate']
                self.acc_label_error_rate[gpu_id] += calculate_return['label_error_rate']
                self.num_batch[gpu_id] += 1
                if self.num_batch[gpu_id] % int(self.steps_per_checkpoint_cf/50) == 0:
                    logging.info("Batch: %d current averagelabel error rate : %f, mean loss : %f" % (int(self.steps_per_checkpoint_cf/50), total_curr_error_rate / int(self.steps_per_checkpoint_cf/50),total_curr_mean_loss / int(self.steps_per_checkpoint_cf/50) ))
                    total_curr_error_rate = 0.0
                    total_curr_mean_loss = 0.0
                    logging.info("Batch: %d current total averagelabel error rate : %f, mean loss : %f" % (self.num_batch[gpu_id], self.acc_label_error_rate[gpu_id] / self.num_batch[gpu_id], total_mean_loss/ self.num_batch[gpu_id]))
            # print batch sentence time info
            print("******thread_name: ", thread_name,  num_sentence, "io time:",time2-time1, "calculation time:",time5-time1)
            num_sentence += 1

        logging.info('******end SliceTrainFunction******')

    def GetFeatAndLabel(self):
        return self.kaldi_io_nstream.GetInput()

    def GetAverageLabelErrorRate(self):
        tot_label_error_rate = 0.0
        tot_num_batch = 0
        for i in range(self.num_threads_cf):
            tot_label_error_rate += self.acc_label_error_rate[i]
            tot_num_batch += self.num_batch[i]
        if tot_num_batch == 0:
            average_label_error_rate = 1.0
        else:
            average_label_error_rate = tot_label_error_rate / tot_num_batch
        self.tot_lab_err_rate += tot_label_error_rate
        self.tot_num_batch += tot_num_batch
        #self.ResetAccuracy(tot_reset = False)
        return average_label_error_rate

    def GetTotLabErrRate(self):
        return self.tot_lab_err_rate/self.tot_num_batch

    def ResetAccuracy(self, tot_reset = True):
        for i in range(len(self.acc_label_error_rate)):
            self.acc_label_error_rate[i] = 0.0
            self.num_batch[i] = 0
        
        if tot_reset:
            self.tot_lab_err_rate = 0
            self.tot_num_batch = 0
            for i in range(5):
                self.all_lab_err_rate.append(1.1)
            self.num_save = 0




if __name__ == "__main__":
    # First read parameters
    # Read config
    conf_dict = parse_args(sys.argv[1:])
    
    ps_hosts = conf_dict['ps_hosts'].split(',')
    worker_hosts = conf_dict['worker_hosts'].split(',')
    job_name = conf_dict['job_name']
    task_index = conf_dict['task_index']

    # Create checkpoint dir if needed
    if not os.path.exists(conf_dict["checkpoint_dir"]):
        os.makedirs(conf_dict["checkpoint_dir"])

    # Set logging framework
    if conf_dict["log_file"] is not None:
        logging.basicConfig(filename = conf_dict["log_file"])
        logging.getLogger().setLevel(conf_dict["log_level"])
    else:
        raise 'no log file in config file'

    logging.info(conf_dict)

    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
    server = tf.train.Server(cluster, job_name=job_name, task_index=task_index)

    num_worker = len(worker_hosts)
    num_ps = len(ps_hosts)
    if job_name == 'ps':
        logging.info("******Start server******")
        server.join()
    elif job_name == 'worker':
        train_class = TrainClass(conf_dict)
        device = tf.train.replica_device_setter(worker_device='/job:worker/task:%d' % task_index, cluster=cluster)
        train_class.ConstructGraph(device,server)
        #print_trainable_variables(train_class.sess, 'save.model.txt')

        iter = 0
        err_rate = 1.0

        # every five minutes start one job
        if conf_dict["use_sync"] is False:
            wait_time = 60 * 5 * task_index 
            time.sleep(wait_time)
        while iter < 15:
            train_start_t = time.time()
            shuffle = False
            if iter > 0:
                shuffle = True
            tmp_tr_err_rate = train_class.TrainLogic(device, shuffle = shuffle, train_loss = True, skip_offset = iter)

            train_end_t = time.time()
            logging.info("******train %d iter time is %f ******" % (iter, train_end_t-train_start_t))
            # write text model
            if task_index == 0:
                print_trainable_variables(train_class.sess, conf_dict["checkpoint_dir"] + '/save.model.txt-' + str(iter))
#            tmp_cv_err_rate = train_class.TrainLogic(device, shuffle = False, train_loss = False, skip_offset = iter)
            iter += 1


