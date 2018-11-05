#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
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

class TrainClass(object):
    '''
    '''
    def __init__(self, conf_dict):
        # configure paramtere
        self.conf_dict = conf_dict
        self.print_trainable_variables_cf = False
        self.use_normal_cf = False
        self.use_sgd_cf = True
        self.restore_training_cf = True
        self.checkpoint_dir_cf = None
        self.num_threads_cf = 1
        self.queue_cache_cf = 100
        self.task_index_cf = -1
        self.grad_clip_cf = 5.0
        self.feature_transfile_cf = None
        self.learning_rate_cf = 0.001
        self.batch_size_cf = 10

        # initial configuration parameter
        for attr in self.__dict__:
            if len(attr.split('_cf')) != 2:
                continue;
            key = attr.split('_cf')[0]
            if key in conf_dict.keys():
                self.__dict__[attr] = conf_dict[key]

        if self.feature_transfile_cf == None:
            logging.info('No feature_transfile,it must have.')
            sys.exit(1)
        feat_trans = FeatureTransform()
        feat_trans.LoadTransform(self.feature_transfile_cf)

        # init train file
        self.kaldi_io_nstream_train = KaldiDataReadParallel()
        self.input_dim = self.kaldi_io_nstream_train.Initialize(conf_dict,
                scp_file = conf_dict['tr_scp'], label = conf_dict['tr_label'],
                feature_transform = feat_trans, criterion = 'ctc')
        # init cv file
        self.kaldi_io_nstream_cv = KaldiDataReadParallel()
        self.kaldi_io_nstream_cv.Initialize(conf_dict,
                scp_file = conf_dict['cv_scp'], label = conf_dict['cv_label'],
                feature_transform = feat_trans, criterion = 'ctc')

        self.num_batch_total = 0
        self.tot_lab_err_rate = 0.0
        self.tot_num_batch = 0.0
        self.num_save = 0

        logging.info(self.kaldi_io_nstream_train.__repr__())
        logging.info(self.kaldi_io_nstream_cv.__repr__())
        
        # Initial input queue.
        self.input_queue = Queue.Queue(self.queue_cache_cf)
        return
    
    # multi computers construct train graph
    def ConstructGraph(self, device, server):
        with tf.device(device):
            self.X = tf.placeholder(tf.float32, [None, None, self.input_dim], 
                    name='feature')
            self.Y = tf.sparse_placeholder(tf.int32, name="labels")
            self.seq_len = tf.placeholder(tf.int32,[None], name = 'seq_len')
            self.learning_rate_var_tf = tf.Variable(float(self.learning_rate_cf), 
                    trainable=False, name='learning_rate')
            if self.use_sgd_cf:
                optimizer = tf.train.GradientDescentOptimizer(self.learning_rate_var_tf)
            else:
                optimizer = tf.train.AdamOptimizer(learning_rate=
                        self.learning_rate_var_tf, beta1=0.9, beta2=0.999, epsilon=1e-08)
            nnet_model = LstmModel(self.conf_dict)
            ctc_mean_loss, ctc_loss , label_error_rate, decoded = nnet_model.CtcLoss(
                    self.X, self.Y, self.seq_len)

            if self.use_sgd_cf and self.use_normal_cf:
                tvars = tf.trainable_variables()
                grads, _ = tf.clip_by_global_norm(tf.gradients(
                    ctc_mean_loss, tvars), self.grad_clip_cf)
                train_op = optimizer.apply_gradients(
                        zip(grads, tvars),
                        global_step=tf.contrib.framework.get_or_create_global_step())
            else:
                train_op = optimizer.minimize(ctc_mean_loss)
            # set run operation
            self.run_ops = {'train_op':train_op,
                    'ctc_mean_loss':ctc_mean_loss,
                    'ctc_loss':ctc_loss,
                    'label_error_rate':label_error_rate}

            # set initial parameter
            self.init_para = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())

            tmp_variables=tf.trainable_variables()
            self.saver = tf.train.Saver(tmp_variables, max_to_keep=100)

            self.total_variables = np.sum([np.prod(v.get_shape().as_list()) 
                for v in tf.trainable_variables()])
            logging.info('total parameters : %d' % self.total_variables)

            # set gpu option
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)

            # session config
            sess_config = tf.ConfigProto(intra_op_parallelism_threads=self.num_threads,
                    inter_op_parallelism_threads=self.num_threads,
                    allow_soft_placement=True,
                    log_device_placement=False,gpu_options=gpu_options)
            global_step = tf.contrib.framework.get_or_create_global_step()

            sv = tf.train.Supervisor(is_chief=(self.task_index_cf==0),
                    global_step=global_step,
                    init_op = init,
                    logdir = self.checkpoint_dir_cf,
                    saver=self.saver,
                    save_model_secs=3600,
                    checkpoint_basename='model.ckpt')

            self.sess = sv.prepare_or_wait_for_session(server.target, config=sess_config)

        return 

    def SaveTextModel(self):
        if self.print_trainable_variables_cf == True:
            ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir_cf)
            if ckpt and ckpt.model_checkpoint_path:
                print_trainable_variables(self.sess, ckpt.model_checkpoint_path+'.txt')

    def InputFeat(self, input_lock):
        while True:
            input_lock.acquire()
            feat,label,length = self.kaldi_io_nstream.LoadNextNstreams()
            if length is None:
                break
            if len(label) != self.batch_size_cf:
                break
            sparse_label = sparse_tuple_from(label)
            self.input_queue.put((feat,sparse_label,length))
            self.num_batch_total += 1
            if self.num_batch_total % 3000 == 0:
                self.SaveModel()
                self.AdjustLearnRate()
            print('total_batch_num**********',self.num_batch_total,'***********')
            input_lock.release()
        self.input_queue.put((None, None, None))

    def ThreadInputFeatAndLab(self):
        input_thread = []
        input_lock = threading.Lock()
        for i in range(1):
            input_thread.append(threading.Thread(group=None, target=self.InputFeat,
                args=(input_lock,),name='read_thread'+str(i)))

        for thr in input_thread:
            logging.info('ThreadInputFeatAndLab start')
            thr.start()

        return input_thread

    def SaveModel(self):
        while True:
            time.sleep(1.0)
            if self.input_queue.empty():
                checkpoint_path = os.path.join(self.checkpoint_dir, str(self.num_batch_total)+'_model'+'.ckpt')
                logging.info('save model: '+checkpoint_path+
                        ' --- learn_rate: ' +
                        str(self.sess.run(self.learning_rate_var_tf)))
                self.saver.save(self.sess, checkpoint_path)
                break

    # if current label error rate less then previous five
    def AdjustLearnRate(self):
        curr_lab_err_rate = self.get_avergae_label_error_rate()
        logging.info("current label error rate : %f\n" % curr_lab_err_rate)
        all_lab_err_rate_len = len(self.all_lab_err_rate)
        for i in range(all_lab_err_rate_len):
            if curr_lab_err_rate < self.all_lab_err_rate[i]:
                break
            if i == len(self.all_lab_err_rate)-1:
                self.decay_learning_rate(0.8)
        self.all_lab_err_rate[self.num_save%all_lab_err_rate_len] = curr_lab_err_rate
        self.num_save += 1

    # train_loss is a open train or cv .
    def TrainLogic(self, device, shuffle = False, train_loss = True, skip_offset = 0):
        if train_loss == True:
            self.kaldi_io_nstream = self.kaldi_io_nstream_train
            run_op = self.run_ops[0]
        else:
            self.kaldi_io_nstream = self.kaldi_io_nstream_cv
            run_op = {'label_error_rate':run_op['label_error_rate'],
                    'ctc_mean_loss':run_op['ctc_mean_loss']}
        
        threadinput = self.ThreadInputFeatAndLab()
        time.sleep(3)
        logging.info('train start.')
        with tf.device(device):
            self.TrainFunction(0, run_op, 'train_thread_hubo')

        logging.info('train end.')
        threadinput[0].join()

        tmp_label_error_rate = self.get_avergae_label_error_rate()
        self.kaldi_io_nstream.Reset(shuffle = shuffle, skip_offset = skip_offset)
        self.ResetAccuracy()
        return tmp_label_error_rate

    def TrainFunction(self, gpu_id, run_op, thread_name):
        logging.info('******start TrainFunction******')
        total_acc_error_rate = 0.0
        num_batch = 0
        self.acc_label_error_rate[gpu_id] = 0.0
        self.num_batch[gpu_id] = 0

        while True:
            time1=time.time()
            feat, sparse_label, length = self.GetFeatAndLabel()
            if feat is None:
                logging.info('train thread end : %s' % thread_name)
                break
            time2=time.time()

            feed_dict = {self.X : feat, self.Y : sparse_label, self.seq_len : length}
            time3 = time.time()
            calculate_return = self.sess.run(run_op, feed_dict = feed_dict)
            time4 = time.time()

            print("thread_name: ", thread_name,  num_batch," time:",time2-time1,time3-time2,time4-time3,time4-time1)
            print('label_error_rate:',calculate_return['label_error_rate'])
            print('ctc_mean_loss:',calculate_return['ctc_mean_loss'])
            #print('ctc_loss:',calculate_return['ctc_loss'])

            num_batch += 1
            total_acc_error_rate += calculate_return['label_error_rate']
            self.acc_label_error_rate[gpu_id] += calculate_return['label_error_rate']
            self.num_batch[gpu_id] += 1
        logging.info('******end TrainFunction******')
    
    def GetFeatAndLabel(self):
        return self.input_queue.get()


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

    ps_hosts = conf_dict['ps_hosts'].split(',')
    worker_hosts = conf_dict['worker_hosts'].split(',')
    job_name = conf_dict['job_name']
    task_index = conf_dict['task_index']

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

        iter = 0
        err_rate = 1.0
        while iter < 1:
            train_start_t = time.time()
            tmp_tr_err_rate = train_class.TrainLogic(device, shuffle = False, train_loss = True, skip_offset = iter)

            train_end_t = time.time() 
            tmp_cv_err_rate = train_class.TrainLogic(device, shuffle = False, train_loss = False, skip_offset = iter)


