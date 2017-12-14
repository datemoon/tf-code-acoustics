#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os, sys, shutil, time
import random
import threading
import Queue
import numpy as np
import time
import logging

from io_func import sparse_tuple_from
from io_func.kaldi_io_parallel import KaldiDataReadParallel
from parse_args import parse_args
from model.lstm_model import ProjConfig, LSTM_Model

import tensorflow as tf



class train_class(object):
    def __init__(self, conf_dict):
        self.nnet_conf = ProjConfig()
        self.nnet_conf.initial(conf_dict)

        self.kaldi_io_nstream = None
        # init train file
        self.kaldi_io_nstream_train = KaldiDataReadParallel()
        self.input_dim = self.kaldi_io_nstream_train.initialize_read(conf_dict, True)

        # init cv file
        self.kaldi_io_nstream_cv = KaldiDataReadParallel()
        self.kaldi_io_nstream_cv.initialize_read(conf_dict, first_time_reading = True, 
                scp_file = conf_dict['cv_scp'], label = conf_dict['cv_label'])

        self.num_batch_total = 0

        logging.info(self.nnet_conf.__repr__())
        logging.info(self.kaldi_io_nstream.__repr__())

        self.tf_async_model_prefix = conf_dict['checkpoint_dir']
        self.num_threads = conf_dict['num_threads']
        self.queue_cache = conf_dict['queue_cache']
        self.input_queue = Queue.Queue(self.queue_cache)
        self.acc_label_error_rate = []
        self.num_batch = []
        for i in range(self.num_threads):
            self.acc_label_error_rate.append(1.0)
            self.num_batch.append(0)
        if conf_dict.has_key('print_trainable_variables'):
            self.print_trainable_variables = conf_dict['print_trainable_variables']
        else:
            self.print_trainable_variables = False
        if conf_dict.has_key('use_normal'):
            self.use_normal = conf_dict['use_normal']
        else:
            self.use_normal = False
        if conf_dict.has_key('use_sgd'):
            self.use_sgd = conf_dict['use_sgd']
        else:
            self.use_sgd = True

        if conf_dict.has_key('restore_training'):
            self.restore_training = conf_dict['restore_training']
        else:
            self.restore_training = False

    def get_num(self,str):
        return int(str.split('/')[-1].split('_')[0])
                    #model_48434.ckpt.final
    def construct_graph(self):
        with tf.Graph().as_default():
            self.run_ops = []
            self.X = tf.placeholder(tf.float32, [None, None, self.input_dim], name='feature')
            self.Y = tf.sparse_placeholder(tf.int32, name="labels")
            self.seq_len = tf.placeholder(tf.int32,[None], name = 'seq_len')

            self.learning_rate_var = tf.Variable(float(self.nnet_conf.learning_rate), trainable=False, name='learning_rate')
            if self.use_sgd:
                optimizer = tf.train.GradientDescentOptimizer(self.learning_rate_var)
            else:
                optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_var, beta1=0.9, beta2=0.999, epsilon=1e-08)

            for i in range(self.num_threads):
                with tf.device("/gpu:%d" % i):
                    initializer = tf.random_uniform_initializer(
                            -self.nnet_conf.init_scale, self.nnet_conf.init_scale)
                    model = LSTM_Model(self.nnet_conf)
                    mean_loss, ctc_loss , label_error_rate, decoded, softval = model.loss(self.X, self.Y, self.seq_len)
                    if self.use_sgd and self.use_normal:
                        tvars = tf.trainable_variables()
                        grads, _ = tf.clip_by_global_norm(tf.gradients(
                            mean_loss, tvars), self.nnet_conf.grad_clip)
                        train_op = optimizer.apply_gradients(
                                zip(grads, tvars),
                                global_step=tf.contrib.framework.get_or_create_global_step())
                    else:
                        train_op = optimizer.minimize(mean_loss)

                    run_op = {'train_op':train_op,
                            'mean_loss':mean_loss,
                            'ctc_loss':ctc_loss,
                            'label_error_rate':label_error_rate,
                            'decoded':decoded,
                            'softval':softval}
                    self.run_ops.append(run_op)
                    tf.get_variable_scope().reuse_variables()

            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)
            self.sess = tf.Session(config=tf.ConfigProto(
                intra_op_parallelism_threads=self.num_threads, allow_soft_placement=True,
                log_device_placement=False, gpu_options=gpu_options))
            init = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
            
            tmp_variables=tf.trainable_variables()
            self.saver = tf.train.Saver(tmp_variables, max_to_keep=100)

            #self.saver = tf.train.Saver(max_to_keep=100, sharded = True)
            if self.restore_training:
                self.sess.run(init)
                ckpt = tf.train.get_checkpoint_state(self.tf_async_model_prefix)
                if ckpt and ckpt.model_checkpoint_path:
                    logging.info("restore training")
                    self.saver.restore(self.sess, ckpt.model_checkpoint_path)
                    self.num_batch_total = self.get_num(ckpt.model_checkpoint_path)
                    if self.print_trainable_variables == True:
                        print_trainable_variables(self.sess, ckpt.model_checkpoint_path+'.txt')
                        sys.exit(0)
                    logging.info('model:'+ckpt.model_checkpoint_path)
                    logging.info('restore learn_rate:'+str(self.sess.run(self.learning_rate_var)))
                    #print('*******************',self.num_batch_total)
                    #time.sleep(3)
                    #model_48434.ckpt.final
                    #print("ckpt.model_checkpoint_path",ckpt.model_checkpoint_path)
                    #print("self.tf_async_model_prefix",self.tf_async_model_prefix)
                    #self.saver.restore(self.sess, self.tf_async_model_prefix)
                    
                else:
                    logging.info('No checkpoint file found')
                    self.sess.run(init)
                    logging.info('init learn_rate:'+str(self.sess.run(self.learning_rate_var)))
            else:
                self.sess.run(init)

            self.total_variables = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
            logging.info('total parameters : %d' % self.total_variables)

    def train_function(self, gpu_id, run_op, thread_name):
        total_acc_error_rate = 0.0
        num_batch = 0
        while True:
            time1=time.time()
            feat,sparse_label,length = self.get_feat_and_label()
            if feat is None:
                logging.info('thread end : %s' % thread_name)
                break
            time2=time.time()
            print('******time:',time2-time1, thread_name)

            feed_dict = {self.X : feat, self.Y : sparse_label, self.seq_len : length}

            time3 = time.time()
            calculate_return = self.sess.run(run_op, feed_dict = feed_dict)
            time4 = time.time()

            print(num_batch," time:",time2-time1,time3-time2,time4-time3,time4-time1)
            print('label_error_rate:',calculate_return['label_error_rate'])
            print('mean_loss:',calculate_return['mean_loss'])
            print('ctc_loss:',calculate_return['ctc_loss'])

            num_batch += 1
            total_acc_error_rate += calculate_return['label_error_rate']
            self.acc_label_error_rate[gpu_id] = total_acc_error_rate 
            self.num_batch[gpu_id] = num_batch
        self.acc_label_error_rate[gpu_id] = total_acc_error_rate

    def cv_function(self, gpu_id, run_op, thread_name):
        total_acc_error_rate = 0.0
        num_batch = 0
        while True:
            feat,sparse_label,length = self.get_feat_and_label()
            if feat is None:
                logging.info('thread end : %s' % thread_name)
                break
            feed_dict = {self.X : feat, self.Y : sparse_label, self.seq_len : length}
            run_need_op = {'mean_loss':run_op['mean_loss'],
                    'ctc_loss':run_op['ctc_loss'],
                    'label_error_rate':run_op['label_error_rate']}
            calculate_return = self.sess.run(run_need_op, feed_dict = feed_dict)
            print('label_error_rate:',calculate_return['label_error_rate'])
            num_batch += 1
            total_acc_error_rate += calculate_return['label_error_rate']
            self.acc_label_error_rate[gpu_id] = total_acc_error_rate 
            self.num_batch[gpu_id] = num_batch
        self.acc_label_error_rate[gpu_id] = total_acc_error_rate

    def get_feat_and_label(self):
        return self.input_queue.get()

    def input_feat_and_label(self):
        feat,label,length = self.kaldi_io_nstream.load_next_nstreams()
        if length is None:
            return False
        if len(label) != self.nnet_conf.batch_size:
             return False
        sparse_label = sparse_tuple_from(label)
        self.input_queue.put((feat,sparse_label,length))
        self.num_batch_total += 1
        print('total_batch_num**********',self.num_batch_total,'***********')
        return True

    def cv_logic(self):
        self.kaldi_io_nstream = self.kaldi_io_nstream_cv
        train_thread = []
        for i in range(self.num_threads):
#            self.acc_label_error_rate.append(1.0)
            train_thread.append(threading.Thread(group=None, target=self.cv_function,
                args=(i, self.run_ops[i], 'thread_hubo_'+str(i)), name='thread_hubo_'+str(i)))

        for thr in train_thread:
            thr.start()
        logging.info('cv thread start.')

        while True:
            if self.input_feat_and_label():
                continue
            break

        logging.info('cv read feat ok')
        for thr in train_thread:
            self.input_queue.put((None, None, None))

        while True:
            if self.input_queue.empty():
                break;

        logging.info('cv is end.')
        for thr in train_thread:
            thr.join()
            logging.info('join cv thread %s' % thr.name)
            
        tmp_label_error_rate = self.get_avergae_label_error_rate()
        self.kaldi_io_nstream.reset_read(reset_offset=False)
        self.reset_acc()
        return tmp_label_error_rate

    def train_logic(self):
        self.kaldi_io_nstream = self.kaldi_io_nstream_train 
        train_thread = []
        for i in range(self.num_threads):
#            self.acc_label_error_rate.append(1.0)
            train_thread.append(threading.Thread(group=None, target=self.train_function,
                args=(i, self.run_ops[i], 'thread_hubo_'+str(i)), name='thread_hubo_'+str(i)))

        for thr in train_thread:
            thr.start()

        logging.info('train thread start.')

        all_lab_err_rate = []
        for i in range(5):
            all_lab_err_rate.append(1.1)

        while True:
            if self.num_batch_total % 3000 == 0:
                while True:
                    #print('wait save mode')
                    time.sleep(0.5)
                    if self.input_queue.empty():
                        checkpoint_path = os.path.join(self.tf_async_model_prefix, str(self.num_batch_total)+'_model'+'.ckpt')
                        logging.info('save model: '+checkpoint_path+ 
                                '--- learn_rate: ' + 
                                str(self.sess.run(self.learning_rate_var)))
                        self.saver.save(self.sess, checkpoint_path)

                        if self.num_batch_total == 0:
                            break

                        curr_lab_err_rate = self.get_avergae_label_error_rate()
                        logging.info("current label error rate : %f\n" % curr_lab_err_rate)
                        all_lab_err_rate.sort()
                        for i in range(len(all_lab_err_rate)):
                            if curr_lab_err_rate < all_lab_err_rate[i]:
                                all_lab_err_rate[len(all_lab_err_rate)-1] = curr_lab_err_rate
                                break
                            if i == len(all_lab_err_rate)-1:
                                train_logic.decay_learning_rate(0.8)
                                all_lab_err_rate[len(all_lab_err_rate)-1] = curr_lab_err_rate
                        break
            if self.input_feat_and_label():
                continue
            break
        time.sleep(1)
        logging.info('read feat ok')

        '''
            end train
        '''
        for thr in train_thread:
            self.input_queue.put((None, None, None))

        while True:
            if self.input_queue.empty():
#                logging.info('train is ok')
                checkpoint_path = os.path.join(self.tf_async_model_prefix, str(self.num_batch_total)+'_model'+'.ckpt')
                logging.info('save model: '+checkpoint_path+ 
                        '.final --- learn_rate: ' + 
                        str(self.sess.run(self.learning_rate_var)))
                self.saver.save(self.sess, checkpoint_path+'.final')
                break;
        '''
            train is end
        '''
        logging.info('train is end.')
        for thr in train_thread:
            thr.join()
            logging.info('join thread %s' % thr.name)

        tmp_label_error_rate = self.get_avergae_label_error_rate()
        self.kaldi_io_nstream.reset_read(reset_offset=False)
        self.reset_acc()
        return tmp_label_error_rate

    def decay_learning_rate(self, lr_decay_factor):
        learning_rate_decay_op = self.learning_rate_var.assign(tf.multiply(self.learning_rate_var, lr_decay_factor))
        self.sess.run(learning_rate_decay_op)
        logging.info('learn_rate decay to '+str(self.sess.run(self.learning_rate_var)))
        logging.info('lr_decay_factor is '+str(lr_decay_factor))
#        return learning_rate_decay_op

    def get_avergae_label_error_rate(self):
        tot_label_error_rate = 0.0
        tot_num_batch = 0
        for i in range(self.num_threads):
            tot_label_error_rate += self.acc_label_error_rate[i]
            tot_num_batch += self.num_batch[i]
        if tot_num_batch == 0:
            average_label_error_rate = 1.0
        else:
            average_label_error_rate = tot_label_error_rate / tot_num_batch
#        logging.info("average label error rate : %f\n" % average_label_error_rate)
        return average_label_error_rate

    def reset_acc(self):
        for i in range(len(self.acc_label_error_rate)):
            self.acc_label_error_rate[i] = 1.0
            self.num_batch[i] = 0


if __name__ == "__main__":
    #first read parameters
    # read config file
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

    train_logic = train_class(conf_dict)
    train_logic.construct_graph()
    iter = 0
    err_rate = 1.0
    while iter < 15:
        train_start_t = time.time()
        tmp_err_rate = train_logic.train_logic()

        train_end_t = time.time()
        tmp_cv_err_rate = train_logic.cv_logic()
        cv_end_t = time.time()
        logging.info("iter %d: train data average label error rate : %f" % (iter,tmp_err_rate))
        logging.info("iter %d: cv data average label error rate : %f" % (iter,tmp_cv_err_rate))
        logging.info('train time %f, cv time %f' % 
                (train_end_t-train_start_t, cv_end_t-train_end_t))
        iter += 1
        if tmp_cv_err_rate > 1.0:
            if err_rate != 1.0:
                print('this is a error!')
            continue
        if err_rate > (tmp_cv_err_rate + 0.005):
            err_rate = tmp_cv_err_rate
        else:
            err_rate = tmp_cv_err_rate
            train_logic.decay_learning_rate(0.5)
        #time.sleep(5)
    logging.info('end')


