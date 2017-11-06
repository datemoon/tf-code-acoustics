from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
import sys
import os

class ProjConfig(object):
    """Projection config. Compared with Hidden7Config"""
    def __init__(self):
        self.init_scale = 0.01 # scale to initialize LSTM weights
        self.learning_rate = 1.0
        self.max_grad_norm = 5
        self.num_layers = 3
        self.hidden_size =  1024
        self.num_proj = 512 # NOTE HERE
        self.keep_prob = 1.0
        self.lr_decay = 0.5
        self.batch_size = 16# nstreams
        self.output_size = 4223
        self.time_major = True
        self.forward_only = False
        self.Debug = True
    def initial(self, config_dict):
        for key in self.__dict__:
            if key in config_dict.keys():
                self.__dict__[key] = config_dict[key]
    def __repr__(self):
        pri = ''
        for key in self.__dict__:
            pri += key + ':\t' + str(self.__dict__[key]) +'\n'
        return pri


class LSTM_Model(object):

    '''LSTM model'''
    def __init__(self, config): #initializer=tf.contrib.layers.xavier_initializer()):
        self.num_layers = config.num_layers
        self.hidden_size = config.hidden_size #512
        self.num_proj = config.num_proj       #256
        self.batch_size = config.batch_size
#        self.input_dim = config.input_dim
        self.learn_rate = config.learning_rate

        self.output_size = config.output_size
        
        self.forward_only = config.forward_only
        self.keep_prob = config.keep_prob
        self.time_major = config.time_major
        self.Debug = config.Debug
        
        with tf.variable_scope('Softmax_layer'):
            self.W = tf.get_variable('softmax_w', [self.num_proj, self.output_size], dtype=tf.float32, 
                    initializer=tf.contrib.layers.xavier_initializer())
            self.bias = tf.get_variable('softmax_b', [self.output_size], dtype=tf.float32, 
                    initializer=tf.constant_initializer(0.0))
    # End __init__ 


    def lstm_def(self, rnn_input, seq_len):
        # Automatically reset state in each batch
        # Define cells of acoustic model
        with tf.variable_scope('LSTM'):
            def lstm_cell():
                if self.num_proj == self.hidden_size:
                    return tf.contrib.rnn.LSTMCell(
                            self.hidden_size, state_is_tuple=True, reuse=tf.get_variable_scope().reuse)
                else:
                    return tf.contrib.rnn.LSTMCell(self.hidden_size, use_peepholes=True, num_proj=self.num_proj, 
                            forget_bias=0.0, state_is_tuple=True, reuse=tf.get_variable_scope().reuse)

            layers_list = []
            for _ in range(self.num_layers):
                cell = lstm_cell()
                if not self.forward_only:
                    if self.keep_prob < 1.0:
                        cell = tf.contrib.rnn.DropoutWarpper(cell, output_keep_prob = self.keep_prob)
                layers_list.append(cell)

            # Store the layers in a multi-layer RNN
            cell = tf.contrib.rnn.MultiRNNCell(layers_list, state_is_tuple=True)

        with tf.name_scope("LSTM"):
            rnn_outputs, state = tf.nn.dynamic_rnn(cell=cell,
                    inputs=rnn_input,
                    sequence_length=seq_len,
                    initial_state=None,
                    dtype=tf.float32,
                    time_major=self.time_major)
#        print("rnn_outputs:",rnn_outputs.shape[2])
        if not self.time_major:
            rnnrnn__outputs = tf.transpose(rnn_outputs, [1, 0, 2]) # [time, batch_size, cell_outdim]
        
        batch_size = self.batch_size
        print(batch_size,self.num_proj,self.output_size,seq_len.shape)
        rnn_outputs = tf.reshape(rnn_outputs, [-1, self.num_proj])
        logits = tf.matmul(rnn_outputs, self.W) + self.bias
        logits = tf.reshape(logits, [-1, batch_size, self.output_size])
        #output_log = tf.nn.softmax(logits)
        #output_log = tf.reshape(output_log, [seq_len.shape, -1, self.output_size])
        return logits

    def loss(self, X, labels, seq_len):
        #print(seq_len)
        #print(X)
        #print(labels)
        #sys.exit(0)
        output_log = self.lstm_def(X, seq_len)
        if self.Debug:
            softval = tf.nn.softmax(output_log)
        if True:
            decoded, log_prob = tf.nn.ctc_greedy_decoder(output_log, 
                    seq_len, merge_repeated=True)
        else:
            decoded, log_prob = tf.nn.ctc_beam_search_decoder(output_log, 
                    seq_len, merge_repeated=True)
        # Inaccuracy: label error rate
        label_error_rate = tf.reduce_mean(
                tf.edit_distance(tf.to_int32(decoded[0]),labels))
        
        with tf.name_scope('CTC'):
            ctc_loss = tf.nn.ctc_loss(labels, output_log, seq_len,
                    time_major=self.time_major)
            # Compute the mean loss of the batch (only used to check on progression in learning)
            # The loss is averaged accross the batch but before we take into account the real size of the label
            mean_loss = tf.reduce_mean(tf.truediv(ctc_loss, tf.to_float(seq_len)))

        if self.Debug:
            return mean_loss, ctc_loss ,label_error_rate, decoded[0], softval 
        else:
            return mean_loss, ctc_loss ,label_error_rate, decoded[0]
        

