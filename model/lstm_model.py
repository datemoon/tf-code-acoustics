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
        self.grad_clip = 5
        self.num_layers = 3
        self.hidden_size =  1024
        self.num_proj = 1024 # NOTE HERE
        self.dropout_input_keep_prob = 1.0
        self.dropout_output_keep_prob = 1.0
        self.lr_decay_factor = 0.5
        self.batch_size = 16# nstreams
        self.num_frames_batch = 20
        self.output_size = 4223
        self.time_major = True
        self.forward_only = False
        self.Debug = True
    def initial(self, config_dict):
        for key in self.__dict__:
            if key in config_dict.keys():
                self.__dict__[key] = config_dict[key]
    def __repr__(self):
        pri = '{\nProjConfig:\n'
        for key in self.__dict__:
            pri += key + ':\t' + str(self.__dict__[key]) +'\n'
        pri += '}'
        return pri


class LSTM_Model(object):

    '''LSTM model'''
    def __init__(self, config): #initializer=tf.contrib.layers.xavier_initializer()):
        self.num_layers = config.num_layers
        self.hidden_size = config.hidden_size #512
        self.num_proj = config.num_proj       #256
        self.batch_size = config.batch_size
        self.num_frames_batch = config.num_frames_batch #20
#        self.input_dim = config.input_dim
        self.learn_rate = config.learning_rate

        self.output_size = config.output_size
        
        self.forward_only = config.forward_only
        self.keep_prob = config.dropout_output_keep_prob
        self.time_major = config.time_major
        self.Debug = config.Debug
        
        with tf.variable_scope('Softmax_layer'):
            self.W = tf.get_variable('softmax_w', [self.num_proj, self.output_size], dtype=tf.float32, 
                    initializer=tf.contrib.layers.xavier_initializer())
            self.bias = tf.get_variable('softmax_b', [self.output_size], dtype=tf.float32, 
                    initializer=tf.contrib.layers.xavier_initializer())
                    ##initializer=tf.constant_initializer(0.0))
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

        # Define some variables to store the RNN state
        # Note : tensorflow keep the state inside a batch but it's necessary to do this in order to keep the state
        #        between batches, especially when doing live transcript
        #        Another way would have been to get the state as an output of the session and feed it every time but
        #        this way is much more efficient
        with tf.variable_scope('Hidden_state'):
            state_variables = []
            for state_c, state_h in cell.zero_state(self.batch_size, tf.float32):
                state_variables.append(tf.contrib.rnn.LSTMStateTuple(
                    tf.Variable(state_c, trainable=False),
                    tf.Variable(state_h, trainable=False)))
            # Return as a tuple, so that it can be fed to dynamic_rnn as an initial state
            rnn_tuple_state = tuple(state_variables)

        # Build the RNN
        with tf.name_scope("LSTM"):
            rnn_outputs, new_states = tf.nn.dynamic_rnn(cell=cell,
                    inputs=rnn_input,
                    sequence_length=seq_len,
                    initial_state=None,
                    dtype=tf.float32,
                    time_major=self.time_major)
#        print("rnn_outputs:",rnn_outputs.shape[2])

        # Define an op to keep the hidden state between batches
        update_ops = []
        for state_variable, new_state in zip(rnn_tuple_state, new_states):
            # Assign the new state to the state variables on this layer
            update_ops.extend([state_variable[0].assign(new_state[0]),
                state_variable[1].assign(new_state[1])])
        # Return a tuple in order to combine all update_ops into a single operation.
        # The tuple's actual value should not be used.
        rnn_keep_state_op = tf.tuple(update_ops)
        
        # Define an op to reset the hidden state to zeros
        update_ops = []
        for state_variable in rnn_tuple_state:
            # Assign the new state to the state variables on this layer
            update_ops.extend([state_variable[0].assign(tf.zeros_like(state_variable[0])),
                state_variable[1].assign(tf.zeros_like(state_variable[1]))])
        # Return a tuple in order to combine all update_ops into a single operation.
        # The tuple's actual value should not be used.
        rnn_state_zero_op = tf.tuple(update_ops)


        if not self.time_major:
            rnn_outputs = tf.transpose(rnn_outputs, [1, 0, 2]) # [time, batch_size, cell_outdim]
        
        batch_size = self.batch_size
        print(batch_size,self.num_proj,self.output_size,seq_len.shape)
        rnn_outputs = tf.reshape(rnn_outputs, [-1, self.num_proj])
        logits = tf.matmul(rnn_outputs, self.W) + self.bias
        logits = tf.reshape(logits, [-1, batch_size, self.output_size])
        #output_log = tf.nn.softmax(logits)
        #output_log = tf.reshape(output_log, [seq_len.shape, -1, self.output_size])
        return logits, rnn_keep_state_op, rnn_state_zero_op

    def loss(self, X, labels, seq_len):
        #print(seq_len)
        #print(X)
        #print(labels)
        output_log, rnn_keep_state_op, rnn_state_zero_op = self.lstm_def(X, seq_len)
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
        
    def ce_train(self, X, labels, seq_len):
        output_log, rnn_keep_state_op, rnn_state_zero_op = self.lstm_def(X, seq_len)
        if self.Debug:
            softval = tf.nn.softmax(output_log)
        
        if self.time_major:
            labels = tf.transpose(labels)
        with tf.name_scope('CE'):
            print('********************',labels.shape)
            print('*****************')
            ce_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=output_log, name="ce_loss")
            print('********************',ce_loss.shape,)
            print('*****************')
            mask = tf.cast(tf.reshape(
                tf.transpose(tf.sequence_mask(seq_len, self.num_frames_batch)), [-1]), 
                tf.float32)

            total_frames = tf.cast(tf.reduce_sum(seq_len) ,tf.float32)
            mean_loss = tf.reduce_sum(mask * tf.reshape(ce_loss,[-1])) / total_frames
            
            label_error_rate = self.calculate_label_error_rate(output_log, labels, mask, total_frames)

        return mean_loss, ce_loss, rnn_keep_state_op, rnn_state_zero_op ,label_error_rate

    def calculate_label_error_rate(self, output_log, labels, mask, total_frames):
        #tf.reshape(output_log, [-1])
        #tf.reshape(labels, [-1])
        correct_prediction = tf.equal( tf.argmax(tf.reshape(output_log, [-1,self.output_size]), 1) , 
                tf.cast(tf.reshape(labels, [-1]), tf.int64))
        accuracy = tf.reduce_sum(tf.cast(correct_prediction, tf.float32) * mask) / total_frames
        return 1-accuracy
