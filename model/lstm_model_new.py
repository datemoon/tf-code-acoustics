from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
import sys
import os

from nnet_base import NnetBase
import nnet_compoment

class LstmModel(object, NnetBase):
    '''
    Lstm model.
    '''
    def __init__(self, conf_dict):
        self.num_layers_cf = 0
        self.batch_size_cf = 10
        self.num_frames_batch_cf = 20
        self.learn_rate_cf = 1e-4
        self.output_keep_prob_cf = 1.0
        self.time_major_cf = True
        self.state_is_tuple_cf = True
        self.nnet_conf_cf = None

        # Initial configuration parameter.
        for attr in self.__dict__:
            if len(attr.split('_cf')) != 2:
                continue;
            key = attr.split('_cf')[0]
            if key in conf_dict.keys():
                self.__dict__[attr] = conf_dict[key]

        # Initial nnet parameter
        self.nnet_conf_opt = NnetBase.ReadNnetConf(self.nnet_conf_cf)

    
    def CreateRnnModel(self, input_feats, seq_len):
        rnn_layers = []
        other_layer = []
        for layer_opt in self.nnet_conf_opt:
            if layer_opt['layer_flag'] == 'AffineTransformLayer':
                other_layer.append(nnet_compoment.AffineTransformLayer(layer_opt))
            elif layer_opt['layer_flag'] == 'LstmLayer':
                rnn_layer.append(nnet_compoment.LstmLayer(layer_opt))
            elif layer_opt['layer_flag'] == 'Sigmoid':
                other_layer.append(tf.nn.sigmoid)

        rnn_cells = tf.contrib.rnn.MultiRNNCell(rnn_layers, 
                state_is_tuple=self.state_is_tuple_cf)
        # Define some variables to store the RNN state
        # Note : tensorflow keep the state inside a batch but it's necessary to do this in order to keep the state
        #        between batches, especially when doing live transcript
        #        Another way would have been to get the state as an output of the session and feed it every time but
        #        this way is much more efficient
        with tf.variable_scope('Hidden_state'):
            state_variables = []
            for state_c, state_h in cell.zero_state(self.batch_size_cf,
                    tf.float32):
                state_variables.append(tf.contrib.rnn.LSTMStateTuple(
                    tf.Variable(state_c, trainable=False),
                    tf.Variable(state_h, trainable=False)))
            rnn_tuple_state = tuple(state_variables)

        # Build the RNN
        with tf.name_scope("LSTM"):
            rnn_outputs, new_states = tf.nn.dynamic_rnn(cell=rnn_cells,
                    inputs=input_feats,
                    sequence_length=seq_len,
                    initial_state=rnn_tuple_state,
                    dtype=tf.float32,
                    time_major=self.time_major_cf)

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

        if not self.time_major_cf:
            rnn_outputs = tf.transpose(rnn_outputs, [1, 0, 2]) # [time, batch_size, cell_outdim]

        #
        output = []
        output.append(rnn_outputs)
        for layer in other_layer:
           output.append(layer(output[-1]))

        # Get output dim
        output_dim = other_layer[-1].GetOutputdim()
        # last no softmax
        last_output = tf.reshape(output[-1], 
                [-1, self.batch_size_cf, output_dim])
        return last_output, rnn_keep_state_op, rnn_state_zero_op
                        
    def Run(self, input_feats, seq_len):

