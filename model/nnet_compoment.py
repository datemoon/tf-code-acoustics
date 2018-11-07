from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import sys
import os


class AffineTransformLayer(object):
    '''
    '''
    def __init__(self, conf_opt, dtype = tf.float32, 
            initializer = tf.contrib.layers.xavier_initializer(tf.float32),
            trainable = True):
        self.name = 'AffineTransformLayer'
        self.input_dim = None
        self.output_dim = None
        for key in self.__dict__:
            if key in conf_opt.keys():
                self.__dict__[key] = conf_opt[key]
        assert self.input_dim != None
        assert self.output_dim != None
        self.input_dim = int(self.input_dim)
        self.output_dim = int(self.output_dim)
        
        with tf.variable_scope('AffineLayer'):
            self.weights = tf.get_variable(self.name+'_w',
                    shape = [self.input_dim, self.output_dim],
                    dtype = dtype,
                    initializer = initializer,
                    trainable = trainable)

            self.bias = tf.get_variable(self.name+'_b', 
                    shape = [self.output_dim],
                    dtype = dtype,
                    initializer = initializer,
                    trainable = trainable)

    def __call__(self, input_feats):
        input_feats = tf.reshape(input_feats, [-1, self.input_dim])
        return tf.matmul(input_feats, self.weights) + self.bias

    def GetOutputDim(self):
        return self.output_dim

class LstmLayer(object):
    '''
    name = lstmlayer1; lstm_cell = 1024; use_peepholes = True; cell_clip = 5.0; initializer = None; num_proj = 512; self.proj_clip = 1.0; forget_bias = 0.0; state_is_tuple = True; activation = None; reuse = True; dtype = tf.float32;
    '''
    def __init__(self, conf_opt,
            dtype = tf.float32,
            initializer = tf.contrib.layers.xavier_initializer(tf.float32),
            activation = None):
        self.lstm_cell = None
        self.use_peepholes = False
        self.cell_clip = 50.0
        self.num_proj = None
        self.proj_clip = None # advise 1.0
        self.forget_bias = 0.0
        self.state_is_tuple = True
        self.name = None
        self.reuse = None
        self.keep_prob = 1.0
        for key in self.__dict__:
            if key in conf_opt.keys():
                self.__dict__[key] = conf_opt[key]
        
        # no configure
        self.initializer = initializer
        self.activation = activation # default tanh
        self.dtype = dtype
        
        assert self.lstm_cell != None
        self.lstm_cell = int(self.lstm_cell)
        if self.use_peepholes == 'False':
            self.use_peepholes = False
        else:
            self.use_peepholes = True

        self.cell_clip = float(self.cell_clip)
        if self.num_proj != None:
            self.num_proj = int(self.num_proj)
        if self.proj_clip != None:
            self.proj_clip = float(self.proj_clip)
        self.forget_bias = float(self.forget_bias)
        if self.state_is_tuple == 'False':
            self.state_is_tuple = False
        else:
            self.state_is_tuple = True
        self.keep_prob = float(self.keep_prob)

        self.reuse = tf.get_variable_scope().reuse


    def __call__(self):
        cell = tf.contrib.rnn.LSTMCell(self.lstm_cell,
                use_peepholes = self.use_peepholes,
                cell_clip = self.cell_clip,
                initializer = self.initializer,
                num_proj = self.num_proj,
                proj_clip = self.proj_clip,
                forget_bias = self.forget_bias,
                state_is_tuple = self.state_is_tuple,
                activation = self.activation,
                reuse = self.reuse,
                name = self.name,
                dtype = self.dtype)
        if self.keep_prob < 1.0:
            cell = tf.contrib.rnn.DropoutWarpper(cell, output_keep_prob = self.keep_prob)
        return cell

    def GetOutputDim(self):
        if self.num_proj == None:
            return self.lstm_cell
        else:
            return self.num_proj

# enum layer 
g_layer_dict={'AffineTransformLayer':AffineTransformLayer,
        'LstmLayer':LstmLayer
        }
