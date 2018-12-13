from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import sys
import os


strset=('name')
class AffineTransformLayer(object):
    '''
    '''
    def __init__(self, conf_opt, dtype = tf.float32, 
            initializer = tf.contrib.layers.xavier_initializer(tf.float32),
            trainable = True):
        self.name = 'AffineTransformLayer'
        self.input_dim = None
        self.output_dim = None
        self.dtype = tf.float32
        self.initializer = tf.contrib.layers.xavier_initializer(tf.float32)
        self.trainable = True
        for key in self.__dict__:
            if key in conf_opt.keys():
                if key in strset:
                    continue
                self.__dict__[key] = eval(conf_opt[key])
        assert self.input_dim != None
        assert self.output_dim != None
        
        with tf.variable_scope(self.name):
            self.weights = tf.get_variable(self.name+'_w',
                    shape = [self.input_dim, self.output_dim],
                    dtype = self.dtype,
                    initializer = self.initializer,
                    trainable = self.trainable)

            self.bias = tf.get_variable(self.name+'_b', 
                    shape = [self.output_dim],
                    dtype = dtype,
                    initializer = initializer,
                    trainable = trainable)

    def __call__(self, input_feats):
        input_feats = tf.reshape(input_feats, [-1, self.input_dim])
        return tf.matmul(input_feats, self.weights) + self.bias

    def GetInputDim(self):
        return self.input_dim

    def GetOutputDim(self):
        return self.output_dim

class LstmLayer(object):
    '''
    layer_flag = BLstmLayer; name = lstmlayer1; lstm_cell = 1024; use_peepholes = True; cell_clip = 5.0; initializer = None; num_proj = 512; self.proj_clip = 1.0; forget_bias = 0.0; initializer = tf.contrib.layers.xavier_initializer(tf.float32); activation = None; dtype = tf.float32; reuse = tf.get_variable_scope().reuse; state_is_tuple = True;
    '''
    def __init__(self, conf_opt, prefix=None):
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
        self.initializer = tf.contrib.layers.xavier_initializer(tf.float32)
        self.activation = None # default tanh
        self.dtype = tf.float32
        self.reuse = tf.get_variable_scope().reuse
        for key in self.__dict__:
            if key in conf_opt.keys():
                if key in strset:
                    continue
                if prefix != None:
                    key = prefix+key
                self.__dict__[key] = eval(conf_opt[key])
        
        
        assert self.lstm_cell != None
#        self.lstm_cell = int(self.lstm_cell)
#        if self.use_peepholes == 'False':
#            self.use_peepholes = False
#        else:
#            self.use_peepholes = True

#        self.cell_clip = float(self.cell_clip)
#        if self.num_proj != None:
#            self.num_proj = int(self.num_proj)
#        if self.proj_clip != None:
#            self.proj_clip = float(self.proj_clip)
#        self.forget_bias = float(self.forget_bias)
#        if self.state_is_tuple == 'False':
#            self.state_is_tuple = False
#        else:
#            self.state_is_tuple = True
#        self.keep_prob = float(self.keep_prob)



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

class BLstmLayer(object):
    '''
    layer_flag = BLstmLayer; name = blstmlayer1; fw_lstm_cell = 1024; fw_use_peepholes = True; fw_cell_clip = 5.0; fw_num_proj = 512; fw_proj_clip = 1.0; fw_forget_bias = 0.0; fw_keep_prob = 1.0; bw_lstm_cell = 1024; bw_use_peepholes = True; bw_cell_clip = 5.0; bw_num_proj = 512; bw_proj_clip = 1.0; bw_forget_bias = 0.0; bw_keep_prob = 1.0; state_is_tuple = True; dtype = tf.float32; initializer = tf.contrib.layers.xavier_initializer(tf.float32); activation = None; dtype = tf.float32; reuse = tf.get_variable_scope().reuse;
    '''
    def __init__(self, conf_opt):
        self.conf = conf_opt

    def __cell__(self):
        lstm_fw_cell = LstmLayer(self.conf_fw, prefix = 'fw_')
        lstm_bw_cell = LstmLayer(self.conf_bw, prefix = 'bw_')
        return (lstm_fw_cell, lstm_bw_cell)

    def GetOutputDim(self):
        if self.conf_fw['num_proj'] == None:
            return self.conf['fw_lstm_cell'] + self.conf['bw_lstm_cell']
        else:
            return self.conf['fw_num_proj'] + self.conf['bw_num_proj']

# enum layer 
g_layer_dict={'AffineTransformLayer':AffineTransformLayer,
        'LstmLayer':LstmLayer,
        'BLstmLayer':BLstmLayer
        }
