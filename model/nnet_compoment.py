from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import sys
import os


strset=('name', 'padding', 'data_format')
class NormalizeLayer(object):
    '''
    '''
    def __init__(self, conf_opt):
        self.name = 'NormalizeLayer'
        self.input_dim = 0
        self.target_rms = 1.0
        self.axis = -1
        self.epsilon = 1.3552527156068805425e-20 # 2^-66
        self.dtype = tf.float32
        for key in self.__dict__:
            if key in conf_opt.keys():
                if key in strset:
                    self.__dict__[key] = conf_opt[key]
                else:
                    self.__dict__[key] = eval(conf_opt[key])

        self.scale = self.target_rms * self.target_rms * self.input_dim
        self.scale = pow(self.scale, 1/2)

    def __call__(self, input_feats):
        with tf.name_scope(self.name, "Normalize", [input_feats]) as name:
            #axis = deprecated_argument_lookup("axis", self.axis, "dim", None)
            input_feats = tf.convert_to_tensor(input_feats, dtype=self.dtype, name="input_feats")
            square_sum = tf.reduce_sum(
                    tf.square(input_feats), self.axis, keepdims=True)
            x_inv_norm = tf.rsqrt(tf.maximum(square_sum, self.epsilon))
            x_inv_norm = tf.multiply(x_inv_norm, self.scale)

            return tf.multiply(input_feats, x_inv_norm, name=name)

    def GetOutputDim(self):
        return self.input_dim

class ReluLayer(object):
    '''
    '''
    def __init__(self, conf_opt):
        self.name = 'ReluLayer'
        self.input_dim = 0
        self.output_dim = 0
        for key in self.__dict__:
            if key in conf_opt.keys():
                if key in strset:
                    self.__dict__[key] = conf_opt[key]
                else:
                    self.__dict__[key] = eval(conf_opt[key])

        assert self.input_dim == self.output_dim

    def __call__(self, input_feats):
        return tf.nn.relu(input_feats, self.name)
    
    def GetOutputDim(self):
        return self.output_dim

class SpliceLayer(object):
    '''
    '''
    def __init__(self, conf_opt):
        self.splice = [-1,0,1]
        self.input_dim = 0
        self.name = 'SpliceLayer'
        self.time_major = True
        self.splice_padding = False
        for key in self.__dict__:
            if key in conf_opt.keys():
                if key in strset:
                    self.__dict__[key] = conf_opt[key]
                else:
                    self.__dict__[key] = eval(conf_opt[key])
        assert self.splice[0] <= 0
        assert self.splice[-1] >= 0
        if self.time_major is True:
            self.time = 0
        else:
            self.time = 1

    #t1 = [[1, 2, 3], [4, 5, 6]]
    #t2 = [[7, 8, 9], [10, 11, 12]]
    #tf.concat([t1, t2], 0)  # [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
    #tf.concat([t1, t2], 1)  # [[1, 2, 3, 7, 8, 9], [4, 5, 6, 10, 11, 12]]
    # padding
    def SpliceFeats(self, input_feats):
        output = None
        start_nframe = 0
        end_nframe = 0
        for i in self.splice:
            if i < 0:
                start = input_feats[:1]
                start_end = input_feats[:i]
                while i < 0:
                    start_end = tf.concat([start, start_end], self.time)
                    i += 1
                concat_input = start_end
                start_nframe += 1
            elif i == 0:
                concat_input = input_feats
            elif i > 0:
                end = input_feats[-1:]
                end_start = input_feats[i:]
                while i > 0:
                    end_start = tf.concat([end_start, end], self.time)
                    i -= 1
                concat_input = end_start
                end_nframe += 1

            if output is None:
                output = concat_input
            else:
                output = tf.concat([output, concat_input], -1)

        #first = input_feats[:1]
        #end = input_feats[-1:]
        #f = input_feats[:-1]
        #b = input_feats[1:]
        #fi = tf.concat([first, f],0)
        #bi = tf.concat([b, end], 0)
        #out = tf.concat([fi, input_feats, bi],-1)
        return output
        #return output[start_nframe:-end_nframe]

    # b=tf.pad(input_feats,[[1,1],[0,0],[0,0]],"SYMMETRIC")
    def __call__(self, input_feats):
        '''
        in advance at head and tail add frames, 
        so extract need features
        '''
        # no padding
        if self.splice_padding is False:
            output = None
            start_nframe = -1 * self.splice[0]
            end_nframe = self.splice[-1]
            for i in self.splice:
                #if i < 0:
                start = start_nframe + i
                end = end_nframe - i
                if end == 0:
                    concat_input = input_feats[start:]
                else:
                    concat_input = input_feats[start:-1*end]
                
                #concat_input = input_feats[start:-1*end]
                #elif i == 0:
                #    concat_input = input_feats[start_nframe:-1*end_nframe]
                #elif i > 0:
                #    start = start_nframe + i
                #    end = end_nframe-i
                
                if output is None:
                    output = concat_input
                else:
                    output = tf.concat([output, concat_input], -1)
            return output

        else:
            # padding
            return self.SpliceFeats(input_feats)

    def GetOutputDim(self):
        return len(self.splice) * self.input_dim


class AffineTransformLayer(object):
    '''
    '''
    def __init__(self, conf_opt, dtype = tf.float32, 
            initializer = tf.contrib.layers.xavier_initializer(tf.float32),
            trainable = True,
            name = 'AffineTransformLayer'):
        self.name = name
        self.input_dim = None
        self.output_dim = None
        self.dtype = tf.float32
        self.initializer = tf.contrib.layers.xavier_initializer(tf.float32)
        self.trainable = True
        for key in self.__dict__:
            if key in conf_opt.keys():
                if key in strset:
                    self.__dict__[key] = conf_opt[key]
                else:
                    self.__dict__[key] = eval(conf_opt[key])
        if name != 'AffineTransformLayer':
            self.name = name
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
                    dtype = self.dtype,
                    initializer = self.initializer,
                    trainable = self.trainable)

    def __call__(self, input_feats):
        input_feats = tf.reshape(input_feats, [-1, self.input_dim])
        return tf.matmul(input_feats, self.weights) + self.bias

    def GetInputDim(self):
        return self.input_dim

    def GetOutputDim(self):
        return self.output_dim

class Affine2TransformLayer(object):
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
                    self.__dict__[key] = conf_opt[key]
                else:
                    self.__dict__[key] = eval(conf_opt[key])
        assert self.input_dim != None
        assert self.output_dim != None
        self.affine1 = AffineTransformLayer(conf_opt, name=self.name+'_1')
        self.affine2 = AffineTransformLayer(conf_opt, name=self.name+'_2')
        
    def __call__(self, input_feats):
        input_feats = tf.reshape(input_feats, [-1, self.input_dim])
        output1 = self.affine1(input_feats)
        output2 = self.affine2(input_feats)
        return [output1, output2]

    def GetInputDim(self):
        return self.input_dim

    def GetOutputDim(self):
        return self.output_dim

class TdnnLayer(object):
    '''
    '''
    def __init__(self, conf_opt):
        self.name = 'TdnnLayer'
        self.input_dim = 0
        self.output_dim = 0
        self.time_major = True
        for key in self.__dict__:
            if key in conf_opt.keys():
                if key in strset:
                    self.__dict__[key] = conf_opt[key]
                else:
                    self.__dict__[key] = eval(conf_opt[key])

        self.layers = []
        # SpliceLayer
        conf_opt['name'] = self.name + 'splice'
        self.layers.append(SpliceLayer(conf_opt))
        # AffineTransformLayer
        conf_opt['input_dim'] = str(self.layers[-1].GetOutputDim())
        conf_opt['name'] = self.name + 'affine'
        self.layers.append(AffineTransformLayer(conf_opt))
        # ReluLayer
        conf_opt['input_dim'] = str(self.layers[-1].GetOutputDim())
        conf_opt['name'] = self.name + 'relu'
        self.layers.append(ReluLayer(conf_opt))
        # NormalizeLayer
        conf_opt['name'] = self.name + 'normalize'
        self.layers.append(NormalizeLayer(conf_opt))

    def __call__(self, input_feats):
        output = [input_feats]
        for layer in self.layers:
            if type(layer) is ReluLayer:
                output.append(tf.nn.relu(output[-1]))
                #layer(output[-1])
            else:
                output.append(layer(output[-1]))
        return output[-1]

    def GetOutputDim(self):
        return self.output_dim

    def GetInputDim(self):
        return self.input_dim

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
            if prefix != None:
                opt_key = prefix+key
            else:
                opt_key = key
            if opt_key in conf_opt.keys():
                if key in strset:
                    self.__dict__[key] = conf_opt[opt_key]
                else:
                    self.__dict__[key] = eval(conf_opt[opt_key])
        
        
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
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob = self.keep_prob)
        return cell

    def GetOutputDim(self):
        if self.num_proj == None:
            return self.lstm_cell
        else:
            return self.num_proj

    def Name(self):
        return self.name

class LcBLstmLayer(object):
    '''
    layer_flag = BLstmLayer; name = lcablstmlayer1; fw_lstm_cell = 1024; fw_use_peepholes = True; fw_cell_clip = 5.0; fw_num_proj = 512; fw_proj_clip = 1.0; fw_forget_bias = 0.0; fw_keep_prob = 1.0; bw_lstm_cell = 1024; bw_use_peepholes = True; bw_cell_clip = 5.0; bw_num_proj = 512; bw_proj_clip = 1.0; bw_forget_bias = 0.0; bw_keep_prob = 1.0; latency_controlled=None; state_is_tuple = True; dtype = tf.float32; initializer = tf.contrib.layers.xavier_initializer(tf.float32); activation = None; dtype = tf.float32; reuse = tf.get_variable_scope().reuse;
    '''
    def __init__(self, conf_opt):
        self.conf = conf_opt

    def __call__(self):
        lstm_fw_cell = LstmLayer(self.conf, prefix = 'fw_')
        lstm_bw_cell = LstmLayer(self.conf, prefix = 'bw_')
        return (lstm_fw_cell(), lstm_bw_cell())

    def GetOutputDim(self):
        if self.conf['fw_num_proj'] == None:
            return int(self.conf['fw_lstm_cell']) + int(self.conf['bw_lstm_cell'])
        else:
            return int(self.conf['fw_num_proj']) + int(self.conf['bw_num_proj'])

    def Name(self):
        return self.conf['name']

    def GetLatencyControlled(self):
        return eval(self.conf['latency_controlled'])


class BLstmLayer(object):
    '''
    layer_flag = BLstmLayer; name = blstmlayer1; fw_lstm_cell = 1024; fw_use_peepholes = True; fw_cell_clip = 5.0; fw_num_proj = 512; fw_proj_clip = 1.0; fw_forget_bias = 0.0; fw_keep_prob = 1.0; bw_lstm_cell = 1024; bw_use_peepholes = True; bw_cell_clip = 5.0; bw_num_proj = 512; bw_proj_clip = 1.0; bw_forget_bias = 0.0; bw_keep_prob = 1.0; state_is_tuple = True; dtype = tf.float32; initializer = tf.contrib.layers.xavier_initializer(tf.float32); activation = None; dtype = tf.float32; reuse = tf.get_variable_scope().reuse;
    '''
    def __init__(self, conf_opt):
        self.conf = conf_opt

    def __call__(self):
        lstm_fw_cell = LstmLayer(self.conf, prefix = 'fw_')
        lstm_bw_cell = LstmLayer(self.conf, prefix = 'bw_')
        return (lstm_fw_cell(), lstm_bw_cell())

    def GetOutputDim(self):
        if self.conf['fw_num_proj'] == None:
            return int(self.conf['fw_lstm_cell']) + int(self.conf['bw_lstm_cell'])
        else:
            return int(self.conf['fw_num_proj']) + int(self.conf['bw_num_proj'])

    def Name(self):
        return self.conf['name']

class Cnn2d(object):
    def __init__(self, conf_opt):
        self.height = 11
        self.width = 71
        self.in_channels = 1
        self.filter_height = 11
        self.filter_width = 9
        self.out_channels = 128
        self.height_strides = 1
        self.width_strides = 2
        self.padding = 'VALID'
        self.data_format = 'NHWC'
        self.dilations = [1, 1, 1, 1]
        self.name = 'cnn2d'
        self.initializer = tf.contrib.layers.xavier_initializer(tf.float32)
        self.dtype = tf.float32
        self.trainable = True

        for key in self.__dict__:
            if key in conf_opt.keys():
                if key in strset:
                    self.__dict__[key] = conf_opt[key]
                else:
                    self.__dict__[key] = eval(conf_opt[key])
        
        self.cnn_filter = [self.filter_height, self.filter_width, self.in_channels, self.out_channels]
        self.strides = [1, self.height_strides, self.width_strides, 1]
        
        with tf.variable_scope(self.name):
            self.kernel = tf.get_variable(self.name+'_w',
                    shape = self.cnn_filter,
                    dtype = self.dtype,
                    initializer = self.initializer,
                    trainable = self.trainable)

    def __call__(self, input_feats):
        return tf.nn.conv2d(input_feats, self.kernel, 
                strides = self.strides, padding = self.padding, 
                use_cudnn_on_gpu=True,
                data_format=self.data_format, dilations = self.dilations, name = self.name)

    def GetOutputDim(self):
        return [int((self.height - self.filter_height + 1)/self.height_strides), int((self.width - self.filter_width + 1)/self.width_strides), self.out_channels]

class MaxPool2d(object):
    '''
    inputdim is [height, width, channels]
    '''
    def __init__(self, conf_opt):
        self.pool_height = 1
        self.pool_width = 4
        self.height_strides = 1
        self.width_strides = 4
        self.padding = 'valid'
        self.data_format = 'channels_last'
        self.name = 'maxpool2d'
        self.inputheight = 1
        self.inputwidth = 32
        self.inputchannels = 128

        for key in self.__dict__:
            if key in conf_opt.keys():
                if key in strset:
                    self.__dict__[key] = conf_opt[key]
                else:
                    self.__dict__[key] = eval(conf_opt[key])
        self.inputdim = [self.inputheight, self.inputwidth, self.inputchannels]
        self.pool_size = [self.pool_height, self.pool_width]
        self.strides = [self.height_strides, self.width_strides]
        return
    # input_feats it's [batch, height, width, channels]
    def __call__(self, input_feats):
        return tf.layers.max_pooling2d(input_feats, pool_size = self.pool_size,
                strides = self.strides,
                padding = self.padding,
                data_format = self.data_format,
                name = self.name)

    def GetOutputDim(self):
        outputheight = int((self.inputheight - self.pool_height)/self.height_strides + 1)
        outputwidth = int((self.inputwidth - self.pool_width)/self.width_strides + 1)
        outputchannels = self.inputchannels
        return [outputheight, outputwidth, outputchannels]


# enum layer 
g_layer_dict={'NormalizeLayer':NormalizeLayer,
        'ReluLayer':ReluLayer,
        'SpliceLayer':SpliceLayer,
        'AffineTransformLayer':AffineTransformLayer,
        'Affine2TransformLayer':Affine2TransformLayer,
        'TdnnLayer':TdnnLayer,
        'LstmLayer':LstmLayer,
        'LcBLstmLayer':LcBLstmLayer,
        'BLstmLayer':BLstmLayer,
        'Cnn2d':Cnn2d,
        'MaxPool2d':MaxPool2d
        }

