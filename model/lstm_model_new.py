from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
import sys
import os

try:
    from tensorflow_py_api import mmi,mpe
except ImportError:
    print("no mmi module")

try:
    from tf_chain_py_api import chainloss
except ImportError:
    print("no chainloss module")


from model.nnet_base import NnetBase
import model.nnet_compoment as nnet_compoment
import model.lc_blstm_rnn as lc_blstm_rnn

class LstmModel(NnetBase):
    '''
    Lstm model.
    '''
    def __init__(self, conf_dict):
        self.num_layers = 0
        self.batch_size_cf = 10
        self.num_frames_batch_cf = 20
#        self.learn_rate_cf = 1e-4
#        self.output_keep_prob_cf = 1.0
        self.time_major_cf = True
        self.state_is_tuple_cf = True
        self.nnet_conf_cf = None
        # task_index must be uniqueness
        self.task_index_cf = None
        self.lc = None

        # Initial configuration parameter.
        for attr in self.__dict__:
            if len(attr.split('_cf')) != 2:
                continue;
            key = attr.split('_cf')[0]
            if key in conf_dict.keys():
                self.__dict__[attr] = conf_dict[key]

        # Initial nnet parameter
        self.nnet_conf_opt = NnetBase().ReadNnetConf(self.nnet_conf_cf)

        self.cnn_conf_opt = None
        self.rnn_conf_opt = None
        
    def PrevLayerIs(self, layers, nntype):
        if len(layers) != 0:
            if layers[-1][0] == nntype:
                return True
        return False

    def LayerIs(self, layers, nntype, n):
        if len(layers) > n:
            if layers[n][0] == nntype:
                return True
        return False
    
    def CreateModelGraph(self, nnet_conf = None):
        layers = []
        if nnet_conf == None:
            nnet_conf = self.nnet_conf_opt
        # analysis config and construct nnet graph
        for layer_opt in nnet_conf:
            if layer_opt['layer_flag'] == 'AffineTransformLayer':
                if self.PrevLayerIs(layers, 'AffineTransformLayer'):
                    layers[-1].append(nnet_compoment.AffineTransformLayer(layer_opt))
                else:
                    layers.append(['AffineTransformLayer', 
                        nnet_compoment.AffineTransformLayer(layer_opt)])
            elif layer_opt['layer_flag'] == 'LstmLayer':
                if self.PrevLayerIs(layers, 'LstmLayer'):
                    layers[-1].append(nnet_compoment.LstmLayer(layer_opt))
                else:
                    layers.append(['LstmLayer', 
                        nnet_compoment.LstmLayer(layer_opt)])
            elif layer_opt['layer_flag'] == 'Sigmoid':
                if self.PrevLayerIs(layers, 'Sigmoid'):
                    layers[-1].append(tf.nn.sigmoid)
                else:
                    layers.append(['Sigmoid',tf.nn.sigmoid])
            elif layer_opt['layer_flag'] == 'BLstmLayer':
                if self.PrevLayerIs(layers, 'BLstmLayer'):
                    layers[-1].append(nnet_compoment.BLstmLayer(layer_opt))
                else:
                    layers.append(['BLstmLayer' ,
                        nnet_compoment.BLstmLayer(layer_opt)])

            elif layer_opt['layer_flag'] == 'LcBLstmLayer':
                if self.PrevLayerIs(layers, 'LcBLstmLayer'):
                    layers[-1].append(nnet_compoment.LcBLstmLayer(layer_opt))
                else:
                    layers.append(['LcBLstmLayer' ,
                        nnet_compoment.LcBLstmLayer(layer_opt)])

            elif layer_opt['layer_flag'] == 'Cnn2d':
                if self.PrevLayerIs(layers, 'Cnn2d'):
                    layers[-1].append(nnet_compoment.Cnn2d(layer_opt))
                else:
                    layers.append(['Cnn2d' ,
                        nnet_compoment.Cnn2d(layer_opt)])
            elif layer_opt['layer_flag'] == 'MaxPool2d':
                if self.PrevLayerIs(layers, 'MaxPool2d'):
                    layers[-1].append(nnet_compoment.MaxPool2d(layer_opt))
                else:
                    layers.append(['MaxPool2d',
                        nnet_compoment.MaxPool2d(layer_opt)])
            elif layer_opt['layer_flag'] == 'SpliceLayer':
                if self.PrevLayerIs(layers, 'SpliceLayer'):
                    layers[-1].append(nnet_compoment.SpliceLayer(layer_opt))
                else:
                    layers.append(['SpliceLayer',
                        nnet_compoment.SpliceLayer(layer_opt)])
            elif layer_opt['layer_flag'] == 'NormalizeLayer':
                if self.PrevLayerIs(layers, 'NormalizeLayer'):
                    layers[-1].append(nnet_compoment.NormalizeLayer(layer_opt))
                else:
                    layers.append(['NormalizeLayer',
                        nnet_compoment.NormalizeLayer(layer_opt)])
            elif layer_opt['layer_flag'] == 'ReluLayer':
                if self.PrevLayerIs(layers, 'ReluLayer'):
                    layers[-1].append(nnet_compoment.ReluLayer(layer_opt))
                else:
                    layers.append(['ReluLayer',
                        nnet_compoment.ReluLayer(layer_opt)])
            elif layer_opt['layer_flag'] == 'TdnnLayer':
                if self.PrevLayerIs(layers, 'TdnnLayer'):
                    layers[-1].append(nnet_compoment.TdnnLayer(layer_opt))
                else:
                    layers.append(['TdnnLayer',
                        nnet_compoment.TdnnLayer(layer_opt)])
            else:
                logging.info('No this layer '+ layer_opt['layer_flag'] + '...')
                assert 'no this layer' and False
        return layers

    def KeepLstmHiddenState(self, tuple_state, new_states):
        '''Define an op to keep the hidden state between batches'''
        update_ops = []
        for state_variable, new_state in zip(tuple_state, new_states):
            # Assign the new state to the state variables on this layer
            update_ops.extend([state_variable[0].assign(new_state[0]),
                state_variable[1].assign(new_state[1])])
        # Return a tuple in order to combine all update_ops into a single operation.
        # The tuple's actual value should not be used.
        rnn_keep_state_op = tf.tuple(update_ops)
        return rnn_keep_state_op

    def ResetLstmHiddenState(self, tuple_state):
        '''Define an op to reset the hidden state to zeros'''
        update_ops = []
        for state_variable in tuple_state:
            # Assign the new state to the state variables on this layer
            update_ops.extend([state_variable[0].assign(tf.zeros_like(state_variable[0])),
                state_variable[1].assign(tf.zeros_like(state_variable[1]))])

        # Return a tuple in order to combine all update_ops into a single operation.
        # The tuple's actual value should not be used.
        rnn_state_zero_op = tf.tuple(update_ops)
        return rnn_state_zero_op                                                                                                                        
    def CreateCnnModel(self, input_feats):
        layers = self.CreateModelGraph()
        outputs = [input_feats]
        shape = np.shape(input_feats)
        # [hight, weight, channels]
        output_dim = [shape[1], shape[2], shape[3]]
        for layer in layers:
            if layer[0] == 'Cnn2d':
                for cnn in layer[1:]:
                    outputs.append(cnn(outputs[-1]))
                output_dim = layer[-1].GetOutputDim()
            elif layer[0] == 'MaxPool2d':
                for maxpool in layer[1:]:
                    outputs.append(maxpool(outputs[-1]))
                output_dim = layer[-1].GetOutputDim()
            else:
                break
                logging.info('No this layer '+ layer[0] + '...')
                assert 'no this layer' and False
        
        output_dim = output_dim[0] * output_dim[1] * output_dim[2]
        if self.time_major_cf:
            outputs[-1] = tf.reshape(outputs[-1],
                    [-1, self.batch_size_cf, output_dim])
        else:
            outputs[-1] = tf.reshape(outputs[-1],
                    [self.batch_size_cf, -1, output_dim])

        self.output_size = output_dim
        return outputs[-1]


    def CreateModel(self, input_feats, seq_len):
        '''
        feature_shape it's for cnn feature reshape,it;it splice info for example [11, 40],
        11 it's frame, 40 it's dim.
        '''
        layers = self.CreateModelGraph()
        outputs = [input_feats]
        output_dim = np.shape(input_feats)[-1]
        rnn_keep_state_op = []
        rnn_state_zero_op = []
        nlayer = len(layers)
        n = 0
        while n < nlayer:
#        for layer in layers:
            layer = layers[n]
            n+=1
            # add Cnn2d
            if layer[0] == 'Cnn2d':
                shape = np.shape(input_feats)
                output_dim = [shape[1], shape[2], shape[3]]
                for cnn in layer[1:]:
                    outputs.append(cnn(outputs[-1]))
                output_dim = layer[-1].GetOutputDim()
                if not self.LayerIs(layers, 'MaxPool2d', n):
                    assert 'Cnn2d next layer it\'s MaxPool2d' and False
            # add MaxPool2d
            elif layer[0] == 'MaxPool2d':
                for maxpool in layer[1:]:
                    outputs.append(maxpool(outputs[-1]))
                output_dim = layer[-1].GetOutputDim()
                # judge next layer it's cnn or other
                if self.LayerIs(layers, 'Cnn2d', n):
                    continue
                else:
                    # exchange dim
                    output_dim = output_dim[0] * output_dim[1] * output_dim[2]
                    if self.time_major_cf:
                        outputs[-1] = tf.reshape(outputs[-1],
                                [-1, self.batch_size_cf, output_dim])
                    else:
                        outputs[-1] = tf.reshape(outputs[-1],
                                [self.batch_size_cf, -1, output_dim])
            # add AffineTransformLayer
            elif layer[0] == 'AffineTransformLayer':
                assert output_dim == layer[1].GetInputDim()
                for mlp in layer[1:]:
                    outputs.append(mlp(outputs[-1]))
                output_dim = layer[-1].GetOutputDim()
            # add SpliceLayer
            elif layer[0] == 'SpliceLayer':
                for splice in layer[1:]:
                    outputs.append(splice(outputs[-1]))
                output_dim = layer[-1].GetOutputDim()
            # add sigmoid
            elif layer[0] == 'Sigmoid':
                for sig in layer[1:]:
                    outputs.append(sig(outputs[-1]))
            # add ReluLayer
            elif layer[0] == 'ReluLayer':
                for relu in layer[1:]:
                    tf.nn.relu(outputs[-1])
                    #relu(outputs[-1])
            # add NormalizeLayer
            elif layer[0] == 'NormalizeLayer':
                for norm in layer[1:]:
                    outputs.append(norm(outputs[-1]))
            # add TdnnLayer
            elif layer[0] == 'TdnnLayer':
                for tdnn in layer[1:]:
                    tdnninput_dim = tdnn.GetInputDim()
                    if self.time_major_cf:
                        outputs[-1] = tf.reshape(outputs[-1],
                                [-1, self.batch_size_cf, tdnninput_dim])
                    else:
                        outputs[-1] = tf.reshape(outputs[-1],
                                [self.batch_size_cf, -1, tdnninput_dim])
                    outputs.append(tdnn(outputs[-1]))
                output_dim = layer[-1].GetOutputDim()
            # add LstmLayer
            elif layer[0] == 'LstmLayer':
                lstm_layer = []
                name = layer[1].Name()
                for lstm_nn in layer[1:]:
                    lstm_layer.append(lstm_nn())
                rnn_cells = tf.contrib.rnn.MultiRNNCell(lstm_layer,
                        state_is_tuple=self.state_is_tuple_cf)
                # Define some variables to store the RNN state
                # Note : tensorflow keep the state inside a batch but it's necessary to do this in order to keep the state
                #        between batches, especially when doing live transcript
                #        Another way would have been to get the state as an output of the session and feed it every time but
                #        this way is much more efficient
                with tf.variable_scope('LstmHidden_state' + name + '_' + str(self.task_index_cf), reuse=False):
                    state_variables = []
                    for state_c, state_h in rnn_cells.zero_state(self.batch_size_cf,
                            tf.float32):
                        state_variables.append(tf.contrib.rnn.LSTMStateTuple(
                            tf.Variable(state_c, trainable=False,  validate_shape=False, collections=[tf.GraphKeys.LOCAL_VARIABLES]),
                            tf.Variable(state_h, trainable=False,  validate_shape=False, collections=[tf.GraphKeys.LOCAL_VARIABLES])))
                    rnn_tuple_state = tuple(state_variables)
                # Build the RNN
                # time is major
                if self.time_major_cf:
                    outputs[-1] = tf.reshape(outputs[-1],
                            [-1, self.batch_size_cf, output_dim])
                else:
                    outputs[-1] = tf.reshape(outputs[-1],
                            [self.batch_size_cf, -1, output_dim])
                
                with tf.name_scope("LSTM" + name):
                    rnn_outputs, new_states = tf.nn.dynamic_rnn(cell=rnn_cells,
                            inputs = outputs[-1],
                            sequence_length=seq_len,
                            initial_state=rnn_tuple_state,
                            dtype=tf.float32,
                            time_major=self.time_major_cf,
                            scope = 'LSTM' + name)

                rnn_keep_state_op.append(self.KeepLstmHiddenState(rnn_tuple_state, new_states))
                
                rnn_state_zero_op.append(self.ResetLstmHiddenState(rnn_tuple_state))
                '''# Define an op to keep the hidden state between batches
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
                rnn_state_zero_op = tf.tuple(update_ops)'''
                if not self.time_major_cf:
                    rnn_outputs = tf.transpose(rnn_outputs, [1, 0, 2]) # [time, batch_size, cell_outdim]

                outputs.append(rnn_outputs)
                output_dim = layer[-1].GetOutputDim()
            # add BLstmLayer
            elif layer[0] == 'BLstmLayer':
                fw_lstm_layer = []
                bw_lstm_layer = []
                name = layer[1].Name()
                for blstm_l in layer[1:]:
                    blstm_nn = blstm_l()
                    fw_lstm_layer.append(blstm_nn[0])
                    bw_lstm_layer.append(blstm_nn[1])
                #fw_rnn_cells = tf.contrib.rnn.MultiRNNCell(fw_lstm_layer,
                #        state_is_tuple=self.state_is_tuple_cf)
                #bw_rnn_cells = tf.contrib.rnn.MultiRNNCell(bw_lstm_layer,
                #        state_is_tuple=self.state_is_tuple_cf)

                # Define some variables to store the RNN state
                # Note : tensorflow keep the state inside a batch but it's necessary to do this in order to keep the state
                #        between batches, especially when doing live transcript
                #        Another way would have been to get the state as an output of the session and feed it every time but
                #        this way is much more efficient
                with tf.variable_scope('Blstm_FwHidden_state' + name + '_' + str(self.task_index_cf), reuse=False):
                    state_variables = []
                    for f_lstm in fw_lstm_layer:
                        state_c, state_h = f_lstm.zero_state(self.batch_size_cf, tf.float32)
                        state_variables.append(tf.contrib.rnn.LSTMStateTuple(
                            tf.Variable(state_c, name=name, trainable=False, validate_shape=False, collections=[tf.GraphKeys.LOCAL_VARIABLES]),
                            tf.Variable(state_h, name=name, trainable=False, validate_shape=False, collections=[tf.GraphKeys.LOCAL_VARIABLES])))
                    fw_rnn_tuple_state = state_variables
                    #fw_rnn_tuple_state = tuple(state_variables)

                with tf.variable_scope('Blstm_BwHidden_state' + name + '_' + str(self.task_index_cf), reuse=False):
                    state_variables = []
                    for b_lstm in bw_lstm_layer:
                        state_c, state_h = b_lstm.zero_state(self.batch_size_cf, tf.float32)
                        state_variables.append(tf.contrib.rnn.LSTMStateTuple(
                            tf.Variable(state_c, name = name, trainable=False, validate_shape=False, collections=[tf.GraphKeys.LOCAL_VARIABLES]),
                            tf.Variable(state_h, name = name, trainable=False, validate_shape=False, collections=[tf.GraphKeys.LOCAL_VARIABLES])))
                    bw_rnn_tuple_state = state_variables
                    #bw_rnn_tuple_state = tuple(state_variables)

                # Build the RNN
                # time is major
                if self.time_major_cf:
                    outputs[-1] = tf.reshape(outputs[-1],
                            [-1, self.batch_size_cf, output_dim])
                else:
                    outputs[-1] = tf.reshape(outputs[-1],
                            [self.batch_size_cf, -1, output_dim])

                with tf.name_scope("BLSTM" + name):
                    brnn_outputs, output_states_fw, output_states_bw = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
                            cells_fw = fw_lstm_layer,
                            cells_bw = bw_lstm_layer,
                            inputs = outputs[-1],
                            initial_states_fw = fw_rnn_tuple_state,
                            initial_states_bw = bw_rnn_tuple_state,
                            dtype = tf.float32,
                            sequence_length = seq_len,
                            parallel_iterations = None,
                            time_major = self.time_major_cf,
                            scope = 'stack_bidirectional_rnn' + name)

                fw_rnn_keep_state_op = self.KeepLstmHiddenState(fw_rnn_tuple_state, output_states_fw)
                bw_rnn_keep_state_op = self.KeepLstmHiddenState(bw_rnn_tuple_state, output_states_bw)
                
                fw_rnn_state_zero_op = self.ResetLstmHiddenState(fw_rnn_tuple_state)
                bw_rnn_state_zero_op = self.ResetLstmHiddenState(bw_rnn_tuple_state)
                rnn_keep_state_op.append((fw_rnn_keep_state_op, bw_rnn_keep_state_op))
                rnn_state_zero_op.append((fw_rnn_state_zero_op, bw_rnn_state_zero_op))
                if not self.time_major_cf:
                    brnn_outputs = tf.transpose(brnn_outputs, [1, 0, 2]) # [time, batch_size, cell_outdim]

                outputs.append(brnn_outputs)
                output_dim = layer[-1].GetOutputDim()
            elif layer[0] == 'LcBLstmLayer':
                fw_lstm_layer = []
                bw_lstm_layer = []
                name = layer[1].Name()
                self.lc = layer[1].GetLatencyControlled()
                for blstm_l in layer[1:]:
                    blstm_nn = blstm_l()
                    fw_lstm_layer.append(blstm_nn[0])
                    bw_lstm_layer.append(blstm_nn[1])
                #fw_rnn_cells = tf.contrib.rnn.MultiRNNCell(fw_lstm_layer,
                #        state_is_tuple=self.state_is_tuple_cf)
                #bw_rnn_cells = tf.contrib.rnn.MultiRNNCell(bw_lstm_layer,
                #        state_is_tuple=self.state_is_tuple_cf)

                # Define some variables to store the RNN state
                # Note : tensorflow keep the state inside a batch but it's necessary to do this in order to keep the state
                #        between batches, especially when doing live transcript
                #        Another way would have been to get the state as an output of the session and feed it every time but
                #        this way is much more efficient
                with tf.variable_scope('LcBlstm_FwHidden_state' + name + '_' + str(self.task_index_cf), reuse=False):
                    state_variables = []
                    for f_lstm in fw_lstm_layer:
                        state_c, state_h = f_lstm.zero_state(self.batch_size_cf, tf.float32)
                        state_variables.append(tf.contrib.rnn.LSTMStateTuple(
                            tf.Variable(state_c, name=name, trainable=False, validate_shape=False, collections=[tf.GraphKeys.LOCAL_VARIABLES]),
                            tf.Variable(state_h, name=name, trainable=False, validate_shape=False, collections=[tf.GraphKeys.LOCAL_VARIABLES])))
                    fw_rnn_tuple_state = state_variables
                    #fw_rnn_tuple_state = tuple(state_variables)

                with tf.variable_scope('LcBlstm_BwHidden_state' + name + '_' + str(self.task_index_cf), reuse=False):
                    state_variables = []
                    for b_lstm in bw_lstm_layer:
                        state_c, state_h = b_lstm.zero_state(self.batch_size_cf, tf.float32)
                        state_variables.append(tf.contrib.rnn.LSTMStateTuple(
                            tf.Variable(state_c, name = name, trainable=False, validate_shape=False, collections=[tf.GraphKeys.LOCAL_VARIABLES]),
                            tf.Variable(state_h, name = name, trainable=False, validate_shape=False, collections=[tf.GraphKeys.LOCAL_VARIABLES])))
                    bw_rnn_tuple_state = state_variables
                    #bw_rnn_tuple_state = tuple(state_variables)

                # Build the RNN
                # time is major
                if self.time_major_cf:
                    outputs[-1] = tf.reshape(outputs[-1],
                            [-1, self.batch_size_cf, output_dim])
                else:
                    outputs[-1] = tf.reshape(outputs[-1],
                            [self.batch_size_cf, -1, output_dim])

                with tf.name_scope("LCBLSTM" + name):
                    brnn_outputs, output_states_fw, output_states_bw = lc_blstm_rnn.stack_bidirectional_dynamic_rnn(
                            cells_fw = fw_lstm_layer,
                            cells_bw = bw_lstm_layer,
                            inputs = outputs[-1],
                            initial_states_fw = fw_rnn_tuple_state,
                            initial_states_bw = None,
                            latency_controlled=self.lc,
                            dtype = tf.float32,
                            sequence_length = seq_len,
                            parallel_iterations = None,
                            time_major = self.time_major_cf,
                            scope = 'stack_bidirectional_rnn' + name)

                fw_rnn_keep_state_op = self.KeepLstmHiddenState(fw_rnn_tuple_state, output_states_fw)
                bw_rnn_keep_state_op = self.KeepLstmHiddenState(bw_rnn_tuple_state, output_states_bw)
                
                fw_rnn_state_zero_op = self.ResetLstmHiddenState(fw_rnn_tuple_state)
                bw_rnn_state_zero_op = self.ResetLstmHiddenState(bw_rnn_tuple_state)
                rnn_keep_state_op.append((fw_rnn_keep_state_op, bw_rnn_keep_state_op))
                rnn_state_zero_op.append((fw_rnn_state_zero_op, bw_rnn_state_zero_op))
                if not self.time_major_cf:
                    brnn_outputs = tf.transpose(brnn_outputs, [1, 0, 2]) # [time, batch_size, cell_outdim]

                outputs.append(brnn_outputs)
                output_dim = layer[-1].GetOutputDim()
            else:
                continue

        if self.time_major_cf:
            last_output = tf.reshape(outputs[-1],
                    [-1, self.batch_size_cf, output_dim])
        else:
            last_output = tf.reshape(outputs[-1],
                    [self.batch_size_cf, -1, output_dim])

        self.output_size = output_dim
        return last_output, rnn_keep_state_op, rnn_state_zero_op

    def CreateRnnModel(self, input_feats, seq_len):
        rnn_layers = []
        self.other_layer = []
        for layer_opt in self.nnet_conf_opt:
            if layer_opt['layer_flag'] == 'AffineTransformLayer':
                self.other_layer.append(nnet_compoment.AffineTransformLayer(layer_opt))
            elif layer_opt['layer_flag'] == 'LstmLayer':
                rnn_layers.append(nnet_compoment.LstmLayer(layer_opt)())
            elif layer_opt['layer_flag'] == 'Sigmoid':
                self.other_layer.append(tf.nn.sigmoid)

        rnn_cells = tf.contrib.rnn.MultiRNNCell(rnn_layers, 
                state_is_tuple=self.state_is_tuple_cf)
        # Define some variables to store the RNN state
        # Note : tensorflow keep the state inside a batch but it's necessary to do this in order to keep the state
        #        between batches, especially when doing live transcript
        #        Another way would have been to get the state as an output of the session and feed it every time but
        #        this way is much more efficient
        with tf.variable_scope('Hidden_state' + '_' + str(self.task_index_cf), reuse=False):
            state_variables = []
            for state_c, state_h in rnn_cells.zero_state(self.batch_size_cf,
                    tf.float32):
                state_variables.append(tf.contrib.rnn.LSTMStateTuple(
                    tf.Variable(state_c, trainable=False, validate_shape=False, collections=[tf.GraphKeys.LOCAL_VARIABLES]),
                    tf.Variable(state_h, trainable=False, validate_shape=False, collections=[tf.GraphKeys.LOCAL_VARIABLES])))
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
        for layer in self.other_layer:
           output.append(layer(output[-1]))

        # Get output dim
        self.output_dim = self.other_layer[-1].GetOutputDim()
        # last no softmax
        last_output = tf.reshape(output[-1], 
                [-1, self.batch_size_cf, self.output_dim])
        return last_output, rnn_keep_state_op, rnn_state_zero_op

    # define model
    def CeCnnBlstmLoss(self, input_feats, labels, seq_len):
        cnn_output = self.CreateCnnModel(input_feats)
        return self.CeLoss(cnn_output, labels, seq_len)
    #
    #
    def CtcLoss(self, input_feats, labels, seq_len):
        #last_output, rnn_keep_state_op, rnn_state_zero_op = self.CreateRnnModel(
        #        input_feats, seq_len)
        last_output, rnn_keep_state_op, rnn_state_zero_op = self.CreateModel(
                input_feats, seq_len)

        if True:
            decoded, log_prob = tf.nn.ctc_greedy_decoder(last_output,
                    seq_len, merge_repeated=True)
        else:
            decoded, log_prob = tf.nn.ctc_beam_search_decoder(last_output,
                    seq_len, merge_repeated=True)

        # Inaccuracy: label error rate
        label_error_rate = tf.reduce_mean(
                tf.edit_distance(tf.to_int32(decoded[0]),labels))
        with tf.name_scope('CTC'):
            ctc_loss = tf.nn.ctc_loss(labels, last_output, seq_len, 
                    time_major=self.time_major_cf)
            # Compute the mean loss of the batch (only used to check on progression in learning)
            # The loss is averaged accross the batch but before we take into account the real size of the label
            ctc_mean_loss = tf.reduce_mean(tf.truediv(ctc_loss, tf.to_float(seq_len)))
        return ctc_mean_loss, ctc_loss ,label_error_rate, decoded[0]

    def CeLoss(self, input_feats, labels, seq_len):
        #last_output, rnn_keep_state_op, rnn_state_zero_op = self.CreateRnnModel(
        #        input_feats, seq_len)
        last_output, rnn_keep_state_op, rnn_state_zero_op = self.CreateModel(
                input_feats, seq_len)

        # this function labels from time_major = False change major = True
        if self.time_major_cf:
            labels = tf.transpose(labels)

        def true_length(length):
            if length > tf.lc:
                return tf.lc
            else:
                return length
        # lc blstm model , process seq_len
        if self.lc is not None:
            cond = self.lc > seq_len
            lc = seq_len-seq_len+self.lc
            seq_len = tf.where(cond, seq_len, lc)
            #seq_len = np.array([ true_length(now_len) for now_len in seq_len ])
        
        with tf.name_scope('CE'):
            print('********************',labels.shape,input_feats.shape)
            ce_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=last_output, name="ce_loss")
            print('********************',ce_loss.shape)
            #mask = tf.cast(tf.reshape(tf.transpose(tf.sequence_mask(
            #        seq_len, self.num_frames_batch_cf)), [-1]), tf.float32)
            nframes = tf.shape(labels)[0]
            mask = tf.cast(tf.reshape(tf.transpose(tf.sequence_mask(
                    seq_len, nframes)), [-1]), tf.float32)

            total_frames = tf.cast(tf.reduce_sum(seq_len) ,tf.float32)
            ce_mean_loss = tf.reduce_sum(mask * tf.reshape(ce_loss,[-1])) / total_frames
            label_error_rate = self.CalculateLabelErrorRate(last_output, labels, mask, total_frames)

        return ce_mean_loss, ce_loss, label_error_rate , rnn_keep_state_op, rnn_state_zero_op

    def ChainLoss(self, input_feats,
            indexs, in_labels, weights, statesinfo, num_states, frames,
            label_dim,
            den_indexs, den_in_labels, den_weights, den_statesinfo, den_num_states,
            den_start_state = 0 ,delete_laststatesuperfinal = True,
            l2_regularize = 0.0, leaky_hmm_coefficient = 0.0, xent_regularize =0.0):
        seq_len = None
        last_output, rnn_keep_state_op, rnn_state_zero_op = self.CreateModel(
                input_feats, seq_len)
        
        last_output = last_output[-1 * frames[0]:]

        with tf.name_scope('ChainLoss'):
            chain_loss = chainloss(last_output,
                    indexs, in_labels, weights, statesinfo, num_states,
                    label_dim,
                    den_indexs, den_in_labels, den_weights, den_statesinfo, den_num_states,
                    den_start_state, delete_laststatesuperfinal,
                    l2_regularize, leaky_hmm_coefficient, xent_regularize, 
                    time_major = self.time_major_cf)

            total_frames = 0
            chain_mean_loss = chain_loss[0]

        return chain_mean_loss, chain_loss, None, rnn_keep_state_op, rnn_state_zero_op


    #def MmiLoss(self, input_feats, labels, seq_len, lattice, old_acoustic_scale = 0.0, acoustic_scale = 0.083, time_major = True):
    def MmiLoss(self, input_feats, labels, seq_len,
            indexs, pdf_values, lm_ws, am_ws, statesinfo, num_states,
            old_acoustic_scale = 0.0, acoustic_scale = 0.083, 
            time_major = True, drop_frames = True):
        last_output, rnn_keep_state_op, rnn_state_zero_op = self.CreateModel(
                input_feats, seq_len)

        with tf.name_scope('MMI'):
            mmi_loss = mmi(last_output, seq_len, labels, 
                    indexs = indexs,
                    pdf_values = pdf_values,
                    lm_ws = lm_ws,
                    am_ws = am_ws,
                    statesinfo = statesinfo,
                    num_states = num_states,
                    old_acoustic_scale = old_acoustic_scale,
                    acoustic_scale = acoustic_scale,
                    drop_frames = drop_frames,
                    time_major = self.time_major_cf)

            if self.time_major_cf:
                labels = tf.transpose(labels)

            nframes = tf.shape(labels)[0]
            mask = tf.cast(tf.reshape(tf.transpose(tf.sequence_mask(
                seq_len, nframes)), [-1]), tf.float32)

            total_frames = tf.cast(tf.reduce_sum(seq_len) ,tf.float32)
            mmi_mean_loss = tf.reduce_sum(mmi_loss) / total_frames
            label_error_rate = self.CalculateLabelErrorRate(last_output, labels,  mask, total_frames)

        return mmi_mean_loss, mmi_loss, label_error_rate, rnn_keep_state_op, rnn_state_zero_op

    def MpeLoss(self, input_feats, labels, seq_len,
            indexs, pdf_values, lm_ws, am_ws, statesinfo, num_states,
            silence_phones,# = [96],
            pdf_to_phone,
            log_priors = None,
            one_silence_class = True,
            criterion = 'smbr', # or 'mfpe'
            old_acoustic_scale = 0.0, acoustic_scale = 0.083,
            time_major = True):
        last_output, rnn_keep_state_op, rnn_state_zero_op = self.CreateModel(
                input_feats, seq_len)
        
        with tf.name_scope('MPE'):
            if log_priors is not None:
                last_output = tf.subtract(last_output,log_priors)
            mpe_loss = mpe(last_output, seq_len, labels, 
                    indexs = indexs,
                    pdf_values = pdf_values,
                    lm_ws = lm_ws,
                    am_ws = am_ws,
                    statesinfo = statesinfo,
                    num_states = num_states,
                    silence_phones = silence_phones,
                    pdf_to_phone = pdf_to_phone,
                    one_silence_class = one_silence_class,
                    criterion = criterion,
                    old_acoustic_scale = old_acoustic_scale,
                    acoustic_scale = acoustic_scale,
                    time_major = self.time_major_cf)

            if self.time_major_cf:
                labels = tf.transpose(labels)

            nframes = tf.shape(labels)[0]
            mask = tf.cast(tf.reshape(tf.transpose(tf.sequence_mask(
                seq_len, nframes)), [-1]), tf.float32)

            total_frames = tf.cast(tf.reduce_sum(seq_len) ,tf.float32)
            mpe_mean_loss = tf.reduce_sum(mpe_loss) / total_frames
            label_error_rate = self.CalculateLabelErrorRate(last_output, labels,  mask, total_frames)

        return mpe_mean_loss, mpe_loss, label_error_rate, rnn_keep_state_op, rnn_state_zero_op
    
    def CalculateLabelErrorRate(self, output_log, labels, mask, total_frames):
        correct_prediction = tf.equal( 
                tf.argmax(tf.reshape(output_log, [-1,self.output_size]), 1), 
                tf.cast(tf.reshape(labels, [-1]), tf.int64))
        accuracy = tf.reduce_sum(tf.cast(correct_prediction, tf.float32) * mask) / total_frames
        return 1-accuracy


