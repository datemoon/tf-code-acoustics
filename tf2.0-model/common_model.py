from __future__ import absolute_import, division, print_function, unicode_literals
import os, sys, shutil, time
import logging
import tensorflow as tf

import nnet_compoment

class CreateModel(object):
    def __init__(self, nnet_conf):
        self.nnet_conf = nnet_conf
        self.layers_queue = []

        # analysis config and construct nnet graph
        for  layer_flag,layer_opt in self.nnet_conf:
            curr_layer = None
            if layer_flag == 'AffineTransformLayer':
                #AffineTransformLayer
                curr_layer = tf.keras.layers.Dense.from_config(layer_opt)
            elif layer_flag == 'Affine2TransformLayer':
                pass
            elif layer_flag == 'SpliceLayer':
                #SpliceLayer
                curr_layer = nnet_compoment.SpliceLayer.from_config(layer_opt)
            elif layer_flag == 'TdnnLayer':
                pass
            #RNN
            elif layer_flag == 'GRU':
                pass
            elif layer_flag == 'LstmLayer':
                #LSTM
                curr_layer = tf.keras.layers.LSTM.from_config(layer_opt)
            elif layer_flag == 'BLstmLayer':
                #BLSTM
                curr_layer = tf.keras.layers.Bidirectional.from_config(layer_opt)
            elif layer_flag == 'LcBLstmLayer':
                pass 
            #CNN
            elif layer_flag == 'Cnn2d':
                #Cnn2d
                curr_layer = tf.keras.layers.Conv2D.from_config(layer_opt)
            elif layer_flag == 'MaxPool2d':
                #MaxPool2d
                curr_layer = tf.keras.layers.MaxPool2D.from_config(layer_opt)
            # activations
            elif layer_flag == 'Sigmoid':
                #Sigmoid
                curr_layer = nnet_compoment.Sigmoid.form_config(layer_opt)
            elif layer_flag == 'Softmax':
                #Softmax
                curr_layer = tf.keras.layers.Softmax.from_config(layer_opt)
            elif layer_flag == 'NormalizeLayer':
                #NormalizeLayer
                curr_layer = nnet_compoment.NormalizeLayer.from_config(layer_opt)
            elif layer_flag == 'ReluLayer':
                #ReluLayer
                curr_layer = tf.keras.layers.ReLU.from_config(layer_opt)
            else:
                logging.info('No this layer '+ layer_flag + '...')
                assert 'no this layer' and False
            self.layers_queue.append(curr_layer)

    def PrevLayerIs(self, nntype):
        if len(self.layers) != 0:
            if self.layers[-1][0] == nntype:
                return True
        return False
        
    def GetLayers(self):
        return self.layers_queue

    def GetLayersConf(self):
        conf = []
        for layer in self.layers_queue:
            conf.append(self.layers_queue.get_config())
        return conf


class CommonModel(tf.keras.Model):

    def __init__(self, nnet_conf):
        super(CommonModel, self).__init__()
        # analysis config and construct nnet graph
        self.model = CreateModel(nnet_conf)
        # get layer
        self.layers_queue = self.model.GetLayers()

    def call(self, inputs):
        outputs = inputs
        for layer in self.layers_queue:
            outputs = layer(outputs)
            #print(layer)
            #print(outputs.shape)
        return outputs

    def SaveModelWeights(self, weights_name):
        #config = self.get_config()
        #weights = self.get_weights()
        self.save_weights(weights_name)
    
    def ReadLoadWeights(self, weights_name):
        self.load_weights(weights_name)


    @classmethod
    def ReStoreModel(cls, nnet_conf, weights_name):
        model = cls(nnet_conf)
        model.ReadLoadWeights(weights_name)
        return model


        
