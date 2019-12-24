




from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import sys
import os

class SpliceLayer(tf.keras.layers.Layer):
    def __init__(self, splice, time_major=True, splice_padding=False, **kwargs):
        super(SpliceLayer, self).__init__(**kwargs)
        self.splice = splice
        self.time_major = time_major
        self.splice_padding = splice_padding

        if self.time_major is True:
            self.time = 0
        else:
            self.time = 0

    def get_config(self):
        config = {'splice':self.splice, 'time_major':self.time_major, 'splice_padding':self.splice_padding}
        base_config = super(SpliceLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


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
        if self.time_major is False:
            output = tf.transpose(output, [1, 0, 2])
        #first = input_feats[:1]
        #end = input_feats[-1:]
        #f = input_feats[:-1]
        #b = input_feats[1:]
        #fi = tf.concat([first, f],0)
        #bi = tf.concat([b, end], 0)
        #out = tf.concat([fi, input_feats, bi],-1)
        return output
    
    # b=tf.pad(input_feats,[[1,1],[0,0],[0,0]],"SYMMETRIC")
    def call(self, input_feats):
        '''
        in advance at head and tail add frames,
        so extract need features
        '''
        if self.time_major is False:
            input_feats = tf.transpose(input_feats, [1, 0, 2])
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

            if self.time_major is False:
                output = tf.transpose(output, [1, 0, 2])
            return output
        else:
            # padding
            return self.SpliceFeats(input_feats)

class TdnnLayer(tf.keras.layers.Layer):
    def __init__(self, units, splice, time_major=True, splice_padding=False, **kwargs):
        super(TdnnLayer, self).__init__(**kwargs)
        self.layers = []
        self.layers.append(SpliceLayer(splice, time_major, splice_padding))
        self.layers.append(tf.keras.layers.Dense(units))
        self.layers.append(tf.keras.activations.relu)
        self.layers.append(NormalizeLayer)


class NormalizeLayer(tf.keras.layers.Layer):
    def __init__(self, input_dim=0, target_rms=1.0, axis=-1, epsilon=1.3552527156068805425e-20, **kwargs):
        super(NormalizeLayer, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.target_rms = target_rms
        self.axis = axis
        self.epsilon = epsilon
        self.scale = self.target_rms * self.target_rms * self.input_dim
        self.scale = pow(self.scale, 1/2)


    def call(self, input_feats):
        #input_feats = tf.convert_to_tensor(input_feats, dtype=self.dtype, name="input_feats")
        square_sum = tf.math.reduce_sum(
                tf.math.square(input_feats), self.axis, keepdims=True)
        x_inv_norm = tf.math.rsqrt(tf.maximum(square_sum, self.epsilon))
        x_inv_norm = tf.math.multiply(x_inv_norm, self.scale)
        return tf.math.multiply(input_feats, x_inv_norm)

    def get_config(self):
        config = {'input_dim':self.input_dim, 'target_rms':self.target_rms, 'axis':self.axis,
                'epsilon':self.epsilon, 'scale':self.scale}
        base_config = super(NormalizeLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Sigmoid(tf.keras.layers.Layer):
    """Sigmoid activation function.
    Input shape:
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
    Output shape:
        Same shape as the input.
    Arguments:
        axis: Integer, axis along which the sigmoid normalization is applied.
    """
    def __init__(self, **kwargs):
        super(Sigmoid, self).__init__(**kwargs)

    def call(self, input_feats):
        return tf.keras.activations.sigmoid(input_feats)
    
    def get_config(self):
        config = {}
        base_config = super(Sigmoid, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


