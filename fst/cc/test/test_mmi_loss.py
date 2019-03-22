from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import numpy as np
import sys
from tensorflow_py_api import mmi
from tensorflow.python.client import device_lib
from io_test import *
import time

class MMILossTest(tf.test.TestCase):

    def _run_mmi(self, inputs, sequence_length, labels,
            indexs, pdf_values, lm_ws, am_ws, statesinfo, num_states,
            expected_gradients, expected_costs,
            old_acoustic_scale = 0.0,
            acoustic_scale = 1.0, drop_frames = True, time_major = True,
            expected_error=None):
        self.assertEqual(inputs.shape, expected_gradients.shape)

        inputs_t = tf.constant(inputs)
        sequence_length_t = tf.constant(sequence_length)
        labels_t = tf.constant(labels)
        indexs_t = tf.constant(indexs)
        pdf_values_t = tf.constant(pdf_values)
        lm_ws_t = tf.constant(lm_ws)
        am_ws_t = tf.constant(am_ws)
        statesinfo_t = tf.constant(statesinfo)
        num_states_t = tf.constant(num_states)

        costs = mmi(inputs_t, sequence_length_t, labels_t,
                indexs_t, pdf_values_t, lm_ws_t, am_ws_t, 
                statesinfo_t, num_states_t,
                old_acoustic_scale = old_acoustic_scale,
                acoustic_scale = acoustic_scale, 
                drop_frames = drop_frames, 
                time_major = time_major)

        grad = tf.gradients(costs, [inputs_t])[0]

        log_dev_placement = False
        config = tf.ConfigProto(log_device_placement=log_dev_placement,
                device_count={'GPU': 0})
        with self.test_session(use_gpu=False) as sess:
            if expected_error is None:
                start_time = time.time()
                (tf_costs, tf_grad) = sess.run([costs, grad])
                end_time = time.time()
                print("session run mmi time:",end_time-start_time)
                #self.assertAllClose(tf_costs, expected_costs, atol=1e-5)
                self.assertAllClose(tf_grad, expected_gradients, atol=1e-5)
            else:
                with self.assertRaisesOpError(expected_error):
                    sess.run([costs, grad])

    def testBasic(self):
        '''Test one batch'''
        #indexs, pdf_values, lm_ws, am_ws, statesinfo, h_nnet_out_h, pdf_ali, gradient = ReadInput('python.source')

        batch_indexs, batch_pdf_values, batch_lm_ws, batch_am_ws, batch_statesinfo, batch_h_nnet_out_h, batch_pdf_ali, batch_gradient, batch_sequence_length, batch_num_states, batch_expected_costs = BatchIo('python.source', 16)
        
        inputs = batch_h_nnet_out_h
        sequence_length = batch_sequence_length
        labels = batch_pdf_ali
        num_states = batch_num_states
        expected_gradients = batch_gradient
        expected_costs = batch_expected_costs

        indexs = batch_indexs
        pdf_values = batch_pdf_values
        lm_ws = batch_lm_ws
        am_ws = batch_am_ws
        statesinfo = batch_statesinfo
        
        while True:
            self._run_mmi(inputs, sequence_length, labels,
                indexs, pdf_values, lm_ws, am_ws, statesinfo, num_states,
                expected_gradients, expected_costs,
                old_acoustic_scale = 0.0,
                acoustic_scale = 0.083, drop_frames = True, time_major = True,
                expected_error=None)


if __name__ == "__main__":
    tf.test.main()

