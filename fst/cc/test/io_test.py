from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import numpy as np

def ReadInput(source_file):
    source = {}
    val = []
    key = ''
    row = 0
    r = 0
    #for line in open(source_file,'r'):
    with open(source_file,'r') as fp:
        for line in fp:
	        if '#' == line[0]:
	            continue
	        if ':' in line:
	            if row != 0:
	                source[key] = val
	                key = ''
	                assert r == row
	                row = 0
	                r = 0
	                val = []
	            key = line.strip().split(':')[0]
	            row = int(line.strip().split(':')[1])
	            continue
	        val.append(line.strip().split())
	        r += 1

    assert r == row
    source[key] = val

    indexs = np.array(source['indexs'], dtype=np.int32)
    indexs = np.reshape(indexs, (1,indexs.shape[0], indexs.shape[1]))
    pdf_values = np.array(source['pdf_values'], dtype=np.int32)
    pdf_values = np.reshape(pdf_values, (1,-1))

    lm_ws = np.array(source['lm_ws'], dtype=np.float32)
    lm_ws = np.reshape(lm_ws, (1,-1))
    am_ws = np.array(source['am_ws'], dtype=np.float32)
    am_ws = np.reshape(am_ws, (1,-1))
    statesinfo = np.array(source['statesinfo'], dtype=np.int32)
    statesinfo = np.reshape(statesinfo,(1, statesinfo.shape[0],statesinfo.shape[1]))

    h_nnet_out_h = np.array(source['h_nnet_out_h'], dtype=np.float32)
    h_nnet_out_h_shape = h_nnet_out_h.shape
    h_nnet_out_h = np.reshape(h_nnet_out_h, (h_nnet_out_h_shape[0],1,h_nnet_out_h_shape[1]))

    pdf_ali = np.array(source['pdf_ali'], dtype=np.int32)
    pdf_ali = np.reshape(pdf_ali, (1,-1))
    gradient = np.array(source['gradient'],dtype=np.float32)
    gradient = np.reshape(gradient, h_nnet_out_h.shape)
    return indexs, pdf_values, lm_ws, am_ws, statesinfo, h_nnet_out_h, pdf_ali, gradient

def BatchIo(source_file, batch):
    indexs, pdf_values, lm_ws, am_ws, statesinfo, h_nnet_out_h, pdf_ali, gradient = ReadInput('python.source')
    
    batch_indexs = indexs
    batch_pdf_values = pdf_values
    batch_lm_ws = lm_ws
    batch_am_ws = am_ws
    batch_statesinfo = statesinfo
    batch_h_nnet_out_h = h_nnet_out_h
    batch_pdf_ali = pdf_ali
    batch_gradient = gradient
    sequence_length = np.array([pdf_ali.shape[1]],dtype=np.int32)
    batch_sequence_length = sequence_length
    num_states = np.array([statesinfo.shape[1]],dtype=np.int32)
    batch_num_states = num_states
    expected_costs = np.array([0.0],dtype=np.float32)
    batch_expected_costs = expected_costs
    for i in range(batch-1):
        batch_indexs = np.vstack((batch_indexs, indexs))
        batch_pdf_values = np.vstack((batch_pdf_values, pdf_values))
        batch_lm_ws = np.vstack((batch_lm_ws, lm_ws))
        batch_am_ws = np.vstack((batch_am_ws, am_ws))
        batch_statesinfo = np.vstack((batch_statesinfo, statesinfo))
        batch_h_nnet_out_h = np.hstack((batch_h_nnet_out_h, h_nnet_out_h))
        batch_pdf_ali = np.vstack((batch_pdf_ali, pdf_ali))
        batch_gradient = np.hstack((batch_gradient, gradient))
        batch_sequence_length = np.hstack((batch_sequence_length, sequence_length))
        batch_num_states = np.hstack((batch_num_states, num_states))
        batch_expected_costs = np.hstack((batch_expected_costs, expected_costs))

    
    return batch_indexs, batch_pdf_values, batch_lm_ws, batch_am_ws, batch_statesinfo, batch_h_nnet_out_h, batch_pdf_ali, batch_gradient, batch_sequence_length, batch_num_states, batch_expected_costs

if __name__ == '__main__':
    ReadInput(sys.argv[1])
    BatchIo(sys.argv[1],2)
