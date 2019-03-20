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


if __name__ == '__main__':
    ReadInput(sys.argv[1])
