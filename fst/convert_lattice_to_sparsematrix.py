from __future__ import print_function
import numpy as np
import sys


from lattice import *


def ConvertLatticeToSparseMatrix(lat):
    '''
    (input) lat : must be topsort and have super final

    return      : indexs info, pdf_values , lmweight_values, amweight_values, statesinfo, shape

    '''
    indexs = []               # record [instate, nextstate]
    pdf_values = []           # arc pdf id
    lmweight_values = []      # arc lm weight
    amweight_values = []      # arc am weight
    statesinfo = []           # state in indexs offset and length [offset, len]

    num_states = lat.NumStates()
    start_state = lat.Start()

    assert start_state == 0 and 'start state id must be 0'
    s = 0
    offset = 0
    while s < num_states:
        length = 0
        state = lat.GetState(s)
        for arc in state.GetArcs():
            index = [s, arc._nextstate]
            pdf_v = arc._ilabel
            lm_weight_v = arc._weight._value1
            am_weight_v = arc._weight._value2

            indexs.append(index)
            pdf_values.append(pdf_v)
            lmweight_values.append(lm_weight_v)
            amweight_values.append(am_weight_v)
            length += 1
        statesinfo.append([offset, length])
        offset += length
        s += 1
    shape = [num_states, num_states]
    
    return np.array(indexs, dtype=np.int32), np.array(pdf_values, dtype=np.int32), np.array(lmweight_values, dtype=np.float32), np.array(amweight_values, dtype=np.float32), np.array(statesinfo, dtype=np.int32), shape



