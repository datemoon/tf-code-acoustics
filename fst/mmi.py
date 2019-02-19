
import numpy as np

def LatticeAcousticRescore(nnet_out, state_times, lat):
    # lat must be top sort and ilabel = pdf+1
    # check
    for s in range(lat.NumStates()):
        state = lat.GetState(s)
        t = state_times[s]
        for arc in state.GetArcs():
            if arc._ilabel != 0:
                pdf = arc._ilabel - 1
                arc._weight._value2 -= nnet_out[t][pdf]
        # end state
    # end all states

        
