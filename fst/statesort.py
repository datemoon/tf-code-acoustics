
import logging
import sys
sys.path.extend(["../","./"])
from fst.fst_base import *

def StateSort(fst, order):
    if len(order) != fst.NumStates():
        logging.info("StateSort: Bad order vector size: %d" % len(order))
        return

    if fst.Start() == kNoStateId:
        return
    done = [ False for x in range(len(order))]
    arcsa = []
    arcsb = []
    start = fst.Start()
    fst.SetStart(order[start])
    for s1 in range(fst.NumStates()):
        if done[s1]:
            continue
        state1 = fst.GetState(s1)
        while done[s1] is False:
            s2 = order[s1]
            if done[s2] is False:
                state2 = fst.GetState(s2)
            
            # change nextstate
            for arc in state1.GetArcs():
                arc._nextstate = order[arc._nextstate]
            fst.SetState(s2, state1)
            done[s1] = True
            
            s1 = s2
            state1 = state2
