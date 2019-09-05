
import sys
sys.path.extend(["../","./"])
from fst.fst_base import *

def SuperFinalFst(fst):
    '''Convert only one final fst and the final is last.
    fst (input) : fst
    fst (output): bool
    '''
    num_states = fst.NumStates()
    final_state_list = []
    for s in range(num_states):
        if fst.GetState(s).IsFinal():
            final_state_list.append(s)
    
    if len(final_state_list) == 1:
        if fst.Final(final_state_list[0]).IsOne():
            return True
    super_final_stateid = None
    # find super final state
    for f_state in final_state_list:
        if fst.Final(f_state).IsOne():
            super_final_stateid = f_state
            break
    # if not find super final state ,create 
    if super_final_stateid is None:
        super_final_stateid = fst.AddState()

    # weight type
    state_unfinal_weightclass = None
    if fst.ArcType() == 'standard':
        fst.SetFinal(super_final_stateid, Weight(0.0))
        state_unfinal_weightclass = Weight
    elif fst.ArcType() == 'compactlattice44':
        fst.SetFinal(super_final_stateid, CompactLatticeWeightFloat().SetValue(0.0, 0.0))
        state_unfinal_weightclass = CompactLatticeWeightFloat
    elif fst.ArcType() == 'lattice4':
        fst.SetFinal(super_final_stateid, LatticeWeightFloat(0.0, 0.0))
        state_unfinal_weightclass = LatticeWeightFloat
    
    for s in final_state_list:
        # don't process super final state
        if s == super_final_stateid:
            continue
        state = fst.GetState(s)
        
        arc_weight = state.Final()
        state.SetFinal(state_unfinal_weightclass())
        arc = Arc(state_unfinal_weightclass, 0, 0, super_final_stateid)
        arc.SetWeight(arc_weight)
        # add arc from current state to super final
        fst.AddArc(s, arc)

    return True
