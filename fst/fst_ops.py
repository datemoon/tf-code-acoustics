
from fst import *

def SuperFinalFst(fst):
    '''
    fst (input): fst
    fst (output): only one final fst
    '''
    num_states = fst.NumStates()
    final_state_list = []
    for s in range(num_states):
        if fst.GetState(s).IsFinal():
            final_state_list.append(s)
    
    if len(final_state_list) == 1:
        if fst.Final(final_state_list[0]).IsOne():
            return True
        
    super_final_stateid = fst.AddState()
    state_unfinal_weightclass
    if fst.ArcType() == 'standard':
        fst.SetFinal(super_final_stateid, Weight(0.0))
        state_unfinal_weightclass = Weight
    elif self.ArcType() == 'compactlattice44':
        fst.SetFinal(super_final_stateid, CompactLatticeWeightFloat().SetValue(0.0, 0.0))
        state_unfinal_weightclass = CompactLatticeWeightFloat
    elif self.ArcType() == 'lattice4':
        fst.SetFinal(super_final_stateid, LatticeWeightFloat(0.0, 0.0))
        state_unfinal_weightclass = LatticeWeightFloat
    
    for s in final_state_list:
        state = fst.GetState(s)
        
        arc_weight = state.Final().Value()
        state.SetFinal(state_unfinal_weightclass())
        # add arc from current state to super final
        fst.AddArc(s, arc(arc_weight, 0, 0, super_final_stateid))

    return True
