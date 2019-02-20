from __future__ import print_function
import numpy as np
import struct
import sys
from fst import *
from topsort import TopSort

sys.path.append("../")
from io_func import smart_open
from io_func.matio import read_token
from lattice_functions import *

class Lattice(FstHeader, object):
    def __init__(self):
        self._states = []
        self._wclass = LatticeWeightFloat
    
    def Read(self, fp):
        FstHeader.Read(self, fp)
        if self.ArcType() == 'compactlattice44':
            self._wclass = CompactLatticeWeightFloat
        elif self.ArcType() == 'lattice4':
            self._wclass = LatticeWeightFloat
            pass
        nstate = 0
        while nstate < self.NumStates():
            self._states.append(State(self._wclass).Read(fp, self._wclass))
            nstate += 1

    def Write(self, fp = None):
        if fp is None:
            nstate = 0
            for state in self._states:
                arcs = state.GetArcs()
                for arc in arcs:
                    if self.ArcType() == 'compactlattice44':
                        print('%d\t%d\t%d\t' % (nstate, arc._nextstate, arc._olabel), end='')
                    elif self.ArcType() == 'lattice4':
                        print('%d\t%d\t%d\t%d\t' % (nstate, arc._nextstate, arc._ilabel, arc._olabel), end='')
                    print(arc._weight, end = '\n')
                if state.IsFinal():
                    print('%d\t' % (nstate), end = '')
                    print(state._final, end ='\n')
                nstate += 1

    def SetArcType(self, arctype):
        FstHeader.SetArcType(self, arctype)
        if self.ArcType() == 'lattice4':
            self._wclass = LatticeWeightFloat
        elif self.ArcType() == 'compactlattice44':
            self._wclass = CompactLatticeWeightFloat

    def AddState(self):
        self._states.append(State(self._wclass))
        return len(self._states) - 1

    def AddArc(self, s, arc):
        assert s < len(self._states)
        self._states[s].AddArc(arc)

    def Final(self, s):
        assert s < len(self._states)
        return self._states[s]._final

    def SetFinal(self, s, weight):
        assert s < len(self._states)
        self._states[s].SetFinal(weight)

    def GetState(self, s):
        assert s < len(self._states)
        return self._states[s]

    def GetArcs(self, s):
        return self._states[s].GetArcs()

    def SetState(self, s, state):
        self._states[s] = state

def ConvertLattice(compactlat):
    '''
    convert compact lattice to fst lattice
    '''
    lat = Lattice()
    lat.SetArcType('lattice4')

    num_states = compactlat.NumStates()
    s = 0
    while s < num_states:
        news = lat.AddState()
        assert news == s
        s += 1

    lat.SetStart(compactlat.Start())
    
    tot_arc = 0
    s = 0
    while s < num_states:
        compact_final = compactlat.Final(s)
        if compact_final.IsZero() is False:
            string_length = len(compact_final._string)
            cur_state = s
            n = 0
            while n < string_length:
                next_state = lat.AddState()
                arc = Arc(lat._wclass)
                arc._ilabel = compact_final._string[n]
                arc._olabel = 0
                arc._nextstate = next_state
                if n == 0:
                    arc._weight = compact_final._weight
                else:
                    arc._weight = compact_final._weight.One()
                lat.AddArc(cur_state, arc)
                tot_arc += 1
                cur_state = next_state
                n += 1
            # add final
            if string_length > 0:
                lat.SetFinal(cur_state, compact_final._weight.One())
            else:
                lat.SetFinal(cur_state, compact_final._weight)
        arcs = compactlat.GetState(s).GetArcs()
        # extend arc
        for arc in arcs:
            string_length = len(arc._weight._string)
            cur_state = s
            # for all but the last element in the string--
            # add a temporary state.
            n = 0
            while n < string_length - 1:
                next_state = lat.AddState()
                new_arc = Arc(lat._wclass)
                new_arc._ilabel = arc._weight._string[n]
                new_arc._nextstate = next_state
                if n == 0:
                    new_arc._weight = arc._weight._weight
                    new_arc._olabel = arc._ilabel
                else:
                    new_arc._weight = arc._weight._weight.One()
                    new_arc._olabel = 0
                lat.AddArc(cur_state, new_arc)
                tot_arc += 1
                cur_state = next_state
                n += 1
            new_arc = Arc(lat._wclass)
            if string_length > 0:
                new_arc._ilabel = arc._weight._string[-1]
            else:
                new_arc._ilabel = 0
            if string_length <= 1:
                new_arc._olabel = arc._ilabel
                new_arc._weight = arc._weight._weight
            else:
                new_arc._olabel = 0
                new_arc._weight = arc._weight._weight.One()
            new_arc._nextstate = arc._nextstate
            lat.AddArc(cur_state, new_arc)
            tot_arc += 1
        s += 1
    lat.SetNumStates(len(lat._states))
    lat.SetNumArcs(tot_arc)
    lat.SetStart(0)
    return lat


if __name__ == "__main__":
    fp = open(sys.argv[1],'r')
    while True:
        lattice = Lattice()
        key = read_token(fp)
        if key is None:
            break
        lattice.Read(fp)
        print(key)
        lattice.Write()
        lat = ConvertLattice(lattice)
        lat.Write()
        TopSort(lat)
        lat.Write()
        ScaleLattice(lat, 1.0, 0.083)
        tot_backward_prob, acoustic_like_sum, post = LatticeForwardBackward(lat)
        #break
    fp.close()
