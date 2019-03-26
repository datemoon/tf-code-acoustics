from __future__ import print_function
import sys
sys.path.extend(["../","./"])
from fst.fst import *
from fst.topsort import TopSort
from fst.fst_ops import *

from fst.lattice_functions import *
from fst.convert_lattice_to_sparsematrix import *
from io_func import smart_open
from io_func.matio import read_token

class Lattice(Fst, object):
    def __init__(self, key = None, wclass = Weight):
        super(Lattice, self).__init__()
        self._wclass = wclass
        self._key = key

    def Read(self, fp):
        self._key = read_token(fp)
        if self._key is None:
            return self._key
        Fst.Read(self, fp)
        return self._key

    def Write(self, fp = None):
        if self._key is not None:
            print('%s' % (self._key))
        Fst.Write(self, fp)

    def ReadScp(self, scp_line):
        if scp_line is None:
            return None
        self._key, path_pos = scp_line.replace('\n','').split(' ')
        path, pos = path_pos.split(':')
        latfp = smart_open(path, 'rb')
        latfp.seek(int(pos),0)
        Fst.Read(self, latfp)
        latfp.close()
        return self._key

    def SetKey(self, key):
        self._key = key

def ConvertLattice(compactlat):
    '''
    convert compact lattice to fst lattice
    '''
    if 'compact' not in compactlat.ArcType():
        return compactlat
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
    with open(sys.argv[1],'r') as lat_scp_fp:
        for scp_line in lat_scp_fp:
            lattice = Lattice()
            key = lattice.ReadScp(scp_line)
            lattice.Write()
            lat = ConvertLattice(lattice)
            lat.SetKey(key)
            SuperFinalFst(lat)
            TopSort(lat)
            # save lm weight and am weight set zero
            ScaleLattice(lat, 1.0, 0.0)
            lat.Write()
            indexs_info, pdf_values , lmweight_values, amweight_values, statesinfo, shape = ConvertLatticeToSparseMatrix(lat)

    exit(0)
    fp = open(sys.argv[1],'r')
    while True:
        lattice = Lattice()
        #key = read_token(fp)
        key = lattice.Read(fp) 
        if key is None:
            break
        print(key)
        lattice.Write()
        lat = ConvertLattice(lattice)
        lat.Write()
        TopSort(lat)

        indexs_info, pdf_values , lmweight_values, amweight_values, statesinfo, shape = ConvertLatticeToSparseMatrix(lat)
        lat.Write()
        ScaleLattice(lat, 1.0, 0.083)
        tot_backward_prob, acoustic_like_sum, post = LatticeForwardBackward(lat)
        #break
    fp.close()
