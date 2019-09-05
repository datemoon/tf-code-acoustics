from __future__ import unicode_literals
from __future__ import print_function
import numpy as np
import struct
import sys
sys.path.extend(["../","./"])
from fst.weight import *

kNoStateId = -1

class FstHeader(object):
    def __init__(self):
        self._fsttype = ''    # E.g. "vector".
        self._arctype = ''    # E.g. "standard".
        self._version = 0     # int32 Type version number.
        self._flags = 0       # int32 File format bits.
        self._properties = 0  # uint64 FST property bits.
        self._start = -1      # int64 Start state.
        self._numstates = 0   # int64 states
        self._numarcs = 0     # int64 arcs

    def FstType(self):
        if type(self._fsttype) is bytes:
            return bytes.decode(self._fsttype)
        else:
            return self._fsttype

    def ArcType(self):
        if type(self._arctype) is bytes:
            return bytes.decode(self._arctype)
        else:
            return self._arctype

    def Version(self):
        return self._version

    def GetFlags(self):
        return self._flags

    def Properties(self):
        return self._properties

    def Start(self):
        return self._start

    def NumStates(self):
        return self._numstates

    def NumArcs(self):
        return self._numarcs

    def SetFstType(self, fsttype):
        self._fsttype = fsttype

    def SetArcType(self, arctype):
        self._arctype = arctype

    def SetVersion(self, version):
        self._version = version

    def SetFlags(self, flags):
        self._flags = flags

    def SetProperties(self, properties):
        self._properties = properties

    def SetStart(self, start):
        self._start = start


    def SetNumStates(self, numstates):
        self._numstates = numstates

    def SetNumArcs(self, numarcs):
        self._numarcs = numarcs

    def Read(self, fp):
        magicnumber = struct.unpack(str('<i'), fp.read(4))[0]
        assert 2125659606 == magicnumber
        
        length = struct.unpack(str('<i'), fp.read(4))[0]
        self._fsttype = fp.read(length)

        length = struct.unpack(str('<i'), fp.read(4))[0]
        self._arctype = fp.read(length)

        self._version = struct.unpack(str('<i'), fp.read(4))[0]
        self._flags = struct.unpack(str('<i'), fp.read(4))[0]
        self._properties = struct.unpack(str('<Q'), fp.read(8))[0]
        self._start = struct.unpack(str('<q'), fp.read(8))[0]
        self._numstates = struct.unpack(str('<q'), fp.read(8))[0]
        self._numarcs = struct.unpack(str('<q'), fp.read(8))[0]

    def Write(self, fp = None):
        if fp is None:
            print("fst type     :" , self._fsttype)
            print("arc type     :" , self._arctype)
            print("version      :" , self._version)
            print("flags        :" , self._flags)
            print("properties   :" , self._properties)
            print("start        :" , self._start)
            print("numstates    :" , self._numstates)
            print("numarcs      :" , self._numarcs)
        else:
            # write file
            pass

    def __repr__(self):
        pri = 'fstheader parameters:\n'
        for key in self.__dict__:
            pri += key + ':\t' + str(self.__dict__[key]) +'\n'
        return pri

class Arc(object):
    def __init__(self, weight = Weight, ilabel = -1, olabel = -1, n = -1):
        self._ilabel = ilabel
        self._olabel = olabel
        self._weight = weight()
        self._nextstate = n

    def SetWeight(self, w):
        self._weight = w

    def __repr__(self):
        pri = str(self._nextstate) + '\t' 
        pri += str(self._ilabel) + '\t'
        pri += str(self._olabel) + '\t'
        pri += self._weight.__repr__()
        return pri

class State(object):
    '''
    weight: it's class
    '''
    def __init__(self, weight = Weight):
        # list
        self._arcs = []
        self._final = weight()

    def Read(self, fp, weight = Weight):
        self._final.Read(fp)
        #self._final = struct.unpack(str('<f'), fp.read(4))[0]
        arcs_num = struct.unpack(str('<q'), fp.read(8))[0] # int64
        n = 0
        while n < arcs_num:
            arc = Arc(weight)
            arc._ilabel = struct.unpack(str('<i'), fp.read(4))[0]
            arc._olabel = struct.unpack(str('<i'), fp.read(4))[0]
            arc._weight.Read(fp)
            arc._nextstate = struct.unpack(str('<i'), fp.read(4))[0]
            self._arcs.append(arc)
            #print('%d %d %d %f' % (arc._nextstate, arc._ilabel, arc._olabel, arc._weight.Value()))
            n += 1
        return self

    def GetArcs(self):
        return self._arcs

    def IsFinal(self):
        if self._final.IsZero():
            return False
        else:
            return True

    def Final(self):
        return self._final

    def AddArc(self, arc):
        self._arcs.append(arc)

    def SetFinal(self, weight):
        self._final = weight

class Fst(FstHeader, object):
    def __init__(self, wclass = Weight):
        # list
        super(Fst, self).__init__()
        self._states = []
        self._wclass = wclass

    def Read(self, fp):
        FstHeader.Read(self, fp)
        if self.ArcType() == 'standard':
            self._wclass = Weight
        elif self.ArcType() == 'compactlattice44':
            self._wclass = CompactLatticeWeightFloat
        elif self.ArcType() == 'lattice4':
            self._wclass = LatticeWeightFloat
        else:
            assert 'no this arc type' and False

        nstate = 0
        while nstate < self.NumStates():
            self._states.append(State(self._wclass).Read(fp, self._wclass))
            nstate += 1

    def Write(self, fp = None):
        FstHeader.Write(self, fp)
        if fp is None:
            nstate = 0
            for state in self._states:
                for arc in state.GetArcs():
                    if self.ArcType() == 'standard':
                        print('%d %d %d %d ' % (nstate, arc._nextstate, arc._ilabel, arc._olabel), end = '')
                    elif self.ArcType() == 'compactlattice44':
                        print('%d\t%d\t%d\t' % (nstate, arc._nextstate, arc._olabel), end='')
                    elif self.ArcType() == 'lattice4':
                        print('%d\t%d\t%d\t%d\t' % (nstate, arc._nextstate, arc._ilabel, arc._olabel), end='')

                    print(arc._weight, end = '\n')
                if state.IsFinal():
                    print('%d\t' % (nstate), end = '')
                    print(state._final, end='\n')
                nstate += 1
        else: # write file
            pass

    def SetArcType(self, arctype):
        FstHeader.SetArcType(self, arctype)
        if self.ArcType() == 'standard':
            self._wclass = Weight
        elif self.ArcType() == 'lattice4':
            self._wclass = LatticeWeightFloat
        elif self.ArcType() == 'compactlattice44':
            self._wclass = CompactLatticeWeightFloat
        else:
            return False
        return True

    def AddState(self):
        self._states.append(State(self._wclass))
        self._numstates += 1
        return len(self._states) - 1

    def AddArc(self, s, arc):
        assert s < len(self._states)
        self._states[s].AddArc(arc)
        self._numarcs += 1

    def Final(self, s):
        '''
        s (input): fst state id
        return   : fst the s final weight
        '''
        assert s < len(self._states)
        return self._states[s].Final()

    def SetFinal(self, s, weight):
        assert s < len(self._states)
        self._states[s].SetFinal(weight)

    def GetState(self, s):
        assert s < len(self._states)
        return self._states[s]

    def GetStates(self):
        return self._states

    def GetArcs(self, s):
        return self._states[s].GetArcs()

    def SetState(self, s, state):
        self._states[s] = state



if __name__ == "__main__":
    fst = Fst()
    fp = open(sys.argv[1],'r')
    fst.Read(fp)
    fst.Write()
    fp.close()

