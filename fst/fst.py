from __future__ import unicode_literals
import numpy as np
import struct
import sys
from weight import *

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
        return self._fsttype

    def ArcType(self):
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

    def Write(self, fp):
        pass

class Arc(object):
    def __init__(self, weight, ilabel = -1, olabel = -1, n = -1):
        self._ilabel = ilabel
        self._olabel = olabel
        self._weight = weight
        self._nextstate = n

class State(object):
    def __init__(self):
        # list
        self._arcs = []
        self._final = float('inf')

    def Read(self, fp, weight):
        self._final = weight.Read(fp)
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

class Fst(object):
    def __init__(self):
        # list
        self._fstheader = FstHeader()
        self._states = []

    def Start(self):
        pass

    def Read(self, fp):
        magicnumber = struct.unpack(str('<i'), fp.read(4))[0]
        assert 2125659606 == magicnumber

        self._fstheader.Read(fp)

#        length = struct.unpack(str('<i'), fp.read(4))[0]
#        fsttype = fp.read(length)

#        length = struct.unpack(str('<i'), fp.read(4))[0]
#        arctype = fp.read(length)

#        version = struct.unpack(str('<i'), fp.read(4))[0]
#        flags = struct.unpack(str('<i'), fp.read(4))[0]
#        properties = struct.unpack(str('<Q'), fp.read(8))[0]
#        start = struct.unpack(str('<q'), fp.read(8))[0]
#        numstates = struct.unpack(str('<q'), fp.read(8))[0]
#        numarcs = struct.unpack(str('<q'), fp.read(8))[0]

        nstate = 0
        while nstate < self._fstheader.NumStates():
            weight = Weight()
            self._states.append(State().Read(fp, weight))
            nstate += 1

    def Write(self, fp = None):
        if fp is None:
            nstate = 0
            for state in self._states:
                for arc in state.GetArcs():
                    print('%d %d %d %d %f' % (nstate, arc._nextstate, arc._ilabel,arc._olabel, arc._weight.Value()))
                nstate += 1
                
if __name__ == "__main__":
    fst = Fst()
    fp = open(sys.argv[1],'r')
    fst.Read(fp)
    fst.Write()

