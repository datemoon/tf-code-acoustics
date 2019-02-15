from __future__ import print_function
import struct

class LatticeWeightFloat(object):
    def __init__(self, value1 = float('inf'), value2 = float('inf')):
        self._value1 = value1
        self._value2 = value2

    def Read(self, fp):
        self._value1 = struct.unpack(str('<f'), fp.read(4))[0]
        self._value2 = struct.unpack(str('<f'), fp.read(4))[0]

    def Value(self):
        return self._value1 + self._value2

    def __repr__(self):
        pri = str(self._value1) + ',' + str(self._value2)
        return pri

    def IsZero(self):
        if self.Value() == float('inf'):
            return True
        else:
            return False

    def Zero(self):
        return LatticeWeightFloat(value1=float('inf'), value2=float('inf'))

    def One(self):
        return LatticeWeightFloat(value1=0.0, value2=0.0)


class Weight(object):
    def __init__(self, value = float('inf')):
        self._value = value

    def Read(self, fp):
        self._value = struct.unpack(str('<f'), fp.read(4))[0]

    def Value(self):
        return self._value
    
    def IsZero(self):
        if self.Value() == float('inf'):
            return True
        else:
            return False

    def Zero(self):
        return Weight(value = float('inf'))

    def One(self):
        return Weight(value = 0.0)

    def __repr__(self):
        pri = str(self._value) 
        return pri

class CompactLatticeWeightFloat(object):
    def __init__(self, weight = LatticeWeightFloat):
        self._weight = weight()
        self._string = []

    def Read(self, fp):
        self._weight.Read(fp)
        sz = struct.unpack(str('<i'), fp.read(4))[0]
        nilable = 0
        while nilable < sz:
            ilabel = struct.unpack(str('<i'), fp.read(4))[0]
            self._string.append(ilabel)
            nilable += 1


    def Value(self):
        return self._weight.Value()
    
    def IsZero(self):
        if self.Value() == float('inf'):
            return True
        else:
            return False

    def __repr__(self):
        pri = self._weight.__repr__() 
        if len(self._string) >= 1:
            pri += ','
            for ilabel in self._string[:-1]:
                pri += str(ilabel) + '_'
            pri += str(self._string[-1])
        return pri
