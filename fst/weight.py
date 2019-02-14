import struct

class LatticeWeightFloat(object):
    def __init__(self, value1 = 0.0, value2 = 0.0):
        self._value1 = value1
        self._value2 = value2

    def Read(self, fp):
        self._value1 = struct.unpack(str('<f'), fp.read(4))[0]
        self._value2 = struct.unpack(str('<f'), fp.read(4))[0]

    def Value(self):
        return self._value1 + self._value2

class Weight(object):
    def __init__(self, value = 0.0):
        self._value = value

    def Read(self, fp):
        self._value = struct.unpack(str('<f'), fp.read(4))[0]

    def Value(self):
        return self._value

class CompactLatticeWeightFloat(object):
    def __init__(self, weight = LatticeWeightFloat(), string = []):
        self._weight = weight
        self._string = string

    def Read(self, fp):
        self._weight.Read(fp)
        sz = struct.unpack(str('<i'), fp.read(4))[0]
        nilable = 0
        while nilable < sz:
            ilabel = struct.unpack(str('<i'), fp.read(4))[0]
            self._string.append(ilabel)


    def Value(self):
        return self._weight.Value()


