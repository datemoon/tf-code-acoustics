from __future__ import print_function

import struct
import sys


sys.path.extend(["../","./"])
from fst.fst_base import *
from io_func.matio import read_token, read_matrix_or_vector
from io_func import smart_open
import numpy as np
from six import binary_type

def ReadKey(fd):
    key = read_token(fd, set((' ', '\t', '\n')) )
    return key

def ExpectToken(fd, token):
    """ ExpectToken tries to read in the given token
    """
    tok = read_token(fd)
    if tok == token:
        return True
    else:
        return False
    
def ReadBasicChar(fd, dtype = 'char', endian = '<'):
    return struct.unpack(str(endian + 'c'), fd.read(1))[0]

def ReadBasicType(fd, dtype, endian='<'):
    len_c_in = fd.read(1)
    if len_c_in == '':
        print("ReadBasicType: encountered end of stream.")
    len_c = struct.unpack(str(endian + 'b'),len_c_in)[0]
    if dtype == 'int':
        if len_c != 4:
            print("ReadBasicType: did not get expected integer type, " 
                    + str(len_c) + " vs. " + str(4) + 
                    ".  You can change this code to successfully " +
                    "read it later, if needed.")
        return struct.unpack(str(endian + 'i'), fd.read(4))[0]
    if dtype == 'uint':
        if len_c != 4:
            print("ReadBasicType: did not get expected integer type, " 
                    + str(len_c) + " vs. " + str(4) + 
                    ".  You can change this code to successfully " +
                    "read it later, if needed.")
        return struct.unpack(str(endian + 'I'), fd.read(4))[0]
    elif dtype == 'float':
        if len_c != 4:
            print("ReadBasicType: did not get expected integer type, " 
                    + str(len_c) + " vs. " + str(4) + 
                    ".  You can change this code to successfully " +
                    "read it later, if needed.")
        return struct.unpack(str(endian + 'f'), fd.read(4))[0]
    elif dtype == 'double':
        if len_c != 8:
            print("ReadBasicType: did not get expected integer type, " 
                    + str(len_c) + " vs. " + str(8) + 
                    ".  You can change this code to successfully " +
                    "read it later, if needed.")
        return struct.unpack(str(endian + 'd'), fd.read(8))[0]
    elif dtype == 'char':
        if len_c != 1:
            print("ReadBasicType: did not get expected integer type, " 
                    + str(len_c) + " vs. " + str(1) + 
                    ".  You can change this code to successfully " +
                    "read it later, if needed.")
        return struct.unpack(str(endian + 'c'), fd.read(1))[0]
    elif dtype == 'singedchar':
        if len_c != 1:
            print("ReadBasicType: did not get expected integer type, " 
                    + str(len_c) + " vs. " + str(1) + 
                    ".  You can change this code to successfully " +
                    "read it later, if needed.")
        return struct.unpack(str(endian + 'b'), fd.read(1))[0]
    elif dtype == 'bool':
        if len_c != 1:
            print("ReadBasicType: did not get expected integer type, " 
                    + str(len_c) + " vs. " + str(1) + 
                    ".  You can change this code to successfully " +
                    "read it later, if needed.")
        if 'F' == struct.unpack(str(endian + 'c'), fd.read(1))[0]:
            return False
        else:
            return True

class Supervision(object):
    def __init__(self):
        self.weight = 0.0
        self.num_sequences = 0
        self.frames_per_sequence = 0
        self.label_dim = 0
        self.e2e = False
        self.fst = Fst()

    def Read(self, fd, binary=True):
        '''
        read supervision
        '''
        ExpectToken(fd, "<Supervision>")
        
        ExpectToken(fd, "<Weight>");
        self.weight = ReadBasicType(fd, 'float')
        
        ExpectToken(fd, "<NumSequences>")
        self.num_sequences = ReadBasicType(fd, 'int')
        
        ExpectToken(fd, "<FramesPerSeq>")
        self.frames_per_sequence = ReadBasicType(fd, 'int')
        
        ExpectToken(fd, "<LabelDim>")
        self.label_dim = ReadBasicType(fd, 'int')
        
        ExpectToken(fd, "<End2End>")
        self.e2e = ReadBasicChar(fd, 'bool')

        if self.e2e == b'F':
            # read fst
            if binary:
                self.fst.Read(fd)
        elif self.e2e == b'T':
            pass
        else:
            pass

        ExpectToken(fd, "</Supervision>")

    def Write(self, fd = None):
        if fd is None:
            # print
            print("<Supervision> <Weight> " + str(self.weight) + 
                    " <NumSequences> " + str(self.num_sequences) + 
                    " <FramesPerSeq> " + str(self.num_sequences) +
                    " <LabelDim> " + str(self.label_dim) + 
                    "<End2End>" + self.e2e.decode() )
            self.fst.Write(fd)
            print("</Supervision>")
        else:
            pass

def ReadIndexVectorElementBinary(fd, i, index):
    c = struct.unpack(str('<' + 'b'), fd.read(1))[0]
    ind = []
    if i == 0:
        if c < 125 and c > -125:
            ind = [0, c, 0]
        else:
            if c != 127:
                assert "Unexpected character " + str(c) + " encountered while reading Index vector."
            ind.append(ReadBasicType(fd, 'int'))
            ind.append(ReadBasicType(fd, 'int'))
            ind.append(ReadBasicType(fd, 'int'))
    else:
        last_ind = index[-1]
        if c < 125 and c > -125:
            ind = [last_ind[0], last_ind[1]+c, last_ind[2]]
        else:
            if c != 127:
                assert "Unexpected character " + str(c) + " encountered while reading Index vector."
            ind.append(ReadBasicType(fd, 'int'))
            ind.append(ReadBasicType(fd, 'int'))
            ind.append(ReadBasicType(fd, 'int'))

    index.append(ind)


def ReadIndexVector(fd, binary=True):
    ExpectToken(fd, "<I1V>")
    size = ReadBasicType(fd, 'int')
    if size < 0 :
        assert "Error reading Index vector: size = " + str(size)
    vec = []
    if binary is True:
        for i in range(size):
            ReadIndexVectorElementBinary(fd, i, vec)
    else:
        pass

    return size, vec

class NnetIo(object):
    '''
    the name of the input in the neural net; in simple setups it
    will just be "input".
    '''
    def __init__(self):
        self.name = None
        self.indexes = None
        self.features = None
        self.size = None

    def GetFeat(self):
        return self.features
    
    def GetIndex(self):
        return self.indexes

    def GetSize(self):
        return self.size

    def Read(self, fd):
        ExpectToken(fd, "<NnetIo>")
        self.name = read_token(fd)
        self.size, self.indexes = ReadIndexVector(fd)
        self.features = read_matrix_or_vector(fd, endian='<',
                return_size=False, read_binary_flag=False)
        ExpectToken(fd, "</NnetIo>")

    def Write(self, fd = None):
        if fd is None:
            #print
            print("<NnetIo> " + self.name )
            print(self.indexes)
            print(self.features)
        else:
            pass

def ReadVectorAsChar(fd, sizeof_type, binary=True):
    if binary is True:
        sacle = float(1.0/255.0)
        pos = fd.tell()
        sz = ReadBasicType(fd, 'singedchar')
        if sz != sizeof_type:
            assert "ReadIntegerVector: expected to see type of size " + str(sizeof_type) + ", saw instead " + str(sz) + ", at file position " + str(pos)

        vecsz = ReadBasicType(fd, 'int')
        vec_val = np.frombuffer(fd.read(sizeof_type * vecsz), dtype=np.int8)
        vec_val = vec_val/sacle
    return list(vec_val)

def ReadVectorAsFloat(fd, binary=True):
    my_token = read_token(fd)
    if my_token == "FV":
        size = ReadBasicType(fd, 'int')
        deriv_weights = np.frombuffer(fd.read(4 * size), dtype=np.float32)
    elif my_token == "DV":
        size = ReadBasicType(fd, 'int')
        deriv_weights = np.frombuffer(fd.read(4 * size), dtype=np.float64)
    else:
        print("ReadVectorAsFloat error, no this type")
        sys.exit(1)

    return list(deriv_weights)


class NnetChainSupervision(object):
    def __init__(self):
        self.name = None
        self.indexes = None
        self.supervision = None
        self.deriv_weights = None
        self.size = None

    def GetFst(self):
        return self.supervision.fst

    def GetDerivWeights(self):
        return self.deriv_weights

    def GetIndex(self):
        return self.indexes

    def GetSize(self):
        return self.size

    def Read(self, fd):
        '''
        Read ChainSupervision data
        '''
        ExpectToken(fd, "<NnetChainSup>")
        self.name = read_token(fd)
        self.size, self.indexes = ReadIndexVector(fd)
        self.supervision = Supervision()
        self.supervision.Read(fd)
        token = read_token(fd)
        
        if token != "</NnetChainSup>":
            assert token == "<DW>" or token == "<DW2>"
            if token == "<DW>":
                self.deriv_weights = ReadVectorAsChar(fd, 1)
            else:
                self.deriv_weights = ReadVectorAsFloat(fd)
                
        ExpectToken(fd, "</NnetChainSup>")

    def Write(self, fd = None):
        if fd is None:
            # print
            print("<NnetChainSup> " + self.name )
            print(self.indexes)
            self.supervision.Write(fd)
            if self.deriv_weights is not None:
                print("<DW2> ")
                print(self.deriv_weights)
            print("</NnetChainSup>")
        else:
            pass

def InitKaldiInputStream(fd, binary):
    pos = fd.tell()
    binary_flag = fd.read(2)
    if binary_flag[:1] == b'\0':
        if binary_flag[1:2] == b'B':
            binary.append(True)
            return True
    else:
        binary.append(False)
        fd.seek(int(pos),0)
        return True

class NnetChainExample(object):
    def __init__(self):
        self.inputs = []
        self.outputs = []
        self.key = None

    def GetKey(self):
        return self.key

    def Input(self):
        return self.inputs

    def Output(self):
        return self.outputs

    def ReadScp(self, scp_line):
        utt_id, path_pos = scp_line.replace('\n','').split(' ')
        path, pos = path_pos.split(':')
        self.key = utt_id
        with open(path, 'rb') as fd:
            fd.seek(int(pos),0)
            return self.Read(fd, read_key=False)

    def Read(self, fd, read_key=True):
        if read_key:
            self.key = ReadKey(fd)
            if self.key in [None, '']:
                # the end of file
                return False
        binary = []
        InitKaldiInputStream(fd, binary)
        if binary[0] is False:
            print("egs is text and it's unsupported type.")
            return False

        ExpectToken(fd, "<Nnet3ChainEg>")
        ExpectToken(fd, "<NumInputs>")
        size = ReadBasicType(fd, 'int')
        if size < 1 or size > 1000000:
            assert 'Invalid size ' + size
        
        for i in range(size):
            nnetio = NnetIo()
            nnetio.Read(fd)
            self.inputs.append(nnetio)
            
        ExpectToken(fd, "<NumOutputs>")
        size = ReadBasicType(fd, 'int')
        if size < 1 or size > 1000000:
            assert 'Invalid size ' + size
            
        for i in range(size):
            nnetchainsupervision = NnetChainSupervision()
            nnetchainsupervision.Read(fd)
            self.outputs.append(nnetchainsupervision)
            
        ExpectToken(fd, "</Nnet3ChainEg>")
        return True

    def Write(self,fd = None):
        if fd is None:
            # print
            print(self.key + '\n')
            for i in self.inputs:
                i.Write(fd)

            for o in self.outputs:
                o.Write(fd)
        else:
            pass

def ReadEgsScp(scp_line):
    chain_example = NnetChainExample()
    chain_example.ReadScp(scp_line)

    return 

def ProcessEgsFeat(feat, in_indexes, out_indexes, splice_info, offset = 0):
    '''
    feat: must be pooling splice
    '''
    const_add = 2
    skip = out_indexes[1][1]-out_indexes[0][1]
    assert offset < skip
    
    in_frames = len(in_indexes)
    assert len(feat) == in_frames

    out_frames = len(out_indexes)

    in_start = in_indexes[0][1]
    in_end = in_indexes[-1][1]

    out_start = out_indexes[0][1]
    out_end = out_indexes[-1][1]

    left = splice_info[0]
    right = splice_info[-1]
    
    # calcluate start frame
    start_numframes = int(int(abs(in_start)-int(const_add/2) - abs(left) - abs(out_start)) / skip)
    end_numframes = int(int(abs(in_end) - int(const_add/2) - abs(right) -abs(out_end))/ skip)

    total_frames = start_numframes + end_numframes + out_frames

    start_id = abs(in_start - out_start + skip * start_numframes)
    end_id = abs(out_end) + end_numframes * skip - in_start 
    assert end_id == start_id + (total_frames-1) * skip 
    # add offset
    ret_feat = feat[start_id + offset : end_id + 1 + offset : skip]
    assert len(ret_feat) == total_frames
    #print("inframes:",in_frames)
    #print("out_frames:",out_frames)
    #print("total_frames:",total_frames)
    #assert total_frames-out_frames==12 or total_frames-out_frames==25
    return ret_feat



if __name__ == '__main__':
#    fd = smart_open(sys.argv[1],'rb')
#    while True:
#        chain_example = NnetChainExample()
#        ret = chain_example.Read(fd)
#        if ret is False:
#            break
#        print('read ' + chain_example.key + ' ok.')
        #chain_example.Write()

    for line in open(sys.argv[1],'rb'):
        chain_example = NnetChainExample()
        chain_example.ReadScp(line)
        print('read ' + chain_example.key + ' ok.')
        inputs = chain_example.Input()
        outputs = chain_example.Output()
        for iput,oput in zip(inputs,outputs):
            feat = iput.GetFeat()
            isize = iput.GetSize()
            assert isize == np.shape(feat)[0]
            feat0 = ProcessEgsFeat(feat, iput.GetIndex(), oput.GetIndex(),[-2,-1,0,1,2],0)
            feat1 = ProcessEgsFeat(feat, iput.GetIndex(), oput.GetIndex(),[-2,-1,0,1,2],1)
            feat2 = ProcessEgsFeat(feat, iput.GetIndex(), oput.GetIndex(),[-2,-1,0,1,2],2)
            print(feat0,feat1,feat2)
        #chain_example.Write()
    print('read NnetChainExample end')

