
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import os
import pickle
import time
import logging
import numpy as np

sys.path.append("../")

from io_func.kaldi_io_parallel import KaldiDataReadParallel

token_dict = {'<Nnet>': 1, '</Nnet>': 2, '<!EndOfComponent>': 3,
        '<Splice>': 102, '<AddShift>': 103, '<Rescale>' : 104,
        '<LearnRateCoef>': 11 }

def GetToken(key_str):
    try:
        return token_dict[key_str]
    except KeyError:
        return None

def ReadData(fp):
    for line in fp:
        line_list = line.rstrip().split()
        flag = 0
        data = []
        if line_list[0] == '<!EndOfComponent>':
            break
        for key in line_list:
            if key == '[':
                flag = 1
                continue
            elif key == ']':
                break
            if flag == 1:
                data.append(key)
#        print(data)
        return np.array(data, dtype=np.float32)

class AddShift(object):
    def __init__(self):
        self.data_ = None
        self.key_ = None
    
    def Propagate(self, input_data):
        return input_data + self.data_

    def Read(self, fp):
        self.data_ = ReadData(fp)

class Rescale(object):
    def __init__(self):
        self.data_ = None
        self.key_ = None

    def Propagate(self, input_data):
        return np.multiply(self.data_, input_data)

    def Read(self, fp):
        self.data_ = ReadData(fp)

class FeatureTransform(object):
    def __init__(self):
        self.trans_ = []

    def LoadTransform(self, file):
        fp = open(file,'r')
        for line in fp:
            line_list = line.rstrip().split()
            for key in line_list:
                if key[0] == '<':
                    if key == '<AddShift>':
                        ash = AddShift()
                        ash.Read(fp)
                        self.trans_.append(ash)
                    elif key == '<Rescale>':
                        rsl = Rescale()
                        rsl.Read(fp)
                        self.trans_.append(rsl)
                    elif key == '<Nnet>':
                        continue
                    elif key == '</Nnet>':
                        return self.trans_

    def Propagate(self, input_data):
        res = input_data
        for cal in self.trans_:
            res = cal.Propagate(res)
        return res

    def Print(self):
        for t in self.trans_:
            print(t.data_)

if __name__ == "__main__":
    file = sys.argv[1]
    feat_trans = FeatureTransform()
    feat_trans.LoadTransform(file)

    feat_trans.Print()

    readkaldi = KaldiDataReadParallel()
    readkaldi.scp_file = 'transdir/test.scp'
    readkaldi.reset_read()

    while True:
        key , value = readkaldi.read_next_utt()
        if key == '':
            break
        trans_res = feat_trans.Propagate(value)

        print(trans_res)
