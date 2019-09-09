
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import os
import pickle
import time
import logging
import numpy as np

token_dict = {'<Nnet>': 1, '</Nnet>': 2, '<!EndOfComponent>': 3,
        '<Splice>': 102, '<AddShift>': 103, '<Rescale>' : 104,
        '<LearnRateCoef>': 11 }

def GetToken(key_str):
    try:
        return token_dict[key_str]
    except KeyError:
        return None

def ReadData(fp, dtype=np.float32):
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
        return np.array(data, dtype=dtype)

class Splice(object):
    def __init__(self, input_dim = None, output_dim = None):
        self.input_dim_ = input_dim
        self.output_dim_ = output_dim
        self.data_ = None
        self.key_ = None

    def GetTypeStr(self):
        return 'Splice'

    def GetSplice(self):
        return self.data_

    def Propagate(self, input_data):
        # now self.data_ must be intervalles 1
        for x in range(len(self.data_)-1):
            assert self.data_[x+1] - self.data_[x] == 1
        feature = input_data
        splice_feature = []
        for offset in self.data_:
            feat = []
            i = offset
            if i < 0:
                while True:
                    feat.append(feature[0])
                    i += 1
                    if i >= 0 :
                        break
                feat.append(feature[:offset])
            elif i == 0:
                feat.append(feature)
            elif i > 0:
                feat.append(feature[offset:])
                while True:
                    feat.append(feature[-1])
                    i -= 1
                    if i <= 0:
                        break
            splice_feature.append(np.vstack(feat))
        return np.hstack(splice_feature)

    def Read(self, fp):
        self.data_ = ReadData(fp, dtype=np.int32 )
    
    def GetOutDim(self):
        return self.output_dim_

    def GetInDim(self):
        return self.input_dim_

class AddShift(object):
    def __init__(self, input_dim = None, output_dim = None):
        self.input_dim_ = input_dim
        self.output_dim_ = output_dim
        self.data_ = None
        self.key_ = None
    
    def GetTypeStr(self):
        return 'AddShift'
    
    def Propagate(self, input_data):
        return input_data + self.data_

    def Read(self, fp):
        self.data_ = ReadData(fp)

    def GetOutDim(self):
        return len(self.data_)
    
    def GetInDim(self):
        return len(self.data_)

class Rescale(object):
    def __init__(self, input_dim = None, output_dim = None):
        self.input_dim_ = input_dim
        self.output_dim_ = output_dim
        self.data_ = None
        self.key_ = None
    
    def GetTypeStr(self):
        return 'Rescale'

    def Propagate(self, input_data):
        return np.multiply(self.data_, input_data)

    def Read(self, fp):
        self.data_ = ReadData(fp)
    
    def GetOutDim(self):
        return len(self.data_)
    
    def GetInDim(self):
        return len(self.data_)

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
                    elif key == '<Splice>':
                        spl = Splice(int(line_list[2]), int(line_list[1]))
                        spl.Read(fp)
                        self.trans_.append(spl)
                    elif key == '<Nnet>':
                        continue
                    elif key == '</Nnet>':
                        return self.trans_

    def Propagate(self, input_data):
        res = input_data
        for cal in self.trans_:
            res = cal.Propagate(res)
        return res

    def GetSplice(self):
        for cal in self.trans_:
            if cal.GetTypeStr() == 'Splice':
                return cal.GetSplice()
        return [0]

    def GetOutDim(self):
        return self.trans_[-1].GetOutDim()

    def GetInDim(self):
        return self.trans_[0].GetInDim()
    
    def Print(self):
        for t in self.trans_:
            print(t.data_)

if __name__ == "__main__":
    #file = sys.argv[1]
    #file = 'transdir/new_final.feature_transform'
    file = '/search/speech/hubo/git/tf-code-acoustics/train-data-7300h/final.feature_transform'
    feat_trans = FeatureTransform()
    feat_trans.LoadTransform(file)

    feat_trans.Print()
    fp = open('transdir/test.txt.ark','r')
    for line in fp:
        if '[' in line:
            mat = []
            for mat_line in fp:
                mat.append(mat_line.rstrip().split()[0:40])
                if ']' in mat_line:
                    break
            value = np.array(mat,dtype=np.float32)
            trans_res = feat_trans.Propagate(value)
            print(trans_res)

