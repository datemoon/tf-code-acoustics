from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import os


class NnetBase(object):
    '''
    This is base class,the method provided must be implemented.
    '''
    def Loss(self, input, labels, seq_len):
        pass

    def CalculateLabelErrorRate(self, output, labels, mask, total_frames):
        pass

    def ReadNnetConf(self, nnet_conf):
        layer_conf=[]
        for line in open(nnet_conf,'r'):
            optlist = line.rstrip().replace(' ','').split(';')
            if len(optlist[0]) == 0 or optlist[0][0] == '#':
                continue
            opt_dict={}
            # A layer configue option.
            for opt in optlist:
                try:
                    if len(opt) == 0:
                        continue
                    key, val = opt.split('=')
                except ValueError:
                    print(opt+' it\'s error format.')
                    exit(1)
                opt_dict[key] = val
            # Add a layer conf in layer_conf.
            layer_conf.append(opt_dict)
        return layer_conf

