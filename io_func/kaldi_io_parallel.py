# Copyright 2014    Yajie Miao    Carnegie Mellon University
#           2015    Yun Wang      Carnegie Mellon University

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
# WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
# MERCHANTABLITY OR NON-INFRINGEMENT.
# See the Apache 2 License for the specific language governing permissions and
# limitations under the License.

import gzip
import os
import sys, re
import glob
import struct

import numpy
#from model_io import log
from io_func import smart_open, preprocess_feature_and_label, shuffle_feature_and_label, make_context, skip_frame
import logging
sys.path.append("../")

from feat_process.feature_transform import FeatureTransform


class KaldiDataReadParallel(object):
    def __init__(self):
        self.scp_file = ''   # path to the .scp file
        self.label = ''
        self.max_input_seq_length = 1500
        self.lcxt = 0
        self.rcxt = 0
        self.batch_size = 1
        self.num_frames_batch = 20
        self.skip_frame = 1
        self.skip_offset = 0
        self.ali_provided = False
        self.scp_file_read = None
        # feature information
        self.original_feat_dim = 0
        self.feat_dim = 0
        self.alignment = {}
        self.random = False

        self.feature_transfile = None
        self.feature_transform = None

        # store features and labels for each data partition

    # read the alignment of all the utterances and keep the alignment in CPU memory.
    def read_alignment(self):
        f_read = smart_open(self.label, 'r')
        for line in f_read:
            line = line.replace('\n','').strip()
            if len(line) < 1: # this is an empty line, skip
                continue
            [utt_id, utt_ali] = line.split(' ', 1)
            # this utterance has empty alignment, skip
            if len(utt_ali) < 1:
                continue
            self.alignment[utt_id] = numpy.fromstring(utt_ali, dtype=numpy.int32, sep=' ')
        f_read.close()

    # read the feature matrix of the next utterance
    def read_next_utt(self):
#        self.scp_cur_pos = self.scp_file_read.tell()
        next_scp_line = self.scp_file_read.readline()
        if next_scp_line == '' or next_scp_line == None:    # we are reaching the end of one epoch
            return '', None
        utt_id, path_pos = next_scp_line.replace('\n','').split(' ')
        path, pos = path_pos.split(':')

        ark_read_buffer = smart_open(path, 'rb')
        ark_read_buffer.seek(int(pos),0)

        # now start to read the feature matrix into a numpy matrix
        header = struct.unpack('<xcccc', ark_read_buffer.read(5))
        if header[0] != "B":
            print "Input .ark file is not binary"; exit(1)

        rows = 0; cols= 0
        m, rows = struct.unpack('<bi', ark_read_buffer.read(5))
        n, cols = struct.unpack('<bi', ark_read_buffer.read(5))

        tmp_mat = numpy.frombuffer(ark_read_buffer.read(rows * cols * 4), dtype=numpy.float32)
        utt_mat = numpy.reshape(tmp_mat, (rows, cols))

        ark_read_buffer.close()

        return utt_id, utt_mat

    def is_finish(self):
        return self.end_reading

    def initialize(self, conf_dict, scp_file = None, label = None):
        for key in self.__dict__:
            if key in conf_dict.keys():
                self.__dict__[key] = conf_dict[key]
        if scp_file != None:
            self.scp_file = scp_file
        if label != None:
            self.label = label
        if not os.path.exists(self.scp_file):
            raise 'no scp file'
        if not os.path.exists(self.label):
            raise 'no label file'
        else:
            self.ali_provided = True


    def initialize_read(self, conf_dict = None, first_time_reading = False, scp_file = None, label = None):
        # first initial configure
        self.initialize(conf_dict, scp_file, label)

        if self.feature_transfile != None:
            self.feature_transform = FeatureTransform()
            self.feature_transform.LoadTransform(self.feature_transfile)

        self.scp_file_read = smart_open(self.scp_file, 'r')
        if first_time_reading:
            utt_id, utt_mat = self.read_next_utt()
            self.original_feat_dim = utt_mat.shape[1]
            self.scp_file_read.seek(0) 

            # compute the feature dimension
            self.feat_dim = (self.lcxt + 1 + self.rcxt) * self.original_feat_dim

            # load alignment file
            if self.ali_provided:
                self.read_alignment()

        self.end_reading = False
        return self.feat_dim

    def reset_read(self, reset_offset = True):
        self.scp_file_read = smart_open(self.scp_file, 'r')
        self.end_reading = False
        if reset_offset:
            self.skip_offset = (self.skip_offset+1)%self.skip_frame
        
    
    # load batch_size features and labels
    def load_next_nstreams(self):
        length = []
        feat_mat = []
        label = []
        nstreams = 0
        max_frame_num = 0
        while nstreams < self.batch_size:
            utt_id,utt_mat = self.read_next_utt()
            if utt_mat is None:
                self.end_reading = True
                break;
            if self.ali_provided and (self.alignment.has_key(utt_id) is False):
                continue
            
            if self.feature_transform != None:
                utt_mat = self.feature_transform.Propagate(utt_mat)

            # delete too length feature
            if (len(utt_mat)/self.skip_frame + 1) > self.max_input_seq_length:
                continue
            try:
                ali_utt = self.alignment[utt_id]
            except KeyError:
                logging.info('no '+ utt_id + ' align')
                continue
            if True: #(len(ali_utt) * 2 + 1) < len(utt_mat):
                label.append(self.alignment[utt_id])
                '''if self.read_opts['lcxt'] != 0 or self.read_opts['rcxt'] != 0:
                    feat_mat.append(make_context(utt_mat, self.read_opts['lcxt'], self.read_opts['rcxt']))
                else:
                    feat_mat.append(utt_mat)'''
                #feat_mat.append(skip_frame(make_context(utt_mat, self.lcxt, self.rcxt),self.skip_frame))
                feat_mat.append(skip_frame(utt_mat ,self.skip_frame, self.skip_offset))
                length.append(len(feat_mat[nstreams]))
            else:
                continue
            if max_frame_num < length[nstreams]:
                max_frame_num = length[nstreams]
            nstreams += 1

        # zero fill
        i = 0
        while i < nstreams:
            if max_frame_num != length[i]:
                feat_mat[i] = numpy.vstack((feat_mat[i], numpy.zeros((max_frame_num-length[i], feat_mat[i].shape[1]),dtype=numpy.float32)))
            i += 1

        if feat_mat.__len__():
            #return feat_mat , label , length
           # print('exchange')
            feat_mat_nstream = numpy.hstack(feat_mat).reshape(-1, nstreams, self.feat_dim)
            #feat_mat_nstream = numpy.vstack(feat_mat).reshape(nstreams, -1, self.original_feat_dim)
            np_length = numpy.vstack(length).reshape(-1)
            return feat_mat_nstream , label , np_length
        else:
            return None,None,None
    def __repr__(self):
        pri = '{\nKaldiDataReadParallel parameters:\n'
        for key in self.__dict__:
            if key != 'alignment':
                pri += key + ':\t' + str(self.__dict__[key]) +'\n'
        pri += '}'
        return pri

    def slice_load_next_nstreams(self):
        length = []
        feat_mat = []
        label = []
        nstreams = 0
        max_frame_num = 0
        while nstreams < self.batch_size:
            utt_id,utt_mat = self.read_next_utt()
            if utt_mat is None:
                self.end_reading = True
                break;
            if self.ali_provided and (self.alignment.has_key(utt_id) is False):
                continue
            
            if self.feature_transform != None:
                utt_mat = self.feature_transform.Propagate(utt_mat)
            
            # delete too length feature
            if (len(utt_mat)/self.skip_frame + 1) > self.max_input_seq_length:
                continue
            try:
                ali_utt = self.alignment[utt_id]
            except KeyError:
                logging.info('no '+ utt_id + ' align')
                continue
            if True: #(len(ali_utt) * 2 + 1) < len(utt_mat):
                label.append(self.alignment[utt_id])
                '''if self.read_opts['lcxt'] != 0 or self.read_opts['rcxt'] != 0:
                    feat_mat.append(make_context(utt_mat, self.read_opts['lcxt'], self.read_opts['rcxt']))
                else:
                    feat_mat.append(utt_mat)'''
                #feat_mat.append(skip_frame(make_context(utt_mat, self.lcxt, self.rcxt),self.skip_frame))
                feat_mat.append(skip_frame(utt_mat ,self.skip_frame, self.skip_offset))
                length.append(len(feat_mat[nstreams]))
            else:
                continue
            if max_frame_num < length[nstreams]:
                max_frame_num = length[nstreams]
            nstreams += 1

        if max_frame_num % self.num_frames_batch != 0:
            max_frame_num = self.num_frames_batch * (max_frame_num / self.num_frames_batch + 1)
        # zero fill
        i = 0
        while i < nstreams:
            if max_frame_num != length[i]:
                feat_mat[i] = numpy.vstack((feat_mat[i], numpy.zeros((max_frame_num-length[i], feat_mat[i].shape[1]),dtype=numpy.float32)))
            i += 1

        if feat_mat.__len__():
            #return feat_mat , label , length
           # print('exchange')
            feat_mat_nstream = numpy.hstack(feat_mat).reshape(-1, nstreams, self.feat_dim)
            #feat_mat_nstream = numpy.vstack(feat_mat).reshape(nstreams, -1, self.original_feat_dim)
            np_length = numpy.vstack(length).reshape(-1)
            array_feat = numpy.split(feat_mat_nstream, max_frame_num / self.num_frames_batch)
            array_label = []
            array_length = []
            for nbatch in range(max_frame_num / self.num_frames_batch):
                array_label.append([])
                tmp_length = []
                offset_n = nbatch * self.num_frames_batch
                # every sentence cut
                for i in range(len(label)):
                    tmp_label = []
                    j = 0
                    while j < self.num_frames_batch :
                        if j < len(label[i])-offset_n:
                            tmp_label.append(label[i][j+offset_n])
                        else:
                            tmp_label.append(0)
                        j += 1
                    if j < len(label[i])-offset_n: 
                        tmp_length.append(j)
                    else:
                        tmp_length.append(len(label[i])-offset_n)
                    array_label[nbatch].append(tmp_label)
                array_length.append(numpy.vstack(tmp_length).reshape(-1))
            return array_feat , array_label , array_length
        else:
            return None,None,None
