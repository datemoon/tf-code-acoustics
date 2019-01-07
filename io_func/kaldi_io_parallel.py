# Copyright 2017/12/25  hubo

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
import random
import logging
import threading
import time

sys.path.append("../")
from io_func import smart_open, skip_frame
from feat_process.feature_transform import FeatureTransform

# read the alignment of all the utterances and keep the alignment in CPU memory.
def read_alignment(ali_file):
    alignment = {}
    f_read = smart_open(ali_file, 'r')
    for line in f_read:
        line = line.replace('\n','').strip()
        if len(line) < 1: # this is an empty line, skip
            continue
        [utt_id, utt_ali] = line.split(' ', 1)
        # this utterance has empty alignment, skip
        if len(utt_ali) < 1:
            continue
        alignment[utt_id] = numpy.fromstring(utt_ali, dtype=numpy.int32, sep=' ')
    f_read.close()
    return alignment

# read the feature matrix 
def read_next_utt(next_scp_line):
    # this shouldn't happen
    if next_scp_line == '' or next_scp_line == None:    # we are reaching the end of one epoch
        return '', None
        
    utt_id, path_pos = next_scp_line.replace('\n','').split(' ')
    path, pos = path_pos.split(':')
    
    ark_read_buffer = smart_open(path, 'rb')
    ark_read_buffer.seek(int(pos),0)

    # now start to read the feature matrix into a numpy matrix
    header = struct.unpack('<xcccc', ark_read_buffer.read(5))
    if header[0] != "B" and header[0] != b'B':
        print ("Input .ark file is not binary"); 
        exit(1)

    rows = 0; cols= 0
    m, rows = struct.unpack('<bi', ark_read_buffer.read(5))
    n, cols = struct.unpack('<bi', ark_read_buffer.read(5))

    tmp_mat = numpy.frombuffer(ark_read_buffer.read(rows * cols * 4), dtype=numpy.float32)
    utt_mat = numpy.reshape(tmp_mat, (rows, cols))

    ark_read_buffer.close()

    return utt_id, utt_mat

def PackageFeatAndAli(all_package, input_lock, package_end, scp_file, ali_file, nstreams, skip_frame = 1,  max_input_seq_length = 1500, criterion = 'ce'):
    logging.info('------start package------')
    start_package = time.time()
    #all_package = []
    # first read ali
    alignment_dict = read_alignment(ali_file)

    scp_list = []
    ali_list = []
    # second read feature scp file and package feature and ali
    for line in open(scp_file, 'r'):
        utt_id, utt_mat = read_next_utt(line)
        logging.debug(utt_id + ' read ok.')
        # overlength
        if int(len(utt_mat)/skip_frame) + 1 > max_input_seq_length:
            continue

        try:
            ali_utt = alignment_dict[utt_id]
        except KeyError:
            logging.info('no '+ utt_id + ' align')
            continue
        if 'ce' in criterion:
            if len(utt_mat) != len(ali_utt):
                # delete one frame ali in utt_mat
                if len(ali_utt) - len(utt_mat) == 1:
                    #logging.warn(utt_id + ' feat length + 1 = ali length , and delete one ali label')
                    ali_utt = numpy.delete(ali_utt, -1)
                else:
                    logging.info(utt_id + ' feat and ali isn\'t equal length')
                    continue
        elif 'ctc' in criterion :
            if len(utt_mat) < len(ali_utt) * 2 - 1:
                logging.info(utt_id + ' feat < ali * 2 - 1 :%d < %d * 2 - 1' % (len(utt_mat), len(ali_utt)))
                continue
        scp_list.append(line)
        ali_list.append(ali_utt)
        if len(scp_list) == nstreams:
            input_lock.acquire()
            all_package.append([scp_list, ali_list])
            input_lock.release()
            scp_list = []
            ali_list = []
    if len(scp_list) != 0:
        while len(scp_list) < nstreams:
            scp_list.append(scp_list[0])
            ali_list.append(ali_list[0])
        input_lock.acquire()
        all_package.append([scp_list, ali_list])
        package_end.append(True)
        input_lock.release()

    end_package = time.time()
    logging.info('------PackageFeatAndAli end. Package time is : %f s' % (end_package - start_package))
    return True

class KaldiDataReadParallel(object):
    '''
    kaldi i.
    max_input_seq_length:allow input max input length
    batch_size:Number of streams in the Multi-stream training
    num_frames_batch:Length of 'one stream' in the Multi-stream training
    skip_frame:skip frame number
    skip_offset:skip_offset
    shuffle:shuffle data
    '''
    def __init__(self):
        self.max_input_seq_length = 1500
        self.batch_size = 1
        self.num_frames_batch = 20
        self.skip_frame = 1
        self.skip_offset = 0
        self.shuffle = False
        
    def Initialize(self, conf_dict = None, scp_file = None, label = None, feature_transform = None, criterion = 'ce'):
        for key in self.__dict__:
            if key in conf_dict.keys():
                self.__dict__[key] = conf_dict[key]
        self.scp_file = ''   # path to the .scp file
        self.label = ''
        self.criterion = criterion
        if scp_file != None:
            self.scp_file = scp_file
        if label != None:
            self.label = label
        if not os.path.exists(self.scp_file):
            raise 'no scp file'
        if not os.path.exists(self.label):
            raise 'no label file'
        
        # feature information
        self.input_feat_dim = 0
        self.output_feat_dim = 0

        # first read scp and ali to self.package_feat_ali 
        # store features and labels for each data partition
        self.package_feat_ali = []  # save format is [scp_line_list, ali_list]
        self.read_offset = 0


        self.feature_transform = None
        # read feature transform parameter
        if feature_transform != None:
            self.feature_transform = feature_transform
            self.input_feat_dim = self.feature_transform.GetInDim()
            self.output_feat_dim = self.feature_transform.GetOutDim()
            self.output_dim = self.output_feat_dim
            # cnn must be in nnet the first layer!!!
            if 'cnn' in self.criterion:
                self.output_feat_dim = [int(self.output_feat_dim/self.input_feat_dim), self.input_feat_dim]
        else:
            logging.info('no feature transform file.')
            sys.exit(1)
        # prepare data, The first loop train must be order for add speed.
        self.input_lock = threading.Lock()

        self.ThreadPackageFeatAndAli()
        #self.package_feat_ali = 
        #start_package = time.time()
        #PackageFeatAndAli(self.package_feat_ali, self.input_lock, self.scp_file, self.label, self.batch_size, self.skip_frame, self.max_input_seq_length, self.criterion)
        #end_package = time.time()
        #logging.info('package time is : %f s' % (end_package - start_package))
        
        #if self.shuffle is True:
        #    random.shuffle(self.package_feat_ali)
        if 'ce' in self.criterion or 'whole' in self.criterion:
            self.do_skip_lab = True
        elif 'ctc' in self.criterion:
            self.do_skip_lab = False
        return self.output_feat_dim

    def ThreadPackageFeatAndAli(self):
        self.package_feat_ali = []
        self.package_end=[False]
        load_thread = threading.Thread(group=None, target=PackageFeatAndAli,
                args=(self.package_feat_ali, self.input_lock, self.package_end, 
                    self.scp_file, self.label, 
                    self.batch_size, self.skip_frame, 
                    self.max_input_seq_length, self.criterion,),
                kwargs={}, name='PackageFeatAndAli_thread')
        load_thread.start()
        logging.info('PackageFeatAndAli thread start.')

    def Reset(self, shuffle = False, skip_offset = 0 ):
        self.skip_offset = skip_offset % self.skip_frame
        self.read_offset = 0
        if shuffle is True or self.shuffle is True:
            self.shuffle = True
            self.input_lock.acquire()
            if self.package_end[-1] is True:
                random.shuffle(self.package_feat_ali)
                logging.info('Reset and shuffle package_feat_ali')
                self.input_lock.release()
                return 
            self.input_lock.release()
        logging.info('Reset and no shuffle package_feat_ali')

    def LoadOnePackage(self):
        while True:
            self.input_lock.acquire()
            if self.read_offset >= len(self.package_feat_ali):
                if self.package_end[-1] is True:
                    self.input_lock.release()
                    return None, None, None, None
                else:
                    self.input_lock.release()
                    continue
            else:
                self.input_lock.release()
                break

        package = self.package_feat_ali[self.read_offset]
        self.read_offset += 1
        
        feat_scp = package[0]
        label = package[1]
        max_frame_num = 0
        length = []
        feat_mat = []
        
        for feat_line in feat_scp:
            utt_id, utt_mat = read_next_utt(feat_line)
            # do feature transform
            if self.feature_transform != None:
                utt_mat = self.feature_transform.Propagate(utt_mat)
                feat_mat.append(
                        skip_frame(utt_mat ,self.skip_frame, self.skip_offset))
            length.append(len(feat_mat[-1]))
            if max_frame_num < length[-1]:
                max_frame_num = length[-1]

        if self.do_skip_lab and self.skip_frame > 1:
            process_lab = []
            for lab in label:
           #     if self.skip_frame > 1:
                process_lab.append(lab[self.skip_offset : len(lab) : self.skip_frame])
            label = process_lab

        return feat_mat, label, length, max_frame_num
    
    # load batch frames features train.The order it's not important.
    def LoadBatch(self):
        if 'cnn' in self.criterion:
            if 'whole' in self.criterion or 'ctc' in self.criterion:
                return self.CnnLoadNextNstreams()
            else:
                return self.CnnSliceLoadNextNstreams()
        else:
            if 'whole' in self.criterion or 'ctc' in self.criterion:
                return self.WholeLoadNextNstreams()
            else:
                return self.SliceLoadNextNstreams()
    
    # Cnn frames features train.
    # slice load
    def CnnSliceLoadNextNstreams(self):
        array_feat , array_label , array_length = self.SliceLoadNextNstreams()
        if array_feat is None:
            return None, None, None
        outdim = self.feature_transform.GetOutDim()
        inputdim = self.feature_transform.GetInDim()
        splicedim = int(outdim / inputdim)
        cnn_feat = []
        for feat in array_feat:
            # -1 it's frames, splicedim it's time, inputdim it's feature dim
            cnn_feat.append(feat.reshape(-1, splicedim, inputdim, 1))
        assert len(cnn_feat) == numpy.shape(array_feat)[0]
        return numpy.array(cnn_feat), array_label, array_length

    # whole sentence load
    def CnnLoadNextNstreams(self):
        if 'whole' in self.criterion:
            feat , label , length = self.WholeLoadNextNstreams()
        else:
            feat , label , length = self.LoadNextNstreams()
        #print(numpy.shape(feat),numpy.shape(label), numpy.shape(length))
        if feat is None:
            return None, None, None
        outdim = self.feature_transform.GetOutDim()
        inputdim = self.feature_transform.GetInDim()
        splicedim = int(outdim / inputdim)
        return feat.reshape(-1, splicedim, inputdim, 1), label, length

    # load batch_size features and labels, it's whole sentence train.
    def LoadNextNstreams(self):
        feat_mat, label, length, max_frame_num = self.LoadOnePackage()
        if feat_mat is None:
            return None, None, None
        # zero fill
        i = 0
        while i < self.batch_size:
            if max_frame_num != length[i]:
                feat_mat[i] = numpy.vstack((feat_mat[i], numpy.zeros((max_frame_num-length[i], feat_mat[i].shape[1]),dtype=numpy.float32)))
            i += 1
        
        if feat_mat.__len__() == self.batch_size:
            feat_mat_nstream = numpy.hstack(feat_mat).reshape(-1, self.batch_size, self.output_dim)
            np_length = numpy.vstack(length).reshape(-1)
            return feat_mat_nstream , label , np_length
        else:
            logging.info('It\'s shouldn\'t happen. feat is less then batch_size.')
            return None, None, None

    # load batch_size features and labels, it's whole sentence train.
    def WholeLoadNextNstreams(self):
        feat, label, length = self.LoadNextNstreams()
        if feat is None:
            return None, None, None
        max_frame_num = numpy.shape(feat)[0]
        nsent = 0
        while nsent < numpy.shape(label)[0]:
            numzeros = max_frame_num - numpy.shape(label[nsent])[0]
            if numzeros != 0:
                label[nsent] = numpy.hstack((label[nsent], 
                    numpy.zeros((numzeros), dtype=numpy.float32)))
            nsent += 1
        return feat, label, length

    # load batch size features and labels,it's ce train, cut sentence.
    def SliceLoadNextNstreams(self):
        feat_mat, label, length, max_frame_num = self.LoadOnePackage()
        if feat_mat is None:
            return None, None, None
        
        if max_frame_num % self.num_frames_batch != 0:
            max_frame_num = self.num_frames_batch * (int(max_frame_num / self.num_frames_batch) + 1)

        # zero fill
        i = 0
        while i < self.batch_size:
            if max_frame_num != length[i]:
                feat_mat[i] = numpy.vstack((feat_mat[i], numpy.zeros((max_frame_num-length[i], feat_mat[i].shape[1]),dtype=numpy.float32)))
            i += 1
        if feat_mat.__len__() == self.batch_size:
            feat_mat_nstream = numpy.hstack(feat_mat).reshape(-1, self.batch_size, self.output_dim)
            np_length = numpy.vstack(length).reshape(-1)
            array_feat = numpy.split(feat_mat_nstream, int(max_frame_num / self.num_frames_batch))
            array_label = []
            array_length = []
            for nbatch in range(int(max_frame_num / self.num_frames_batch)):
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
            logging.info('It\'s shouldn\'t happen. feat is less then batch_size.')
            return None, None, None
            
    # print info
    def __repr__(self):
        pri = '{\nKaldiDataReadParallel parameters:\n'
        for key in self.__dict__:
            if key != 'package_feat_ali':
                pri += key + ':\t' + str(self.__dict__[key]) +'\n'
            else:
                pri += key + ':\t' + 'too big not print' + '\n'
        pri += '}'
        return pri

if __name__ == '__main__':
    conf_dict = { 'batch_size' :100,
            'skip_frame':3,
            'skip_offset': 0,
            'do_skip_lab': True,
            'shuffle': False}
    path = '/search/speech/hubo/git/tf-code-acoustics/train-data'
    feat_trans_file = '/search/speech/hubo/git/tf-code-acoustics/feat_process/transdir/1_final.feature_transform'
    feat_trans = FeatureTransform()
    feat_trans.LoadTransform(feat_trans_file)

    logging.basicConfig(filename = 'test.log')
    logging.getLogger().setLevel('INFO')
    io_read = KaldiDataReadParallel()
    io_read.Initialize(conf_dict, scp_file=path+'/abc.scp',
            label = path+'/merge_sort_cv.labels',
            feature_transform = feat_trans, criterion = 'whole')

            #label = path+'/sort_tr.labels.4026.ce',
    start = time.time()
    while True:
        #feat_mat, label, length = io_read.LoadNextNstreams()
        feat_mat, label, length = io_read.CnnLoadNextNstreams()
        logging.info(str(numpy.shape(feat_mat))+str(numpy.shape(label))+str(numpy.shape(length)))
        #feat_mat, label, length = io_read.SliceLoadNextNstreams()
        #print(numpy.shape(feat_mat),numpy.shape(label),numpy.shape(length))
        if feat_mat is None:
            break
    
    io_read.Reset(shuffle = True)
    while True:
        #feat_mat, label, length = io_read.LoadNextNstreams()
        feat_mat, label, length = io_read.CnnLoadNextNstreams()
        logging.info(str(numpy.shape(feat_mat))+str(numpy.shape(label))+str(numpy.shape(length)))
        #feat_mat, label, length = io_read.SliceLoadNextNstreams()
        #print(numpy.shape(feat_mat),numpy.shape(label),numpy.shape(length))
        if feat_mat is None:
            break

    end = time.time()
    logging.info('load time is : %f s' % (end - start))

