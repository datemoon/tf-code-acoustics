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
import multiprocessing 
import ctypes
import time
try:
    import queue as Queue
except ImportError:
    import Queue


sys.path.extend(["../","./"])
from io_func import smart_open, skip_frame, sparse_tuple_from
from feat_process.feature_transform import FeatureTransform
from io_func.matio import read_next_utt
from fst import *
from io_func.kaldi_io_egs import NnetChainExample,ProcessEgsFeat


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
def read_nocompression_next_utt(next_scp_line):
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

def ReadScp(scp_file):
    scp_dict = {}
    f_read = smart_open(scp_file, 'r')
    for line in f_read:
        line = line.replace('\n','').strip()
        if len(line) < 1: # this is an empty line, skip
            continue
        [utt_id, utt_val] = line.split()
        # this utterance has empty alignment, skip
        scp_dict[utt_id] = line
    f_read.close()
    return scp_dict 

def PackageFeatAndAliAndLat(all_package, input_lock, package_end, feat_scp_file, ali_file, lat_scp_file, nstreams, 
        skip_frame = 1,  max_input_seq_length = 1500, criterion = 'mmi'):
    logging.info('------start PackageFeatAndAliAndLat------')
    start_package = time.time()
    # first read ali
    alignment_dict = read_alignment(ali_file)
    lat_dict = ReadScp(lat_scp_file)

    feat_list = []
    ali_list = []
    lat_list = []

    with open(feat_scp_file, 'r') as feat_fp:
        for line in feat_fp:
            utt_id, utt_mat = read_next_utt(line)
            logging.debug(utt_id + ' read ok.')
            if int(len(utt_mat)/skip_frame) + 1 > max_input_seq_length:
                logging.info(utt_id + ' length '+ str(int(len(utt_mat)/skip_frame)+1) + ' > ' + str(max_input_seq_length))
                continue
            try:
                ali_utt = alignment_dict[utt_id]
            except KeyError:
                logging.info('no '+ utt_id + ' align')
                continue
            try:
                lat_scp_line = lat_dict[utt_id]
            except KeyError:
                logging.info('no '+ utt_id + ' lattice')
                continue

            # should check length is equal.
            # but because of time question , now it's not check length.
            feat_list.append(line)
            ali_list.append(ali_utt)
            lat_list.append(lat_scp_line)
            
            if len(feat_list) == nstreams:
                input_lock.acquire()
                all_package.append([feat_list, ali_list, lat_list])
                input_lock.release()
                feat_list = []
                ali_list = []
                lat_list = []
    
    if len(feat_list) != 0:
        while len(feat_list) < nstreams:
            feat_list.append(feat_list[0])
            ali_list.append(ali_list[0])
            lat_list.append(lat_list[0])
        input_lock.acquire()
        all_package.append([feat_list, ali_list, lat_list])
        input_lock.release()

    input_lock.acquire()
    package_end.append(True)
    input_lock.release()
    
    end_package = time.time()
    logging.info('------PackageFeatAndAliAndLat end. Package time is : %f s, batch number : %d' % (end_package - start_package, len(all_package)))


def PackageFeatAndAli(all_package, input_lock, package_end, scp_file, ali_file, nstreams, skip_frame = 1,  max_input_seq_length = 1500, criterion = 'ce'):
    logging.info('------start PackageFeatAndAli------')
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
            logging.info(utt_id + ' length '+ str(int(len(utt_mat)/skip_frame)+1) + ' > ' + str(max_input_seq_length))
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
        input_lock.release()
    
    input_lock.acquire()
    package_end.append(True)
    input_lock.release()

    end_package = time.time()
    logging.info('------PackageFeatAndAli end. Package time is : %f s, batch number : %d' % (end_package - start_package, len(all_package)))
    return True

def PackageEgs(all_package, input_lock, package_end, scp_file, nstreams):
    logging.info('------start PackageEgs------')
    start_package = time.time()
    #all_package = []
    scp_list = []
    # first read egs scp file and package 
    for line in open(scp_file, 'r'):
        scp_list.append(line)
        #print(len(scp_list),nstreams)
        if len(scp_list) == nstreams:
            input_lock.acquire()
            all_package.append([scp_list])
            input_lock.release()
            scp_list = []

    if len(scp_list) != 0:
        input_lock.acquire()
        all_package.append([scp_list])
        input_lock.release()
    
    input_lock.acquire()
    package_end.append(True)
    input_lock.release()

    end_package = time.time()
    logging.info('------PackageFeatAndAli end. Package time is : %f s, batch number : %d' % (end_package - start_package, len(all_package)))
    #print("********************")
    return True

class KaldiDataReadParallel(object):
    '''
    kaldi i.
    max_input_seq_length :allow input max input length
    batch_size           :Number of streams in the Multi-stream training
    num_frames_batch     :Length of 'one stream' in the Multi-stream training
    skip_frame           :skip frame number
    skip_offset          :skip_offset
    shuffle              :shuffle data
    '''
    def __init__(self):
        # config
        self.max_input_seq_length = 1500
        self.batch_size = 10
        self.num_frames_batch = 20
        self.overlap=0
        self.skip_frame = 1
        self.skip_offset = 0
        self.shuffle = False
        # tdnn parameters
        self.tdnn_start_frames = 0
        self.tdnn_end_frames = 0

        #
        self.queue_cache = 100
        self.io_thread_num = 1
        self.io_end_times = 0  #if self.io_end_times == self.io_thread_num,it's end
        self.scp_file = None   # path to the .scp file
        self.label = None

        self.lat_scp_file = None
        self.ali_map_file = None
        self.ali_to_pdf_phone = None
        self.class_frame_counts = None
        
        self.criterion = None
        self.feature_transform = None

        # self.egs_dict = { key1: index, ...}
        self.egs_dict = {}
        # self.egs_queue = [[queue, ...],[size, ...]]
        self.egs_queue = [[],[]]
        self.max_egs_kind = 5

        self.input_thread = []
        
    def Initialize(self, conf_dict = None, scp_file = None, label = None, feature_transform = None, criterion = None, lat_scp_file = None):
        for key in self.__dict__:
            if key in conf_dict.keys():
                self.__dict__[key] = conf_dict[key]
        
        if scp_file is not None:
            self.scp_file = scp_file
        if label is not None:
            self.label = label
        if lat_scp_file is not None:
            self.lat_scp_file = lat_scp_file
        if criterion is not None:
            self.criterion = criterion
        if self.ali_map_file is not None:
            # load ali_map_file
            self.ali_to_pdf_phone = LoadMapPdfAndPhone(self.ali_map_file)
            # get pdf to phone list
            # mfpe and smbr used
            self.pdf_to_phone = GetPdfToPhoneList(self.ali_to_pdf_phone)
            # get PdfPrior
            if self.class_frame_counts is None:
                self.pdf_prior =None
            else:
                self.pdf_prior = PdfPrior(self.class_frame_counts)

        
        if not os.path.exists(self.scp_file):
            raise 'no scp file'
        #if not os.path.exists(self.label):
        #    raise 'no label file'
        # feature information
        self.input_feat_dim = 0
        self.output_feat_dim = 0

        # first read scp and ali to self.package_feat_ali 
        # store features and labels for each data partition
        self.package_feat_ali = []  # save format is [scp_line_list, ali_list]

        # Initial input queue.
        self.input_queue = multiprocessing.Queue(self.queue_cache)

        # shared memery
        if 'chain' in self.criterion:
            # self.egs_dict = { key1: index, ...}
            self.egs_dict = multiprocessing.Manager().dict()
            #self.max_egs_kind = multiprocessing.Value(ctypes.c_int, self.max_egs_kind, lock=True)
            # self.egs_queue = [[queue, ...],[size, ...]]
            for i in range(self.max_egs_kind):
                self.egs_queue[0].append(multiprocessing.Value(ctypes.c_int, 0, lock=True))
                self.egs_queue[1].append(multiprocessing.Queue(maxsize=0))
        
        self.package_feat_ali = multiprocessing.Manager().list([])
        self.read_offset = multiprocessing.Value(ctypes.c_int, 0, lock=True)
        self.package_end = multiprocessing.Manager().list([False])

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
        
        if 'ctc' in self.criterion:
            self.do_skip_lab = False
        else:
            self.do_skip_lab = True
        
        # prepare data, The first loop train must be order for add speed.
        #self.input_lock = threading.Lock()
        self.input_lock = multiprocessing.Lock()

        self.ThreadPackageFeatAndAli()
        
        # multiprocessing package input 
        self.ThreadPackageInput()
        #if self.shuffle is True:
        #    random.shuffle(self.package_feat_ali)
        return self.output_feat_dim
    

    # package input feats.scp, label and lattice.
    # save index to self.package_feat_ali
    def ThreadPackageFeatAndAli(self):
        if 'chain' in self.criterion:
            load_thread = threading.Thread(group=None, target=PackageEgs,
                    args=(self.package_feat_ali, self.input_lock, self.package_end,
                        self.scp_file, self.batch_size),
                    kwargs={}, name='PackageEgs_thread')
            logging.info('PackageEgs thread start.')

        elif self.lat_scp_file is None and 'mmi' not in self.criterion: 
            load_thread = threading.Thread(group=None, target=PackageFeatAndAli,
                    args=(self.package_feat_ali, self.input_lock, self.package_end, 
                        self.scp_file, self.label, 
                        self.batch_size, self.skip_frame, 
                        self.max_input_seq_length, self.criterion,),
                    kwargs={}, name='PackageFeatAndAli_thread')
            logging.info('PackageFeatAndAli thread start.')

        else:
            load_thread = threading.Thread(group=None, target=PackageFeatAndAliAndLat,
                    args=(self.package_feat_ali, self.input_lock, self.package_end,
                        self.scp_file, self.label, self.lat_scp_file,
                        self.batch_size, self.skip_frame,
                        self.max_input_seq_length, self.criterion,),
                    kwargs={}, name='PackageFeatAndAliAndLat_thread')
            logging.info('PackageFeatAndAliAndLat thread start.')

        load_thread.start()

        # if you want shuffle data , you must be wait load all data
        if self.shuffle is True:
            logging.info('Wait Package thread end and shuffle package')
            load_thread.join()
            assert self.package_end[-1] is True
            logging.info('Shuffle package_feat_ali')
            random.shuffle(self.package_feat_ali)


    def Reset(self, shuffle = False, skip_offset = 0 ):
        if len(self.input_thread) == 0:
            self.skip_offset = skip_offset % self.skip_frame
            self.read_offset.value = 0
            self.io_end_times = 0
            self.ThreadPackageInput()
        logging.info('self.skip_offset:%d, self.read_offset:%d' %(self.skip_offset, self.read_offset.value))
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

    def PackBatchEgs(self):
        name_list = []
        feat_mat = []
        fst_list = []
        deriv_weights_list = []
        osize = 0
        self.input_lock.acquire()
        egs_len = len(self.egs_queue[0])
        for i in range(egs_len):
            if self.egs_queue[0][i].value >= self.batch_size:
                #print("package one batch")
                for n in range(self.batch_size):
                    [name, feat, ofst, osize, deriv_weights] = self.egs_queue[1][i].get()
                    name_list.append(name)
                    feat_mat.append(feat)
                    fst_list.append(ofst)
                    deriv_weights_list.append(deriv_weights)
                self.egs_queue[0][i].value -= self.batch_size
                self.input_lock.release()
                max_frame_num = len(feat_mat[0])
                valid_length = osize
                fst_list = PackageFst(fst_list)
                return feat_mat, deriv_weights_list, valid_length, max_frame_num, fst_list

        self.input_lock.release()
        return None

    # load chain egs
    def LoadOnePackageEgs(self):
        while True:
            self.input_lock.acquire()
            if self.read_offset.value >= len(self.package_feat_ali):
                self.input_lock.release()
                packfst = self.PackBatchEgs()
                if packfst is not None:
                    return packfst
                self.input_lock.acquire()
                if self.package_end[-1] is True:
                    self.input_lock.release()
                    return None, None, None, None, None
                else:
                    self.input_lock.release()
                    time.sleep(0.01)
                    continue
            else:
                package = self.package_feat_ali[self.read_offset.value]
                self.read_offset.value += 1
                self.input_lock.release()
                
                # read one batch egs
                egs_scp = package[0]
                splice_info = self.feature_transform.GetSplice()
                for scp_line in egs_scp:
                    chain_example = NnetChainExample()
                    chain_example.ReadScp(scp_line)
                    # process input features
                    name = chain_example.GetKey()
                    inputs = chain_example.Input()
                    outputs = chain_example.Output()
                    for iput,oput in zip(inputs,outputs):
                        feat = iput.GetFeat()
                        isize = iput.GetSize()
                        # feature_transform
                        feat = self.feature_transform.Propagate(feat)
                        assert isize == np.shape(feat)[0]
                        #  skip frame  
                        feat = ProcessEgsFeat(feat, iput.GetIndex(), oput.GetIndex(), 
                                self.feature_transform.GetSplice(), self.skip_offset)
                        
                        ofst = oput.GetFst()
                        osize = oput.GetSize()

                        deriv_weights = oput.GetDerivWeights()
                        
                        def EgsKey(isize, osize):
                            return str(isize) + '-' + str(osize)
                        egskey = EgsKey(isize, osize)
                        #print('name',name)
                        self.input_lock.acquire()
                        if egskey in self.egs_dict.keys():
                            index = self.egs_dict[egskey]
                            self.egs_queue[0][index].value += 1
                            self.egs_queue[1][index].put([name, feat, ofst, osize, deriv_weights])
                            self.input_lock.release()
                        else:
                            index = len(self.egs_dict.keys())
                            self.egs_dict[egskey] = index
                            assert index < self.max_egs_kind
                            self.egs_queue[0][index].value += 1
                            self.egs_queue[1][index].put([name, feat, ofst, osize, deriv_weights])

                            self.input_lock.release()
                    # end one NnetChainExample

                packfst = self.PackBatchEgs()
                if packfst is not None:
                    # end one batch egs scp
                    return packfst
                else:
                    continue
                # continue next bacth egs scp

    def LoadOnePackage(self):
        while True:
            self.input_lock.acquire()
            if self.read_offset.value >= len(self.package_feat_ali):
                if self.package_end[-1] is True:
                    self.input_lock.release()
                    return None, None, None, None, None
                else:
                    self.input_lock.release()
                    time.sleep(0.05)
                    continue
            else:
                package = self.package_feat_ali[self.read_offset.value]
                self.read_offset.value += 1
                self.input_lock.release()
                break

        feat_scp = package[0]
        label = package[1]
        lat_scp = None
        lat_list = None
        if len(package) == 3:
            lat_scp = package[2]
            # indexs_info_list, pdf_values_list, lmweight_values_list, amweight_values_list, statesinfo_list, statenum_list, time_list
            lat_list = PackageLattice(lat_scp, map_pdf_phone = self.ali_to_pdf_phone)

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


        # check lattice and label and feat
        if lat_list is not None:
            i = 0
            lat_times = lat_list[-1]
            while i < len(lat_times):
                assert lat_times[i] == length[i]
                i += 1
    
        return feat_mat, label, length, max_frame_num, lat_list
    
    # load batch frames features train.The order it's not important.
    def LoadBatch(self):
        while True:
            if 'cnn' in self.criterion:
                if 'whole' in self.criterion or 'ctc' in self.criterion:
                    feat,label,length,lattice = self.CnnLoadNextNstreams()
                else:
                    feat,label,length,lattice = self.CnnSliceLoadNextNstreams()
            elif 'tdnn' in self.criterion:
                feat,label,length,lattice = self.TdnnLoadNextNstreams()
            elif 'chain' in self.criterion:
                feat,label,length,lattice = self.ChainLoadNextNstreams()
            else:
                if 'whole' in self.criterion or 'ctc' in self.criterion:
                    feat,label,length,lattice = self.WholeLoadNextNstreams()
                else:
                    feat,label,length,lattice = self.SliceLoadNextNstreams()
            if label is not None:
                if 'ctc' in self.criterion:
                    label = sparse_tuple_from(label)
            self.input_queue.put((feat,label,length,lattice))
            
            if feat is not None:
                print(numpy.shape(feat))
            else:
                print(feat)
            if feat is None:
                break
        print('end LoadBatch')

    # because efficiency, so should use multiprocessing
    def ThreadPackageInput(self):
        self.input_thread = []
        for i in range(self.io_thread_num):
            self.input_thread.append(multiprocessing.Process(group=None, target=self.LoadBatch,
                args=(),name='read_thread'+str(i)))
            #self.input_thread.append(threading.Thread(group=None, target=self.LoadBatch,
            #    args=(),name='read_thread'+str(i)))
        for thr in self.input_thread:
            logging.info('ProcessPackageInput start')
            thr.start()
    
    def ClearEgsQueue(self):
        self.input_lock.acquire()
        #for i in range(len(self.egs_queue[0])):
        for i in range(len(self.egs_queue[0])):
            for n in range(self.egs_queue[0][i].value):
                self.egs_queue[1][i].get()
            self.egs_queue[0][i].value = 0
        self.input_lock.release()
    
    # must be join io thread
    def JoinInput(self):
        self.ClearEgsQueue()
        for i in range(self.io_thread_num):
            logging.info('ProcessPackageInput end join')
            self.input_thread[i].join()
        self.input_thread = []


    def GetInput(self):
        # if end
        while True:
            feat,label,length,lattice = self.input_queue.get()
            if feat is None:
                self.io_end_times += 1
                # end
                if self.io_end_times == self.io_thread_num:
                    return [None, None, None, None]
                else:
                    continue
            else:
                return [feat,label,length,lattice]

    def ReadEnd(self):
        if self.io_end_times == self.io_thread_num:
            return True
        return False
    
    # Tdnn frames features train.
    def TdnnLoadNextNstreams(self):
        feat , label , length, lat_list = self.WholeLoadNextNstreams()
        if feat is None:
            return None, None, None, None
                    
        outdim = self.feature_transform.GetOutDim()
        inputdim = self.feature_transform.GetInDim()
        head = feat[:1]
        tail = feat[-1:]
        # add start frames
        i = 0
        while i < self.tdnn_start_frames:
            feat = numpy.vstack((head, feat))
            i += 1
        # add end frames
        i = 0
        while i < self.tdnn_end_frames:
            feat = numpy.vstack((feat, tail))
            i += 1
        return feat , label , length, lat_list

    # Cnn frames features train.
    # slice load
    def CnnSliceLoadNextNstreams(self):
        array_feat , array_label , array_length, lat_list = self.SliceLoadNextNstreams()
        if array_feat is None:
            return None, None, None, None
        outdim = self.feature_transform.GetOutDim()
        inputdim = self.feature_transform.GetInDim()
        splicedim = int(outdim / inputdim)
        cnn_feat = []
        for feat in array_feat:
            # -1 it's frames, splicedim it's time, inputdim it's feature dim
            cnn_feat.append(feat.reshape(-1, splicedim, inputdim, 1))
        assert len(cnn_feat) == numpy.shape(array_feat)[0]
        return numpy.array(cnn_feat), array_label, array_length, lat_list

    # whole sentence load
    def CnnLoadNextNstreams(self):
        if 'whole' in self.criterion:
            feat , label , length, lat_list = self.WholeLoadNextNstreams()
        else:
            feat , label , length, lat_list = self.LoadNextNstreams()
        #print(numpy.shape(feat),numpy.shape(label), numpy.shape(length))
        if feat is None:
            return None, None, None, None
        outdim = self.feature_transform.GetOutDim()
        inputdim = self.feature_transform.GetInDim()
        splicedim = int(outdim / inputdim)
        return feat.reshape(-1, splicedim, inputdim, 1), label, length, lat_list

    # load chain egs batch data
    def ChainLoadNextNstreams(self):
        feat_mat, deriv_weights_list, valid_length, max_frame_num, fst_list = self.LoadOnePackageEgs()

        if feat_mat is None:
            return None, None, None, None
        if feat_mat.__len__() == self.batch_size:
            feat_mat_nstream = numpy.hstack(feat_mat).reshape(-1, self.batch_size, self.output_dim)
            # feature, deriv_weights, valid_length, fst_list
            return feat_mat_nstream , deriv_weights_list, valid_length, fst_list
        else:
            return None, None, None, None


    # load batch_size features and labels, it's whole sentence train.
    def LoadNextNstreams(self):
        feat_mat, label, length, max_frame_num , lat_list = self.LoadOnePackage()
        if feat_mat is None:
            return None, None, None, None
        # zero fill
        i = 0
        while i < self.batch_size:
            if max_frame_num != length[i]:
                feat_mat[i] = numpy.vstack((feat_mat[i], numpy.zeros((max_frame_num-length[i], feat_mat[i].shape[1]),dtype=numpy.float32)))
            i += 1
        
        if feat_mat.__len__() == self.batch_size:
            feat_mat_nstream = numpy.hstack(feat_mat).reshape(-1, self.batch_size, self.output_dim)
            np_length = numpy.vstack(length).reshape(-1)
            return feat_mat_nstream , label , np_length, lat_list
        else:
            logging.info('It\'s shouldn\'t happen. feat is less then batch_size.')
            return None, None, None, None

    # load batch_size features and labels, it's whole sentence train.
    def WholeLoadNextNstreams(self):
        feat, label, length, lat_list = self.LoadNextNstreams()
        if feat is None:
            return None, None, None, None
        max_frame_num = numpy.shape(feat)[0]
        nsent = 0
        while nsent < numpy.shape(label)[0]:
            numzeros = max_frame_num - numpy.shape(label[nsent])[0]
            if numzeros != 0:
                label[nsent] = numpy.hstack((label[nsent], 
                    numpy.zeros((numzeros), dtype=numpy.float32)))
            nsent += 1
        return feat, label, length, lat_list

    # load batch size features and labels,it's ce train, cut sentence.
    def SliceLoadNextNstreams(self):
        feat_mat, label, length, max_frame_num, lat_list = self.LoadOnePackage()
        if feat_mat is None:
            return None, None, None, None
        
        if max_frame_num % self.num_frames_batch != 0:
            max_frame_num = self.num_frames_batch * (int(max_frame_num / self.num_frames_batch) + 1)

        # zero fill
        i = 0
        while i < self.batch_size:
            if max_frame_num != length[i]:
                feat_mat[i] = numpy.vstack((feat_mat[i], numpy.zeros((max_frame_num-length[i], feat_mat[i].shape[1]),dtype=numpy.float32)))
            i += 1
        # process package data, slice
        if feat_mat.__len__() == self.batch_size:
            # process feat_mat(list) to time_major numpy [time, batch, dim]
            feat_mat_nstream = numpy.hstack(feat_mat).reshape(-1, self.batch_size, self.output_dim)
            np_length = numpy.vstack(length).reshape(-1)
            # slice feat matrix
            if self.overlap == 0:
                array_feat = numpy.split(feat_mat_nstream, int(max_frame_num / self.num_frames_batch))
            else:
                array_feat = []
                for i in range(int(max_frame_num / self.num_frames_batch)):
                    start_f = int(i*self.num_frames_batch)
                    if i+1 == int(max_frame_num / self.num_frames_batch):
                        end_f = int((i+1)*self.num_frames_batch)
                    else:
                        end_f = int((i+1)*self.num_frames_batch+self.overlap)
                    slice_f = feat_mat_nstream[start_f : end_f]
                    array_feat.append(slice_f)
                array_feat = numpy.array(array_feat)

            array_label = []
            array_length = []
            # all batch
            for nbatch in range(int(max_frame_num / self.num_frames_batch)):
                array_label.append([])
                tmp_length = []
                offset_n = nbatch * self.num_frames_batch
                # one batch
                # every sentence cut
                for i in range(len(label)):
                    tmp_label = []
                    j = 0
                    # slice one sentence label
                    while j < self.num_frames_batch :
                        if j < len(label[i])-offset_n:
                            tmp_label.append(label[i][j+offset_n])
                        else:
                            tmp_label.append(0)
                        j += 1

                    # record feature length
                    if len(label[i])-offset_n > 0:
                        if j + self.overlap > len(label[i])-offset_n:
                            tmp_length.append(len(label[i])-offset_n)
                        else:
                            tmp_length.append(j + self.overlap)
                    else:
                        tmp_length.append(0)

                    array_label[nbatch].append(tmp_label)
                array_length.append(numpy.vstack(tmp_length).reshape(-1))
            return array_feat , array_label , array_length, lat_list
        else:
            logging.info('It\'s shouldn\'t happen. feat is less then batch_size.')
            return None, None, None, None
    
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
    path = '/search/speech/hubo/git/tf-code-acoustics/chain_source/egs-vecfst/'
    conf_dict = { 'batch_size' :64,
            'skip_offset': 0,
            'shuffle': False,
            'queue_cache':200,
            'io_thread_num':5}

    feat_trans_file = '../conf/final.feature_transform'
    feat_trans = FeatureTransform()
    feat_trans.LoadTransform(feat_trans_file)
    logging.basicConfig(filename = 'test.log')
    logging.getLogger().setLevel('INFO')
    io_read = KaldiDataReadParallel()
    
    io_read.Initialize(conf_dict, scp_file=path+'cegs.1.scp',
    #io_read.Initialize(conf_dict, scp_file=path+'../test.1.scp',
            feature_transform = feat_trans, criterion = 'chain')
    start = time.time()
    io_read.Reset(shuffle = False)
    batch_num = 0
    while True:
        start1 = time.time()
        feat_mat, label, length, lat_list = io_read.GetInput()
        #feat_mat, label, length, _, lat_list = io_read.LoadOnePackageEgs()
        end1 = time.time()
        if feat_mat is None:
            break
        logging.info('batch number: '+str(batch_num) + ' ' + str(numpy.shape(feat_mat)))
        logging.info("time:"+str(end1-start1))
        batch_num += 1
    end = time.time()
    io_read.JoinInput()
    logging.info("all process time:"+str(end-start))
    start = time.time()
    io_read.Reset(shuffle = True)
    batch_num = 0
    while True:
        start1 = time.time()
        feat_mat, label, length, lat_list = io_read.GetInput()
        #feat_mat, label, length, _, lat_list = io_read.LoadOnePackageEgs()
        end1 = time.time()
        if feat_mat is None:
            break
        logging.info('batch number: '+str(batch_num) + ' ' + str(numpy.shape(feat_mat)))
        logging.info("time:"+str(end1-start1))
        batch_num += 1
    end = time.time()
    io_read.JoinInput()
    logging.info("all process time:"+str(end-start))

'''
if __name__ == '__main__':
    path = '/search/speech/hubo/git/tf-code-acoustics/fst/cc/source/out-source'
    conf_dict = { 'batch_size' :10,
            'skip_frame':1,
            'skip_offset': 0,
            'do_skip_lab': True,
            'shuffle': False,
            'queue_cache':100,
            'io_thread_num':5,
            'num_frames_batch':20,
            'overlap':10,
            'ali_map_file': path + '/../ali-pdf-phone/map.ali'}
    #path = '/search/speech/hubo/git/tf-code-acoustics/train-data'
    feat_trans_file = '../conf/final.feature_transform'
    feat_trans = FeatureTransform()
    feat_trans.LoadTransform(feat_trans_file)

    logging.basicConfig(filename = 'test.log')
    logging.getLogger().setLevel('INFO')
    io_read = KaldiDataReadParallel()
    io_read.Initialize(conf_dict, scp_file=path+'/feat/500.scp',
            label = path+'/ali/ali.all', 
            lat_scp_file= path + '/decode/lat.all.scp',
#           ali_map_file = path + '/../ali-pdf-phone/map.ali',
            feature_transform = feat_trans, criterion = 'ce')

            #label = path+'/sort_tr.labels.4026.ce',
    start = time.time()
    io_read.Reset(shuffle = True)
    batch_num = 0
    while True:
        #feat_mat, label, length = io_read.LoadNextNstreams()
        #feat_mat, label, length, lat_list = io_read.CnnLoadNextNstreams()
        start1 = time.time()
        feat_mat, label, length, lat_list = io_read.GetInput()
#        feat_mat, label, length, lat_list = io_read.LoadBatch()
        end1 = time.time()
        if feat_mat is None:
            break
        for i in range(numpy.shape(feat_mat)[0]):
            logging.info(str(numpy.shape(feat_mat[i]))+str(length[i]))
        logging.info('batch number: '+str(batch_num) + ' ' + str(numpy.shape(feat_mat))+str(numpy.shape(label))+str(numpy.shape(length)))

        logging.info("time:"+str(end1-start1))
        #feat_mat, label, length = io_read.SliceLoadNextNstreams()
        #print(numpy.shape(feat_mat),numpy.shape(label),numpy.shape(length))
        batch_num += 1
    end = time.time()
    io_read.JoinInput()
    logging.info("all process time:"+str(end-start))
    io_read.Reset(shuffle = True)
    batch_num = 0
    while True:
        #feat_mat, label, length = io_read.LoadNextNstreams()
        #feat_mat, label, length, lat_list = io_read.CnnLoadNextNstreams()
        #feat_mat, label, length, lat_list = io_read.LoadBatch()
        feat_mat, label, length, lat_list = io_read.GetInput()
        if feat_mat is None:
            break
        #logging.info(str(numpy.shape(feat_mat))+str(numpy.shape(label))+str(numpy.shape(length)))
        logging.info('batch number: '+str(batch_num) + ' ' + str(numpy.shape(feat_mat))+str(numpy.shape(label))+str(numpy.shape(length)))
        #feat_mat, label, length = io_read.SliceLoadNextNstreams()
        #print(numpy.shape(feat_mat),numpy.shape(label),numpy.shape(length))
        batch_num += 1

    io_read.JoinInput()
    end = time.time()
    logging.info('load time is : %f s' % (end - start))
'''
