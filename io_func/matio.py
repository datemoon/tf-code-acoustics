from __future__ import unicode_literals

from functools import partial
from io import BytesIO
from io import StringIO
import os
import re
import struct
import sys
import warnings
import logging

import numpy as np
from six import binary_type
from six import string_types

sys.path.append("../")
from io_func.compression_header import GlobalHeader
from io_func.compression_header import PerColHeader
from io_func import smart_open

PY3 = sys.version_info[0] == 3

def read_token(fd, flag=[' ']):
    """Read token
    Args:
        fd (file):
    """
    # add end flag ''
    flag.append('') 
    token = []
    while True:
        char = fd.read(1)
        if isinstance(char, binary_type):
            char = char.decode()
            if char in flag:
                break
            else:
                token.append(char)
    if len(token) == 0:  # End of file
        return None
    return ''.join(token)

def read_matrix_or_vector(fd, endian='<', return_size=False, read_binary_flag = True):
    """Call from load_kaldi_ark
    
    Args:
        fd (file):
        endian (str):
        return_size (bool):
        read_binary_flag (boo): read binary flag
    """
    size = 0
    if read_binary_flag:
        assert fd.read(2) == b'\0B'
        size += 2

    Type = str(read_token(fd))
    size += len(Type) + 1
    
    # CompressedMatrix
    if 'CM' == Type:
        # Read GlobalHeader
        global_header = GlobalHeader.read(fd, Type, str(endian))
        size += global_header.size
        per_col_header = PerColHeader.read(fd, global_header)
        size += per_col_header.size

        # Read data
        buf = fd.read(global_header.rows * global_header.cols)
        size += global_header.rows * global_header.cols
        array = np.frombuffer(buf, dtype=np.dtype(str(endian + 'u1')))
        array = array.reshape((global_header.cols, global_header.rows))

        # Decompress
        array = per_col_header.char_to_float(array)
        array = array.T

    elif 'CM2' == Type:
        # Read GlobalHeader
        global_header = GlobalHeader.read(fd, Type, str(endian))
        size += global_header.size
        
        # Read matrix
        buf = fd.read(global_header.rows * global_header.cols)
        array = np.frombuffer(buf, dtype=np.dtype(str(endian + 'u1')))
        array = array.reshape((global_header.rows, global_header.cols))

        # Decompress
        array = global_header.uint_to_float(array)

    else:
        if Type == 'FM' or Type == 'FV':
            dtype = str(endian) + 'f'
            bytes_per_sample = 4
        elif Type == 'DM' or Type == 'DV':
            dtype = str(endian) + 'd'
            bytes_per_sample = 8
        else:
            raise ValueError(
                    'Unexpected format: "{}". Now FM, FV, DM, DV, '
                    'CM, CM2, CM3 are supported.'.format(Type))

        assert fd.read(1) == b'\4'
        size += 1
        rows = struct.unpack(str(endian + 'i'), fd.read(4))[0]
        size += 4
        dim = rows
        if 'M' in Type:  # As matrix
            assert fd.read(1) == b'\4'
            size += 1
            cols = struct.unpack(str(endian + 'i'), fd.read(4))[0]
            size += 4
            dim = rows * cols

        buf = fd.read(dim * bytes_per_sample)
        size += dim * bytes_per_sample
        array = np.frombuffer(buf, dtype=np.dtype(dtype))

        if 'M' in Type:  # As matrix
            array = np.reshape(array, (rows, cols))

    if return_size:
        return array, size
    else:
        return array

def read_ascii_mat(fd, return_size=False):
    """Call from load_kaldi_ark

    Args:
        fd (file): binary mode
        return_size (bool):
    """
    string = []
    size = 0

    # Find '[' char
    while True:
        try:
            char = fd.read(1).decode()
        except UnicodeDecodeError as e:
            raise UnicodeDecodeError(
                    str(e) + '\nFile format is wrong?')
        size += 1
        if char == ' ' or char == os.linesep:
            continue
        elif char == '[':
            hasparent = True
            break
        else:
            string.append(char)
            hasparent = False
            break

    # Read data
    ndmin = 1
    while True:
        char = fd.read(1).decode()
        size += 1
        if hasparent:
            if char == ']':
                char = fd.read(1).decode()
                size += 1
                assert char == os.linesep or char == ''
                break
            elif char == os.linesep:
                ndmin = 2
            elif char == '':
                raise ValueError(
                        'There are no corresponding bracket \']\' with \'[\'')
        else:
            if char == os.linesep or char == '':
                break
        string.append(char)
    string = ''.join(string)
    assert len(string) != 0
    # Examine dtype
    match = re.match(r' *([^ \n]+) *', string)
    if match is None:
        dtype = np.float32
    else:
        ma = match.group(0)
        # If first element is integer, deal as interger array
        try:
            float(ma)
        except ValueError:
            raise RuntimeError(
                    ma + 'is not a digit\nFile format is wrong?')
        if '.' in ma:
            dtype = np.float32
        else:
            dtype = np.int32
    array = np.loadtxt(StringIO(string), dtype=dtype, ndmin=ndmin)
    if return_size:
        return array, size
    else:
        return array

def read_kaldi(fd, endian='<', return_size=False):
    """Load kaldi
    
    Args:
        fd (file): Binary mode file object. Cannot input string
        endian (str):
        return_size (bool):
    """
    assert str(endian) in ('<', '>'), endian
    pos = fd.tell()

    binary_flag = fd.read(4)
    assert isinstance(binary_flag, binary_type), type(binary_flag)

    fd.seek(int(pos),0)

    # Load as binary
    if binary_flag[:2] == b'\0B':
        array, size = read_matrix_or_vector(fd, endian=str(endian), return_size=True)
        
    # Load as ascii
    else:
        array, size = read_ascii_mat(fd, return_size=True)

    if return_size:
        return array, size
    else:
        return array


def read_ark(ark_file, endian='<', return_position=False):
    assert str(endian) in ('<', '>'), endian
    size = 0
    fd = smart_open(ark_file, 'rb')
    while True:
        key = read_token(fd)
        if key is None:
            break
        size += len(key) + 1
        array, _size = read_kaldi(fd, str(endian), return_size=True)
        print(key, array)
        size += _size


# read the feature matrix
def read_next_utt(next_scp_line):
    # this shouldn't happen
    if next_scp_line == '' or next_scp_line == None:    # we are reaching the end of one epoch
        return '', None

    utt_id, path_pos = next_scp_line.replace('\n','').split(' ')
    path, pos = path_pos.split(':')

    ark_read_buffer = smart_open(path, 'rb')
    ark_read_buffer.seek(int(pos),0)

    endian='<'

    binary_flag = ark_read_buffer.read(4)
    assert isinstance(binary_flag, binary_type), type(binary_flag)

    ark_read_buffer.seek(int(pos),0)
    # Load as binary
    if binary_flag[:2] == b'\0B':
        array, size = read_matrix_or_vector(ark_read_buffer, endian=str(endian), return_size=True)
        
    # Load as ascii
    else:
        array, size = read_ascii_mat(ark_read_buffer, return_size=True)
    
    ark_read_buffer.close()
    return utt_id, array

if __name__ == '__main__':
    read_ark('../train-data/cv.ark')
    #read_ark('cv.compress.ark')
