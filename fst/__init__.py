import numpy as np
import sys

sys.path.extend(["../","./"])
from fst.lattice_functions import *
from fst.lattice import *
from fst.convert_lattice_to_sparsematrix import *
from fst.topsort import *

# realize lattice package io

def LatticeMaxTime(lattice):
    max_time, _ = LatticeStateTimes(lattice)
    return max_time

def ReadLatticeScp(scp_line):
    lattice = Lattice()
    key = lattice.ReadScp(scp_line)
    # convert standered lattice
    lattice = ConvertLattice(lattice)
    # super final lattice
    SuperFinalFst(lattice)
    # top sort lattice
    TopSort(lattice)

    max_time, _ = LatticeStateTimes(lattice)
    return key, max_time, lattice

# zero fill at end
# now it depend fill_dim == -1 only
def ZeroFill(in_np, max_len, dim, dtype ,fill_dim = -1):
    if dim == 1:
        length = np.shape(in_np)[0]
        return np.hstack((in_np, np.zeros((max_len - length), dtype=dtype)))
    elif dim == 2:
        length = np.shape(in_np)[0]
        return np.vstack((in_np, np.zeros((max_len - length, np.shape(in_np)[1]), dtype = dtype)))


def ListZeroFill(in_list, max_len = None):
    # get max_len
    if max_len is None:
        max_len = 0
        for i in in_list:
            if max_len < np.shape(i)[0]:
                max_len = np.shape(i)[0]
    
    #
    i = 0
    dim = len(np.shape(in_list[0]))
    while i < len(in_list):
        length = np.shape(in_list[i])[0]
        if max_len != length:
            in_list[i] = ZeroFill(in_list[i], max_len, dim, in_list[i].dtype)
        
        if dim == 2:
            r,c = np.shape(in_list[i])
            in_list[i] = in_list[i].reshape(-1,r,c)
        i += 1
            
    return np.vstack(in_list)

# load ali to pdf and phone list
# return 3D numpy with cols is 3 (ali,pdf,phone) row is ali number + 1
def LoadMapPdfAndPhone(map_file):
    map_list = []
    with open(map_file,'r') as map_fp:
        for line in map_fp:
            map_list.append(line.strip().split())
    
    map_pdf_phone = []
    # ali:pdf:phone
    if len(map_list) == 3:
        ali = map_list[0]
        pdf = map_list[1]
        phone = map_list[2]
        assert len(ali) == len(pdf) and len(ali) == len(phone)
        for i in range(len(pdf)):
            if i == 0 :
                map_pdf_phone.append([0, -1, -1])
            else:
                assert int(ali[i]) == i
                map_pdf_phone.append([int(ali[i]), int(pdf[i]), int(phone[i])])
    # pdf:phone
    else:
        pdf = map_list[0]
        phone = map_list[1]
        assert len(pdf) == len(phone)
        for i in range(len(pdf)):
            if i == 0 :
                map_pdf_phone.append([ -1, -1])
            else:
                map_pdf_phone.append([ int(pdf[i]), int(phone[i])])

    return np.array(map_pdf_phone, dtype=np.int32)

# ali shouldn't contain 0, unless ali is lattice ilabel list, offset should be 1.
# this lattice ilabel is pdf+1 and 0 is <eps>.
def AliToPdf(map_pdf_phone, ali, offset = 0):
    loc = 0
    if len(map_pdf_phone[0]) == 3:
        loc = 1
    i = 0
    pdf = []
    while i < len(ali):
        pdf.append(map_pdf_phone[ali[i]][loc] + offset)
        i += 1
    if type(ali) is np.ndarray:
        return np.array(pdf, dtype=ali.dtype)
    else:
        return pdf

def AliToPhone(map_pdf_phone, ali):
    loc = 1
    if len(map_pdf_phone[0]) == 3:
        loc = 2
    i = 0
    phone = []
    while i < len(ali):
        phone.append(map_pdf_phone[ali[i]][loc])
        i += 1
    if type(ali) is np.ndarray:
        return np.array(phone, dtype=ali.dtype)
    else:
        return pdf

def PackageLattice(lat_scp_list, map_pdf_phone = None):
    max_time = 0
    max_arcs = 0
    max_states = 0
    # lattice struct
    indexs_info_list = []
    pdf_values_list = []
    lmweight_values_list = []
    amweight_values_list = []
    statesinfo_list = []
    statenum_list = []
    time_list = []
    # convert all lattice
    for scp_line in lat_scp_list:
        key, max_t, lattice = ReadLatticeScp(scp_line)
        time_list.append(max_t)

        indexs_info, pdf_values , lmweight_values, amweight_values, statesinfo, shape = ConvertLatticeToSparseMatrix(lattice)
        if map_pdf_phone is not None:
            pdf_values = AliToPdf(map_pdf_phone, pdf_values, offset = 1)
        arc_n = np.shape(indexs_info)[0]
        state_n = shape[0]
        assert np.shape(statesinfo)[0] == state_n
        if np.shape(pdf_values)[0] > max_arcs:
            max_arcs = arc_n
        if max_time < max_t:
            max_time = max_t
        if max_states < shape[0]:
            max_states = state_n

        indexs_info_list.append(indexs_info)
        pdf_values_list.append(pdf_values)
        lmweight_values_list.append(lmweight_values)
        amweight_values_list.append(amweight_values)
        statesinfo_list.append(statesinfo)
        statenum_list.append(shape[0])
    # package all sparse lattice

    indexs_info_list = ListZeroFill(indexs_info_list, max_arcs)
    pdf_values_list = ListZeroFill(pdf_values_list, max_arcs)
    lmweight_values_list = ListZeroFill(lmweight_values_list, max_arcs)
    amweight_values_list = ListZeroFill(amweight_values_list, max_arcs)
    statesinfo_list = ListZeroFill(statesinfo_list, max_states)
    
    return [indexs_info_list, pdf_values_list, lmweight_values_list, amweight_values_list, statesinfo_list, statenum_list, time_list]
    



if __name__ == '__main__':
    batch = 0
    in_lat_list = []
    map_pdf_phone = None
    if len(sys.argv) == 3:
        map_pdf_phone = LoadMapPdfAndPhone(sys.argv[2])
    with open(sys.argv[1],'r') as fp:
        for line in fp:
            if batch != 0 and batch % 32 == 0:
                indexs_info_list, pdf_values_list, lmweight_values_list, amweight_values_list, statesinfo_list, statenum_list, time_list = PackageLattice(in_lat_list, map_pdf_phone)
                in_lat_list = []
                batch = 0
                print(indexs_info_list.shape, pdf_values_list.shape, lmweight_values_list.shape, amweight_values_list.shape, statesinfo_list.shape, len(statenum_list), time_list)
                print(pdf_values_list)
            in_lat_list.append(line.strip())
            batch += 1

