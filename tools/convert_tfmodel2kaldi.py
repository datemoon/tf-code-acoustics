
import sys
import numpy as np

EndOfComponent_token='<!EndOfComponent> \n'
EndOfComponent_token=''

def GetDim(line):
    [rows, cols] = line.split('(')[1].split(')')[0].replace(' ','').split(',')
    if cols == '':
        return 1, int(rows)
    else:
        return int(rows), int(cols)

def GetCnnDim(line):
    [height, width, inchannel, outchannel] = line.split('(')[1].split(')')[0].replace(' ','').split(',')
    return int(height), int(width), int(inchannel), int(outchannel)

def ReadMatrix(model_file, rows, cols):
    matrix = []
    num_line = 0
    for line in  model_file:
        if len(line.rstrip()) == 0:
            continue
        assert line.split()[0][0] == '[' 
        matrix.append(line.replace('[','').replace(']','').split())
        assert len(matrix[num_line]) == cols
        num_line += 1
        if num_line == rows:
            break
    return matrix

def WriteMatrix(fp, matrix, rows, cols, offset = 0):
    assert rows > 0
    if rows == 1:
        fp.write(' [ ')
    else:
        fp.write(' [\n')
    for r in range(rows):
        line = '  '
        for c in range(cols):
            line += str(matrix[r][offset+c]) + ' '
        if r + 1 == rows:
            line += ']'
        line += '\n'
        fp.write(line)


def LstmTf2KaldiMatrix(np_weights, lstm_cell_dim):
    assert np_weights.shape[0] == 4 * lstm_cell_dim 
    np_ijfo = []
    for num in range(4):
        np_ijfo.append(np_weights[lstm_cell_dim*num : lstm_cell_dim*(num+1)])
    kaldi_gifo = []
    for i in [1,0,2,3]:
        kaldi_gifo.append(np_ijfo[i])

    return np.vstack(kaldi_gifo)

def WriteLstm(fp, w_para, lstm_input, lstm_cell_dim, lstm_proj_dim, add_proj_bias = False):
    weights = w_para['weights']
    biases = w_para['biases']
    np_w = np.array(weights ,dtype = np.float32)
    # tf    is i j f o
    # kaldi is g i f o
    np_w_ijfo_x = np.transpose(np_w[0 : lstm_input])
    np_w_ijfo_m = np.transpose(np_w[lstm_input : lstm_input + lstm_proj_dim])

    np_w_gifo_x = LstmTf2KaldiMatrix(np_w_ijfo_x, lstm_cell_dim)
    np_w_gifo_m = LstmTf2KaldiMatrix(np_w_ijfo_m, lstm_cell_dim)

    np_b_ijfo = np.array(biases, dtype = np.float32).reshape(-1,1)

    np_b_gifo = LstmTf2KaldiMatrix(np_b_ijfo, lstm_cell_dim).reshape(1, -1)

    # write model parameters
    if lstm_cell_dim == lstm_proj_dim:
        token = '<TfLstm> ' + str(lstm_proj_dim) + ' ' +str(lstm_input) + '\n'
    else:
        token = '<LstmProjected> ' + str(lstm_proj_dim) + ' ' + str(lstm_input) + '\n'
        token += '<CellDim> ' + str(lstm_cell_dim) + '\n'
    fp.write(token)
    WriteMatrix(fp, np_w_gifo_x, lstm_cell_dim*4, lstm_input, 0)
    WriteMatrix(fp, np_w_gifo_m, lstm_cell_dim*4, lstm_proj_dim, 0)
    WriteMatrix(fp, np_b_gifo, np_b_gifo.shape[0], np_b_gifo.shape[1])
    proj_weights = None
    for key in ['w_i_diag', 'w_f_diag', 'w_o_diag']:
        try:
            w_ifo_diag = w_para[key]
            np_w_ifo_diag = np.array(w_ifo_diag, dtype = np.float32).reshape(1, -1)
            WriteMatrix(fp, np_w_ifo_diag, np_w_ifo_diag.shape[0], np_w_ifo_diag.shape[1], 0)
        except KeyError:
            print('no i f o diag parameters')
            break
    try:
        proj_weights = w_para['proj_weights']
        np_proj_weights = np.transpose( np.array(proj_weights ,dtype = np.float32))
        WriteMatrix(fp, np_proj_weights, np_proj_weights.shape[0], np_proj_weights.shape[1])
        if add_proj_bias is True:
            assert lstm_proj_dim == np_proj_weights.shape[0]
            np_proj_bias = np.zeros(lstm_proj_dim).reshape(1,-1)
            WriteMatrix(fp, np_proj_bias, np_proj_bias.shape[0], np_proj_bias.shape[1])
    except KeyError:
        print('no project parameters')
    token = EndOfComponent_token
    if token != '':
        fp.write(token)

def WriteLstmWeight(fp, w_para, lstm_input, lstm_cell_dim, lstm_proj_dim):
    weights = w_para['weights']
    biases = w_para['biases']
    np_w = np.array(weights ,dtype = np.float32)
    # tf    is i j f o
    # kaldi is g i f o
    np_w_ijfo_x = np.transpose(np_w[0 : lstm_input])
    np_w_ijfo_m = np.transpose(np_w[lstm_input : lstm_input + lstm_proj_dim])

    np_w_gifo_x = LstmTf2KaldiMatrix(np_w_ijfo_x, lstm_cell_dim)
    np_w_gifo_m = LstmTf2KaldiMatrix(np_w_ijfo_m, lstm_cell_dim)

    np_b_ijfo = np.array(biases, dtype = np.float32).reshape(-1,1)

    np_b_gifo = LstmTf2KaldiMatrix(np_b_ijfo, lstm_cell_dim).reshape(1, -1)

    # write model parameters
    #if lstm_cell_dim == lstm_proj_dim:
    #    token = '<TfLstm> ' + str(lstm_proj_dim) + ' ' +str(lstm_input) + '\n'
    #else:
    #    token = '<LstmProjected> ' + str(lstm_proj_dim) + ' ' + str(lstm_input) + '\n'
    #    token += '<CellDim> ' + str(lstm_cell_dim) + '\n'
    #fp.write(token)
    WriteMatrix(fp, np_w_gifo_x, lstm_cell_dim*4, lstm_input, 0)
    WriteMatrix(fp, np_w_gifo_m, lstm_cell_dim*4, lstm_proj_dim, 0)
    WriteMatrix(fp, np_b_gifo, np_b_gifo.shape[0], np_b_gifo.shape[1])
    proj_weights = None
    for key in ['w_i_diag', 'w_f_diag', 'w_o_diag']:
        try:
            w_ifo_diag = w_para[key]
            np_w_ifo_diag = np.array(w_ifo_diag, dtype = np.float32).reshape(1, -1)
            WriteMatrix(fp, np_w_ifo_diag, np_w_ifo_diag.shape[0], np_w_ifo_diag.shape[1], 0)
        except KeyError:
            print('no i f o diag parameters')
            break
    try:
        proj_weights = w_para['proj_weights']
        np_proj_weights = np.transpose( np.array(proj_weights ,dtype = np.float32))
        WriteMatrix(fp, np_proj_weights, np_proj_weights.shape[0], np_proj_weights.shape[1])
    except KeyError:
        print('no project parameters')
    #token = EndOfComponent_token
    #if token != '':
        #fp.write(token)

def WriteBlstm(fp, fp_out, blstm_para):
    fw_cell_key = blstm_para[0][0]
    bw_cell_key = blstm_para[0][1]
    fw_lstm_para = blstm_para[1][0]
    bw_lstm_para = blstm_para[1][1]
    assert fw_lstm_para[0] == bw_lstm_para[0]
    
    kaldi_input = fw_lstm_para[0]
    
    fw_lstm_cell_dim = fw_lstm_para[1]
    bw_lstm_cell_dim = bw_lstm_para[1] 
    assert fw_lstm_cell_dim == bw_lstm_cell_dim
    kaldi_cell = fw_lstm_cell_dim 
    
    fw_lstm_proj_dim = fw_lstm_para[2]
    bw_lstm_proj_dim = bw_lstm_para[2]
    kaldi_output = fw_lstm_proj_dim + bw_lstm_proj_dim

    token = '<BlstmProjected> ' + str(kaldi_output) + ' ' + str(kaldi_input) + ' \n'
    token += '<CellDim> ' + str(kaldi_cell) + ' <LearnRateCoef> 1 <BiasLearnRateCoef> 1 <CellClip> 5 <DiffClip> 1 <CellDiffClip> 0 <GradClip> 5\n'
    fp_out.write(token)
    # read blstm
    blstm_para = ConvertBLstmLayer(fp, blstm_para)
    
    # write forward parameter
    fw_para = blstm_para['fw_para']
    WriteLstmWeight(fp_out, fw_para, kaldi_input, fw_lstm_cell_dim, fw_lstm_proj_dim)
    # write backward parameter
    bw_para = blstm_para['bw_para']
    WriteLstmWeight(fp_out, bw_para, kaldi_input, bw_lstm_cell_dim, bw_lstm_proj_dim)

    token = EndOfComponent_token
    if token != '':
        fp_out.write(token)

    return blstm_para

def WriteLstmOld(fp, weights, biases, lstm_input, lstm_cell_dim, lstm_proj_dim):
    token = '<TfLstm> ' + str(lstm_cell_dim) + ' ' +str(lstm_input) + '\n'
    fp.write(token)
    
    #np_w = np.transpose(np.array(weights ,dtype = np.float32))
    np_w = np.array(weights ,dtype = np.float32)
    # tf    is i j f o
    # kaldi is g i f o

    np_w_ijfo_x = np.transpose(np_w[0 : lstm_input])
    np_w_ijfo_m = np.transpose(np_w[lstm_input : lstm_input + lstm_proj_dim])

    np_w_gifo_x = LstmTf2KaldiMatrix(np_w_ijfo_x, lstm_cell_dim)
    np_w_gifo_m = LstmTf2KaldiMatrix(np_w_ijfo_m, lstm_cell_dim)
    
    WriteMatrix(fp, np_w_gifo_x, lstm_cell_dim*4, lstm_input, 0)
    WriteMatrix(fp, np_w_gifo_m, lstm_cell_dim*4, lstm_proj_dim, 0)
    #WriteMatrix(fp, np_w, lstm_cell_dim*4, lstm_input, lstm_proj_dim)
    #WriteMatrix(fp, np_w, lstm_cell_dim*4, lstm_proj_dim, 0)

    np_b_ijfo = np.array(biases, dtype = np.float32).reshape(-1,1)

    np_b_gifo = LstmTf2KaldiMatrix(np_b_ijfo, lstm_cell_dim).reshape(1, -1)
    WriteMatrix(fp, np_b_gifo, np_b_gifo.shape[0], np_b_gifo.shape[1])

    token = EndOfComponent_token
    if token != '':
        fp.write(token)

def WriteAffineTransfrom(fp, weights, biases, input_dim, output_dim):
    token = '<AffineTransform> ' + str(output_dim) + ' ' + str(input_dim) + ' \n'
    token += '<LearnRateCoef> 2.5 <BiasLearnRateCoef> 2.5 <MaxNorm> 0\n'
    fp.write(token)

    np_w = np.transpose(np.array(weights ,dtype = np.float32))
    WriteMatrix(fp, np_w, output_dim, input_dim, 0)

    np_b = np.array(biases, dtype = np.float32).reshape(1, -1)
    WriteMatrix(fp, np_b, np_b.shape[0], np_b.shape[1])

    token = EndOfComponent_token
    if token != '':
        fp.write(token)

def WriteSoftmax(fp, dim):
    token = '<Softmax> ' + str(dim) + ' ' + str(dim) + ' \n'
    fp.write(token)
    token = EndOfComponent_token
    if token != '':
        fp.write(token)

#
def ConvertLstmLayer(model_file, cell_key, lstm_para):
    model_file.seek(0)
    lstm_input = lstm_para[0]
    lstm_cell_dim = lstm_para[1]
    lstm_proj_dim = lstm_para[2]
    parameters = {}
    for line in model_file:
        if cell_key in line and cell_key +'/kernel' in line:
            rows , cols = GetDim(line)
            assert rows == lstm_input + lstm_proj_dim
            assert cols == 4 * lstm_cell_dim
            weights = ReadMatrix(model_file, rows, cols)
            parameters['weights'] = weights
        elif cell_key in line and cell_key + '/bias' in line:
            rows , cols = GetDim(line)
            assert cols == 4 * lstm_cell_dim
            biases = ReadMatrix(model_file, rows, cols)
            parameters['biases'] = biases
        elif cell_key in line and cell_key + '/w_f_diag' in line:
            rows , cols = GetDim(line)
            assert cols == lstm_cell_dim
            w_f_diag = ReadMatrix(model_file, rows, cols)
            parameters['w_f_diag'] = w_f_diag

        elif cell_key in line and cell_key + '/w_i_diag' in line:
            rows , cols = GetDim(line)
            assert cols == lstm_cell_dim
            w_i_diag = ReadMatrix(model_file, rows, cols)
            parameters['w_i_diag'] = w_i_diag

        elif cell_key in line and cell_key + '/w_o_diag' in line:
            rows , cols = GetDim(line)
            assert cols == lstm_cell_dim
            w_o_diag = ReadMatrix(model_file, rows, cols)
            parameters['w_o_diag'] = w_o_diag
        elif cell_key in line and cell_key + '/projection/kernel' in line:
            rows , cols = GetDim(line)
            assert cols == lstm_proj_dim
            assert rows == lstm_cell_dim
            proj_weight = ReadMatrix(model_file, rows, cols)
            parameters['proj_weights'] = proj_weight
    return parameters

def ConvertBLstmLayer(model_file, blstm_para):
    fw_cell_key = blstm_para[0][0]
    bw_cell_key = blstm_para[0][1]
    fw_lstm_para = blstm_para[1][0]
    bw_lstm_para = blstm_para[1][1]
    parameters = {}
    parameters['fw_para'] = ConvertLstmLayer(model_file, fw_cell_key, fw_lstm_para)
    parameters['bw_para'] = ConvertLstmLayer(model_file, bw_cell_key, bw_lstm_para)
    return parameters


def ConvertCnnLayer(model_file, cell_key, cnn_para):
    model_file.seek(0)
    h = cnn_para[0]
    w = cnn_para[1]
    i = cnn_para[2]
    o = cnn_para[3]
    parameters = {}
    for line in model_file:
        if cell_key in line and '_w' in line:
            height, width, inchannel, outchannel = GetCnnDim(line)
            assert h == height
            assert w == width
            assert i == inchannel
            assert o == outchannel
            rows = h * w * i
            cols = outchannel
            cnn_weight = ReadMatrix(model_file, rows, cols)
            parameters['weights'] = cnn_weight
            break
    parameters['biases'] = np.zeros(o, dtype=np.float32).reshape((1, o))

    return parameters

def WriteCnnLayer(out_file, cnn_para, cnn_kaldi_para):
    # nnet para
    indim = cnn_kaldi_para[0]
    outdim = cnn_kaldi_para[1]
    patchdim = cnn_kaldi_para[2]
    patchstep = cnn_kaldi_para[3]
    patchstrid = cnn_kaldi_para[4]
    token = '<ConvolutionalComponent> ' + str(outdim) + ' ' + str(indim) + ' \n'
    token += '<PatchDim> ' + str(patchdim) + ' <PatchStep> ' + str(patchstep) + ' <PatchStride> ' + str(patchstrid) + ' \n'
    token += '<LearnRateCoef> 2.5 <BiasLearnRateCoef> 2.5 <MaxNorm> 0\n'
    token += '<Filters>\n'
    out_file.write(token)

    #weight
    weights = cnn_para['weights']
    np_w = np.array(weights ,dtype = np.float32)
    #np_wt = np_w.transpose((1,0))
    np_wt = np.transpose(np_w)
    rows, cols = np.shape(np_wt)
    WriteMatrix(out_file, np_wt, rows, cols, 0)

    # write bias
    token = '<Bias>\n'
    out_file.write(token)
    bias = cnn_para['biases']
    rows, cols = np.shape(bias)
    WriteMatrix(out_file, bias, rows, cols, 0)
    token = EndOfComponent_token
    if token != '':
        out_file.write(token)


def ConvertAffineTransfromLayer(model_file, cell_key, layer_para):
    model_file.seek(0)
    input_dim = layer_para[0]
    output_dim =  layer_para[1]
    weights=[]
    biases=[]
    for line in model_file:
        if cell_key in line and '_w' in line:
            rows , cols = GetDim(line)
            assert rows == input_dim 
            assert cols == output_dim
            weights = ReadMatrix(model_file, rows, cols)

        elif cell_key in line and '_b' in line:
            rows , cols = GetDim(line)
            assert cols == output_dim
            biases = ReadMatrix(model_file, rows, cols)

    return weights, biases

def WriteMaxPool(out_file, cell_para):
    poolinput = cell_para[0]
    pooloutput = cell_para[1]
    poolsize = cell_para[2]
    poolstep = cell_para[3]
    # outchannel
    poolstride = cell_para[4]
    maxpoolnnet = '<MaxPoolingComponent> ' + str(pooloutput) + ' ' + str(poolinput) + ' \n'
    maxpoolnnet += '<PoolSize> ' + str(poolsize) + ' <PoolStep> ' + str(poolstep) + ' <PoolStride> ' + str(poolstride)
    out_file.write(maxpoolnnet) + '\n'
    token = EndOfComponent_token
    if token != '':
        out_file.write(token)


def WriteSigmoid(out_file, dim):
    sigmoidnnet = '<Sigmoid> ' + str(dim) + ' ' + str(dim) + ' \n' 
    out_file.write(sigmoidnnet)
    token = EndOfComponent_token
    if token != '':
        out_file.write(token)


def ConvertTfToKaldi(model_in_tf, model_out_kaldi, lstm_struct, softmax_struct):
    fp = open(model_in_tf, 'r')
    fp_out = open(model_out_kaldi, 'w')
    token = '<Nnet> \n'
    fp_out.write(token)
    weights = []
    for key, layer_para in lstm_struct:
        weights_para = ConvertLstmLayer(fp, key, layer_para)

        WriteLstm(fp_out, weights_para, layer_para[0], layer_para[1], layer_para[2])
        weights.append(weights_para)

    w,b = ConvertAffineTransfromLayer(fp, softmax_struct[0], softmax_struct[1])
    WriteAffineTransfrom(fp_out, w, b, softmax_struct[1][0], softmax_struct[1][1])
    WriteSoftmax(fp_out, softmax_struct[1][1])
    weights.append([w, b])
    
    token = '</Nnet> '
    fp_out.write(token)
    fp.close()
    fp_out.close()
    
# parameter = [tdnn1affine/tdnn1affine, [[355, 625], [-1, 0, 1]]]
def TdnnConvertTfToKaldi(fp_in, fp_out, key, parameter):
    indim = parameter[0][0]
    outdim = parameter[0][1]
    # splice
    splice = str(parameter[1]).replace(',',' ').replace('[','[ ').replace(']', ' ]')
    splice_indim = indim
    splice_outdim =  len(parameter[1]) * indim
    token = '<Splice> ' + str(splice_outdim) + ' ' + str(splice_indim) + '\n'
    fp_out.write(token)
    fp_out.write(splice + '\n')
    # affine

    affine_indim = splice_outdim
    affine_outdim = outdim
    affine_weight, affine_bias = ConvertAffineTransfromLayer(fp_in, key, [affine_indim, affine_outdim])
    WriteAffineTransfrom(fp_out, affine_weight, affine_bias, affine_indim, affine_outdim)
    # relu
    token = '<ReLU> ' + str(affine_outdim) + ' ' + str(affine_outdim) + '\n'
    fp_out.write(token)
    # NormalizeComponent
    token = '<NormalizeComponent> ' + str(affine_outdim) + ' ' + str(affine_outdim)+'\n'
    fp_out.write(token)
    # TargetRms
    token = '<TargetRms> 1\n'
    fp_out.write(token)
    return affine_weight, affine_bias

def ConvertTfToKaldi(model_in_tf, model_out_kaldi, layer_struct):
    fp = open(model_in_tf, 'r')
    fp_out = open(model_out_kaldi, 'w')
    token = '<Nnet> \n'
    fp_out.write(token)
    weights = []
    num_parameters = 0 
    for key, layer_para, conf_dict in layer_struct:
        if 'lstm' in key:
            weights_para = ConvertLstmLayer(fp, key, layer_para)
            add_proj_bias = False
            try :
                if conf_dict['add_proj_bias'] is True:
                    add_proj_bias = True
            except KeyError:
                print('not add proj bias')

            WriteLstm(fp_out, weights_para, layer_para[0], layer_para[1], layer_para[2], add_proj_bias)
            weights.append(weights_para)
            for lstm_p in weights_para.values():
                num_parameters += np.size(lstm_p)
        elif 'tdnn' in key : 
            w,b = TdnnConvertTfToKaldi(fp, fp_out, key, layer_para)
            weights.append([w, b])
            num_parameters += np.size(w) + np.size(b)
        elif 'affine' in key and 'tdnn' not in key: 
            w,b = ConvertAffineTransfromLayer(fp, key, layer_para)
            WriteAffineTransfrom(fp_out, w, b, layer_para[0], layer_para[1])
            weights.append([w, b])
            num_parameters += np.size(w) + np.size(b)
        elif 'softmax' in key:
            WriteSoftmax(fp_out, layer_para)
        else:
            print('no layer.')

    token = '</Nnet> '
    fp_out.write(token)
    fp.close()
    fp_out.close()
    print("total parameters:%d" %(num_parameters) )

    


if __name__ == '__main__':
    lstm = False
    lstmproj = False
    tdnn_lstm = True
    if lstm is True:
        lstm_struct = [['cell_0/lstm_cell',[280,1024,1024], {}], 
                ['cell_1/lstm_cell', [1024,1024,1024], {}],
                ['cell_2/lstm_cell', [1024,1024,1024], {}]
                ]
        softmax_struct = ['Softmax_layer',[1024, 4223], {}] 
        ConvertTfToKaldi(sys.argv[1], sys.argv[2], lstm_struct,softmax_struct)
    elif lstmproj is True:
        lstm_proj_struct = [['cell_0/lstm_cell',[280,1024,512], {}], 
                ['cell_1/lstm_cell', [512,1024,512], {}],
                ['cell_2/lstm_cell', [512,1024,512], {}] 
                ]
    #    lstm_struct = [['cell_0',[280,1024,1024]]]
        softmax_struct = ['Softmax_layer',[512, 4026]] 
        ConvertTfToKaldi(sys.argv[1], sys.argv[2], lstm_proj_struct,softmax_struct)
    elif tdnn_lstm is True:
        layer_struct = [['tdnn1affine',[[355, 625], [0]], {}],
                ['tdnn2affine',[[625, 625], [ -1,0,1 ]], {}],
                ['tdnn3affine',[[625, 625], [ -1,0,1 ]], {}],
                ['lstmlayer1', [625, 1024, 256], {'add_proj_bias':True}],
                ['tdnn4affine',[[256, 625], [ -1,0,1 ]], {}],
                ['tdnn5affine',[[625, 625], [ -1,0,1 ]], {}],
                ['lstmlayer2', [625, 1024, 256], {'add_proj_bias':True}],
                ['tdnn6affine',[[256, 625], [ -1,0,1 ]], {}],
                ['tdnn7affine',[[625, 625], [ -1,0,1 ]], {}],
                ['lstmlayer3', [625, 1024, 256], {'add_proj_bias':True}],
                ['affine2_1_1',[256, 3766], {}]
                #['affine1',[256, 3766], {}]
                ]
        ConvertTfToKaldi(sys.argv[1], sys.argv[2], layer_struct)
    else:
        print("no nnet.")
