
import sys
import numpy as np

def GetDim(line):
    [rows, cols] = line.split('(')[1].split(')')[0].replace(' ','').split(',')
    if cols == '':
        return 1, int(rows)
    else:
        return int(rows), int(cols)

def ReadMatrix(model_file, rows, cols):
    matrix = []
    num_line = 0
    for line in  model_file:
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

def WriteLstm(fp, w_para, lstm_input, lstm_cell_dim, lstm_proj_dim):
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
    except KeyError:
        print('no project parameters')
    token = '<!EndOfComponent> \n'
    fp.write(token)

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

    token = '<!EndOfComponent> \n'
    fp.write(token)

def WriteAffineTransfrom(fp, weights, biases, input_dim, output_dim):
    token = '<AffineTransform> ' + str(output_dim) + ' ' + str(input_dim) + ' \n'
    fp.write(token)

    np_w = np.transpose(np.array(weights ,dtype = np.float32))
    WriteMatrix(fp, np_w, output_dim, input_dim, 0)

    np_b = np.array(biases, dtype = np.float32).reshape(1, -1)
    WriteMatrix(fp, np_b, np_b.shape[0], np_b.shape[1])

    token = '<!EndOfComponent> \n'
    fp.write(token)

def WriteSoftmax(fp, dim):
    token = '<Softmax> ' + str(dim) + ' ' + str(dim) + ' \n' + '<!EndOfComponent> \n'
    fp.write(token)

#
def ConvertLstmLayer(model_file, cell_key, lstm_para):
    model_file.seek(0)
    lstm_input = lstm_para[0]
    lstm_cell_dim = lstm_para[1]
    lstm_proj_dim = lstm_para[2]
    parameters = {}
    for line in model_file:
        if cell_key in line and 'lstm_cell/weights' in line:
            rows , cols = GetDim(line)
            assert rows == lstm_input + lstm_proj_dim
            assert cols == 4 * lstm_cell_dim
            weights = ReadMatrix(model_file, rows, cols)
            parameters['weights'] = weights
        elif cell_key in line and 'lstm_cell/biases' in line:
            rows , cols = GetDim(line)
            assert cols == 4 * lstm_cell_dim
            biases = ReadMatrix(model_file, rows, cols)
            parameters['biases'] = biases
        elif cell_key in line and 'lstm_cell/w_f_diag' in line:
            rows , cols = GetDim(line)
            assert cols == lstm_cell_dim
            w_f_diag = ReadMatrix(model_file, rows, cols)
            parameters['w_f_diag'] = w_f_diag

        elif cell_key in line and 'lstm_cell/w_i_diag' in line:
            rows , cols = GetDim(line)
            assert cols == lstm_cell_dim
            w_i_diag = ReadMatrix(model_file, rows, cols)
            parameters['w_i_diag'] = w_i_diag

        elif cell_key in line and 'lstm_cell/w_o_diag' in line:
            rows , cols = GetDim(line)
            assert cols == lstm_cell_dim
            w_o_diag = ReadMatrix(model_file, rows, cols)
            parameters['w_o_diag'] = w_o_diag
        elif cell_key in line and 'lstm_cell/projection/weights' in line:
            rows , cols = GetDim(line)
            assert cols == lstm_proj_dim
            assert rows == lstm_cell_dim
            proj_weight = ReadMatrix(model_file, rows, cols)
            parameters['proj_weights'] = proj_weight
    return parameters

def ConvertSoftmaxLayer(model_file, cell_key, layer_para):
    model_file.seek(0)
    input_dim = layer_para[0]
    output_dim =  layer_para[1]
    weights=[]
    biases=[]
    for line in model_file:
        if cell_key in line and 'softmax_w' in line:
            rows , cols = GetDim(line)
            assert rows == input_dim 
            assert cols == output_dim
            weights = ReadMatrix(model_file, rows, cols)

        elif cell_key in line and 'softmax_b' in line:
            rows , cols = GetDim(line)
            assert cols == output_dim
            biases = ReadMatrix(model_file, rows, cols)

    return weights, biases

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

    w,b = ConvertSoftmaxLayer(fp, softmax_struct[0], softmax_struct[1])
    WriteAffineTransfrom(fp_out, w, b, softmax_struct[1][0], softmax_struct[1][1])
    WriteSoftmax(fp_out, softmax_struct[1][1])
    weights.append([w, b])
    
    token = '</Nnet> '
    fp_out.write(token)
    fp.close()
    fp_out.close()

if __name__ == '__main__':
    if False:
        lstm_struct = [['cell_0/lstm_cell',[280,1024,1024]], 
                ['cell_1/lstm_cell', [1024,1024,1024]],
                ['cell_2/lstm_cell', [1024,1024,1024]] ]
        softmax_struct = ['Softmax_layer',[1024, 4223]] 
        ConvertTfToKaldi(sys.argv[1], sys.argv[2], lstm_struct,softmax_struct)
    else:
        lstm_proj_struct = [['cell_0/lstm_cell',[280,1024,512]], 
                ['cell_1/lstm_cell', [512,1024,512]],
                ['cell_2/lstm_cell', [512,1024,512]] ]
    #    lstm_struct = [['cell_0',[280,1024,1024]]]
        softmax_struct = ['Softmax_layer',[512, 4026]] 
        ConvertTfToKaldi(sys.argv[1], sys.argv[2], lstm_proj_struct,softmax_struct)
