
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

def WriteLstm(fp, weights, biases, lstm_input, lstm_cell_dim, lstm_proj_dim):
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

    np_b = np.array(biases, dtype = np.float32).reshape(1, -1)
    WriteMatrix(fp, np_b, np_b.shape[0], np_b.shape[1])

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
    weights=[]
    biases=[]
    for line in model_file:
        if cell_key in line and 'weights' in line:
            rows , cols = GetDim(line)
            if lstm_cell_dim == lstm_proj_dim:
                assert rows == lstm_input + lstm_proj_dim
                assert cols == 4 * lstm_cell_dim
            weights = ReadMatrix(model_file, rows, cols)

        elif cell_key in line and 'biases' in line:
            rows , cols = GetDim(line)
            assert cols == 4 * lstm_cell_dim
            biases = ReadMatrix(model_file, rows, cols)

    return weights, biases

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
    biases = []
    for key, layer_para in lstm_struct:
        w,b = ConvertLstmLayer(fp, key, layer_para)

        WriteLstm(fp_out, w, b, layer_para[0], layer_para[1], layer_para[2])
        weights.append(w)
        biases.append(b)

    w,b = ConvertSoftmaxLayer(fp, softmax_struct[0], softmax_struct[1])
    WriteAffineTransfrom(fp_out, w, b, softmax_struct[1][0], softmax_struct[1][1])
    WriteSoftmax(fp_out, softmax_struct[1][1])
    weights.append(w)
    biases.append(b)
    
    token = '</Nnet> '
    fp_out.write(token)
    fp.close()
    fp_out.close()

if __name__ == '__main__':
    lstm_struct = [['cell_0',[280,1024,1024]], 
            ['cell_1', [1024,1024,1024]],
            ['cell_2', [1024,1024,1024]] ]
#    lstm_struct = [['cell_0',[280,1024,1024]]]
    softmax_struct = ['Softmax_layer',[1024, 4223]] 
    ConvertTfToKaldi(sys.argv[1], sys.argv[2], lstm_struct,softmax_struct)

