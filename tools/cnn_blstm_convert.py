import sys
from convert_tfmodel2kaldi import *

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(sys.argv[0] + ' tf.cnn_blstm kaldi.cnn_blstm')
        sys.exit(1)
    cnn_struct = ['cnn2d_0/cnn2d_0', 11, 9 , 1, 192]
    cnn_kaldi_para = [781, 6144, 9, 2, 71]
    maxpool_struct = [6144, 1536, 4, 4, 192]
    sigmoid_struct = 1536
    blstm_struct = [
            [['cell_0/bidirectional_rnn/fw', 'cell_0/bidirectional_rnn/bw'], [[1536, 1024, 512], [1536, 1024, 512]]],
            [['cell_1/bidirectional_rnn/fw', 'cell_1/bidirectional_rnn/bw'], [[1024, 1024, 512], [1024, 1024, 512]]],
            [['cell_2/bidirectional_rnn/fw', 'cell_2/bidirectional_rnn/bw'], [[1024, 1024, 512], [1024, 1024, 512]]],
            [['cell_3/bidirectional_rnn/fw', 'cell_3/bidirectional_rnn/bw'], [[1024, 1024, 512], [1024, 1024, 512]]]
            ]
    affine_structs = [['affine1',[1024, 1024]],
            ['affine2',[1024, 1024]],
            ['affine3',[1024, 6293]]]

    model_in_tf = sys.argv[1]
    model_out_kaldi = sys.argv[2]

    fp = open(model_in_tf, 'r')
    fp_out = open(model_out_kaldi, 'w')
    token = '<Nnet> \n'
    fp_out.write(token)
    # read and write cnn
    cnn_para = ConvertCnnLayer(fp, cnn_struct[0], cnn_struct[1:])
    WriteCnnLayer(fp_out, cnn_para, cnn_kaldi_para)

    # write maxpool
    WriteMaxPool(fp_out, maxpool_struct)

    # write sigmoid
    WriteSigmoid(fp_out, sigmoid_struct)

    # read and write blstm
    blstm_weights = []
    for layer_para in blstm_struct:
        print(str(layer_para))
        weight_para = WriteBlstm(fp, fp_out, layer_para)
        blstm_weights.append(weight_para)

    for affine_struct in affine_structs:
        # read Affine
        w,b = ConvertAffineTransfromLayer(fp, affine_struct[0], affine_struct[1])
        # write Affine
        WriteAffineTransfrom(fp_out, w, b, affine_struct[1][0], affine_struct[1][1])

    # write softmax
    WriteSoftmax(fp_out, affine_struct[1][1])
    token = '</Nnet> ' 
    fp_out.write(token) 
    fp.close()
    fp_out.close() 
    print('cnn blstm convert kaldi ok')


