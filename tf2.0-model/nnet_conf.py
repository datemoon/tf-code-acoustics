

'''
nnet_conf = [
        ['AffineTransformLayer', {'units':1024, 'activation':None, 'use_bias':True}],
        ['SpliceLayer', {'splice':[-1,0,1], 'time_major':True, 'splice_padding':False}],
        ['LstmLayer', {'units':1024, 'activation':'tanh', 'recurrent_activation':'sigmoid', 'use_bias':True, 'time_major':True}],
        ['BLstmLayer', {'layer': {'config': {'units':1024, 'activation':'tanh', 'recurrent_activation':'sigmoid', 'use_bias':True}}, 'merge_mode':'concat'}],
        ['Sigmoid', {}],
        ['Softmax', {'axis':-1}],
        ['NormalizeLayer', {'input_dim':1024}],
        ['ReluLayer', {'max_value':None}]
        ]
'''

nnet_conf = [
#        ['SpliceLayer', {'splice':[0], 'time_major':True, 'splice_padding':False}],
        ['AffineTransformLayer', {'units':1024, 'activation':None, 'use_bias':True}],
        ['ReluLayer', {'max_value':None}],
        ['NormalizeLayer', {'input_dim':1024, 'axis':-1, 'target_rms':1.0}],
        ['SpliceLayer', {'splice':[-1,0,1], 'time_major':True, 'splice_padding':False}],
        ['AffineTransformLayer', {'units':1024, 'activation':None, 'use_bias':True}],
        ['ReluLayer', {'max_value':None}],
        ['NormalizeLayer', {'input_dim':1024, 'axis':-1, 'target_rms':1.0}],
        ['SpliceLayer', {'splice':[-1,0,1], 'time_major':True, 'splice_padding':False}],
        ['AffineTransformLayer', {'units':1024, 'activation':None, 'use_bias':True}],
        ['ReluLayer', {'max_value':None}],
        ['NormalizeLayer', {'input_dim':1024, 'axis':-1, 'target_rms':1.0}],
        ['LstmLayer', {'units':1024, 'activation':'tanh', 'recurrent_activation':'sigmoid', 'use_bias':True, 'time_major':True, 'return_sequences': True}],
        ['SpliceLayer', {'splice':[-1,0,1], 'time_major':True, 'splice_padding':False}],
        ['AffineTransformLayer', {'units':1024, 'activation':None, 'use_bias':True}],
        ['ReluLayer', {'max_value':None}],
        ['NormalizeLayer', {'input_dim':1024, 'axis':-1, 'target_rms':1.0}],
        ['SpliceLayer', {'splice':[-1,0,1], 'time_major':True, 'splice_padding':False}],
        ['AffineTransformLayer', {'units':1024, 'activation':None, 'use_bias':True}],
        ['ReluLayer', {'max_value':None}],
        ['NormalizeLayer', {'input_dim':1024, 'axis':-1, 'target_rms':1.0}],
        ['LstmLayer', {'units':1024, 'activation':'tanh', 'recurrent_activation':'sigmoid', 'use_bias':True, 'time_major':True, 'return_sequences': True}],
        ['SpliceLayer', {'splice':[-1,0,1], 'time_major':True, 'splice_padding':False}],
        ['AffineTransformLayer', {'units':1024, 'activation':None, 'use_bias':True}],
        ['ReluLayer', {'max_value':None}],
        ['NormalizeLayer', {'input_dim':1024, 'axis':-1, 'target_rms':1.0}],
        ['SpliceLayer', {'splice':[-1,0,1], 'time_major':True, 'splice_padding':False}],
        ['AffineTransformLayer', {'units':1024, 'activation':None, 'use_bias':True}],
        ['ReluLayer', {'max_value':None}],
        ['NormalizeLayer', {'input_dim':1024, 'axis':-1, 'target_rms':1.0}],
        ['LstmLayer', {'units':1024, 'activation':'tanh', 'recurrent_activation':'sigmoid', 'use_bias':True, 'time_major':True, 'return_sequences': True}],
        ['AffineTransformLayer', {'units':3766, 'activation':None, 'use_bias':True}]
        ]

