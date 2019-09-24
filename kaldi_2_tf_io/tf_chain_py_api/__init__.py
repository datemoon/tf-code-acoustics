import imp
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops.nn_grad import _BroadcastMul

lib_file = imp.find_module('chainloss', __path__)[1]
_warpchain = tf.load_op_library(lib_file)

def chainloss(inputs, 
        indexs, in_labels, weights, statesinfo, num_states,
        label_dim, 
        den_indexs, den_in_labels, den_weights, den_statesinfo, den_num_states,
        den_start_state = 0 ,delete_laststatesuperfinal = True,
        l2_regularize = 0.0, leaky_hmm_coefficient = 0.0, xent_regularize =0.0,
        time_major = True):
    '''Calculates the Chain loss for each batch entry.
    Also calculates the gradient.

    Args:
         inputs:  A 3-D Tensor of floats. The dimensions
                  (max_time, batch_size, num_classes),
                  the nnet forward logits weihout softmax and logit.
         indexs: A 3-D Tensor of ints, The dimensions (batch_size, arc_num, 2)
                 indexs(i, :) == [b, instate, tostate]
                 means lattice arc instate and tostate
         in_labels: A 2-D Tensor of ints, The dimensions should be
                    (batch_size, arc_num)
         weights: A 2-D Tensor of floats, The dimensions should be
                  (batch_size, arc_num)
         statesinfo: A 3-D Tensor of ints,
                     The dimensions (batch_size, state_num, 2),
                     statesinfo(i, :) == [b, offset, arc_num], the i state offset and arc number.
         num_states: A 1-D Tensor of ints, one taining sequence state number (batch).
    '''
    if not time_major:
        inputs = array_ops.transpose(inputs, [1, 0, 2])  # (B,T,N) => (T,B,N)

    loss, _ = _warpchain.chain_loss(inputs, indexs, in_labels, weights, statesinfo, num_states,
            label_dim, 
            den_indexs, den_in_labels, den_weights, den_statesinfo, den_num_states,
            den_start_state, delete_laststatesuperfinal,
            l2_regularize, leaky_hmm_coefficient, xent_regularize)

    return loss

@ops.RegisterGradient("ChainLoss")
def _ChainLossGrad(op, grad_loss, _):
    grad = op.outputs[1]
    return [ grad,
            None, None, None, None, None, ]

def chainxentloss(inputs, input_xent,
        indexs, in_labels, weights, statesinfo, num_states,
        label_dim, 
        den_indexs, den_in_labels, den_weights, den_statesinfo, den_num_states,
        den_start_state = 0 ,delete_laststatesuperfinal = True,
        l2_regularize = 0.0, leaky_hmm_coefficient = 0.0, xent_regularize =0.0,
        time_major = True):
    '''Calculates the Chain loss for each batch entry.
    Also calculates the gradient.

    Args:
         inputs:  A 3-D Tensor of floats. The dimensions
                  (max_time, batch_size, num_classes),
                  the nnet forward logits weihout softmax and logit.
         input_xent : A 3-D Tensor of floats. shape as inputs
         indexs: A 3-D Tensor of ints, The dimensions (batch_size, arc_num, 2)
                 indexs(i, :) == [b, instate, tostate]
                 means lattice arc instate and tostate
         in_labels: A 2-D Tensor of ints, The dimensions should be
                    (batch_size, arc_num)
         weights: A 2-D Tensor of floats, The dimensions should be
                  (batch_size, arc_num)
         statesinfo: A 3-D Tensor of ints,
                     The dimensions (batch_size, state_num, 2),
                     statesinfo(i, :) == [b, offset, arc_num], the i state offset and arc number.
         num_states: A 1-D Tensor of ints, one taining sequence state number (batch).
    '''
    if not time_major:
        inputs = array_ops.transpose(inputs, [1, 0, 2])  # (B,T,N) => (T,B,N)
        input_xent = array_ops.transpose(input_xent, [1, 0, 2])

    loss, _ = _warpchain.chain_xent_loss(inputs, input_xent,
            indexs, in_labels, weights, statesinfo, num_states,
            label_dim, 
            den_indexs, den_in_labels, den_weights, den_statesinfo, den_num_states,
            den_start_state, delete_laststatesuperfinal,
            l2_regularize, leaky_hmm_coefficient, xent_regularize)

    return loss

@ops.RegisterGradient("ChainXentLoss")
def _ChainXentLossGrad(op, grad_loss, _):
    grad = op.outputs[1]
    grad_xent = op.outputs[2]
    return [ grad, grad_xent,
            None, None, None, None, None, ]

