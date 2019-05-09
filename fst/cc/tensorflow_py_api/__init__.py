import imp
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops.nn_grad import _BroadcastMul

lib_file = imp.find_module('mmi', __path__)[1]
_warpmmi = tf.load_op_library(lib_file)
#_warpmmi = tf.load_op_library('../tensorflow_api/tf_mmi_api.so')


def mmi(inputs, sequence_length, labels, 
        indexs, pdf_values, lm_ws, am_ws, statesinfo, num_states,
        old_acoustic_scale = 0.0,
        acoustic_scale = 1.0, drop_frames = True, time_major = True):
    '''Calculates the MMI Loss (log probability) for each batch entry.  
    Also calculates the gradient.
    
    Args:
        inputs:  A 3-D Tensor of floats. The dimensions 
                 (max_time, batch_size, num_classes), 
                 the nnet forward logits weihout softmax and logit.
        sequence_length: A 1-D Tensor of ints, 
                         one taining sequence lengths (batch).
        labels: A 2-D Tensor of ints, The dimensions (batch_size, max_time)
                a concatenation of all the labels for the minibatch.
        indexs: A 3-D Tensor of ints, The dimensions (batch_size, arc_num, 2)
                 indexs(i, :) == [b, instate, tostate] 
                 means lattice arc instate and tostate
        pdf_values: A 2-D Tensor of ints, The dimensions should be 
                    (batch_size, arc_num)
        lm_ws: A 2-D Tensor of floats, The dimensions should be
               (batch_size, arc_num)
        am_ws: A 2-D Tensor of floats, The dimensions should be
                (batch_size, arc_num)
        statesinfo: A 3-D Tensor of ints, 
                    The dimensions (batch_size, state_num, 2),
                    statesinfo(i, :) == [b, offset, arc_num], the i state offset and arc number.
        num_states: A 1-D Tensor of ints, one taining sequence state number (batch).
    '''
    # For internal calculations, we transpose to [time, batch, num_classes]
    if not time_major:
        inputs = array_ops.transpose(inputs, [1, 0, 2])  # (B,T,N) => (T,B,N)

    loss, _ = _warpmmi.mmi_loss(inputs, sequence_length, labels,
            indexs, pdf_values, lm_ws, am_ws, statesinfo, num_states,
            old_acoustic_scale=old_acoustic_scale,
            acoustic_scale=acoustic_scale,
            drop_frames=drop_frames)

    return loss

@ops.RegisterGradient("MMILoss")
def _MMILossGrad(op, grad_loss, _):
    grad = op.outputs[1]
    #return [_BroadcastMul(grad_loss, grad), 
    return [ grad, 
            None, None, None, None, None, None, None, None]

def mpe(inputs, sequence_length, labels, 
        indexs, pdf_values, lm_ws, am_ws, statesinfo, num_states,
        silence_phones = [-1],
        pdf_to_phone = [[0, 0]],
        one_silence_class = True,
        criterion = 'smbr',
        old_acoustic_scale = 0.0,
        acoustic_scale = 1.0, time_major = True):
    '''Calculates the MMI Loss (log probability) for each batch entry.  
    Also calculates the gradient.
    
    Args:
        inputs:  A 3-D Tensor of floats. The dimensions 
                 (max_time, batch_size, num_classes), 
                 the nnet forward logits weihout softmax and logit.
        sequence_length: A 1-D Tensor of ints, 
                         one taining sequence lengths (batch).
        labels: A 2-D Tensor of ints, The dimensions (batch_size, max_time)
                a concatenation of all the labels for the minibatch.
        indexs: A 3-D Tensor of ints, The dimensions (batch_size, arc_num, 2)
                 indexs(i, :) == [b, instate, tostate] 
                 means lattice arc instate and tostate
        pdf_values: A 2-D Tensor of ints, The dimensions should be 
                    (batch_size, arc_num)
        lm_ws: A 2-D Tensor of floats, The dimensions should be
               (batch_size, arc_num)
        am_ws: A 2-D Tensor of floats, The dimensions should be
                (batch_size, arc_num)
        statesinfo: A 3-D Tensor of ints, 
                    The dimensions (batch_size, state_num, 2),
                    statesinfo(i, :) == [b, offset, arc_num], the i state offset and arc number.
        num_states: A 1-D Tensor of ints, one taining sequence state number (batch).
    '''
    # For internal calculations, we transpose to [time, batch, num_classes]
    if not time_major:
        inputs = array_ops.transpose(inputs, [1, 0, 2])  # (B,T,N) => (T,B,N)

    pdf_to_phone = tf.contrib.util.make_tensor_proto(pdf_to_phone)
    loss, _ = _warpmmi.mpe_loss(inputs, sequence_length, labels,
            indexs, pdf_values, lm_ws, am_ws, statesinfo, num_states,
            silence_phones = silence_phones,
            pdf_to_phone=pdf_to_phone,
            one_silence_class=one_silence_class,
            criterion=criterion,
            old_acoustic_scale=old_acoustic_scale,
            acoustic_scale=acoustic_scale)

    return loss

@ops.RegisterGradient("MPELoss")
def _MPELossGrad(op, grad_loss, _):
    grad = op.outputs[1]
    #return [_BroadcastMul(grad_loss, grad), 
    return [ grad, 
            None, None, None, None, None, None, None, None]


#@ops.RegisterShape("MMILoss")
#def _MMILossShape(op):
#    inputs_shape = op.inputs[0].get_shape().with_rank(3)
#    batch_size = inputs_shape[1]
#    return [batch_size, inputs_shape]

