
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

def lc_bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs, sequence_length=None,
                              initial_state_fw=None, initial_state_bw=None, 
                              latency_controlled=None, 
                              dtype=None, parallel_iterations=None,
                              swap_memory=False, time_major=False, scope=None):
  """Creates a dynamic version of bidirectional recurrent neural network.

  Takes input and builds independent forward and backward RNNs. The input_size
  of forward and backward cell must match. The initial state for both directions
  is zero by default (but can be set optionally) and no intermediate states are
  ever returned -- the network is fully unrolled for the given (passed in)
  length(s) of the sequence(s) or completely unrolled if length(s) is not
  given.

  Args:
    cell_fw: An instance of RNNCell, to be used for forward direction.
    cell_bw: An instance of RNNCell, to be used for backward direction.
    inputs: The RNN inputs.
      If time_major == False (default), this must be a tensor of shape:
        `[batch_size, max_time, ...]`, or a nested tuple of such elements.
      If time_major == True, this must be a tensor of shape:
        `[max_time, batch_size, ...]`, or a nested tuple of such elements.
    sequence_length: (optional) An int32/int64 vector, size `[batch_size]`,
      containing the actual lengths for each of the sequences in the batch.
      If not provided, all batch entries are assumed to be full sequences; and
      time reversal is applied from time `0` to `max_time` for each sequence.
    initial_state_fw: (optional) An initial state for the forward RNN.
      This must be a tensor of appropriate type and shape
      `[batch_size, cell_fw.state_size]`.
      If `cell_fw.state_size` is a tuple, this should be a tuple of
      tensors having shapes `[batch_size, s] for s in cell_fw.state_size`.
    initial_state_bw: (optional) Same as for `initial_state_fw`, but using
      the corresponding properties of `cell_bw`.
    dtype: (optional) The data type for the initial states and expected output.
      Required if initial_states are not provided or RNN states have a
      heterogeneous dtype.
    parallel_iterations: (Default: 32).  The number of iterations to run in
      parallel.  Those operations which do not have any temporal dependency
      and can be run in parallel, will be.  This parameter trades off
      time for space.  Values >> 1 use more memory but take less time,
      while smaller values use less memory but computations take longer.
    swap_memory: Transparently swap the tensors produced in forward inference
      but needed for back prop from GPU to CPU.  This allows training RNNs
      which would typically not fit on a single GPU, with very minimal (or no)
      performance penalty.
    time_major: The shape format of the `inputs` and `outputs` Tensors.
      If true, these `Tensors` must be shaped `[max_time, batch_size, depth]`.
      If false, these `Tensors` must be shaped `[batch_size, max_time, depth]`.
      Using `time_major = True` is a bit more efficient because it avoids
      transposes at the beginning and end of the RNN calculation.  However,
      most TensorFlow data is batch-major, so by default this function
      accepts input and emits output in batch-major form.
    scope: VariableScope for the created subgraph; defaults to
      "bidirectional_rnn"

  Returns:
    A tuple (outputs, output_states) where:
      outputs: A tuple (output_fw, output_bw) containing the forward and
        the backward rnn output `Tensor`.
        If time_major == False (default),
          output_fw will be a `Tensor` shaped:
          `[batch_size, max_time, cell_fw.output_size]`
          and output_bw will be a `Tensor` shaped:
          `[batch_size, max_time, cell_bw.output_size]`.
        If time_major == True,
          output_fw will be a `Tensor` shaped:
          `[max_time, batch_size, cell_fw.output_size]`
          and output_bw will be a `Tensor` shaped:
          `[max_time, batch_size, cell_bw.output_size]`.
        It returns a tuple instead of a single concatenated `Tensor`, unlike
        in the `bidirectional_rnn`. If the concatenated one is preferred,
        the forward and backward outputs can be concatenated as
        `tf.concat(outputs, 2)`.
      output_states: A tuple (output_state_fw, output_state_bw) containing
        the forward and the backward final states of bidirectional rnn.

  Raises:
    TypeError: If `cell_fw` or `cell_bw` is not an instance of `RNNCell`.
  """
  #rnn_cell_impl.assert_like_rnncell("cell_fw", cell_fw)
  #rnn_cell_impl.assert_like_rnncell("cell_bw", cell_bw)

  with tf.variable_scope(scope or "bidirectional_rnn"):
    # Forward direction
    with tf.variable_scope("fw") as fw_scope:
      if latency_controlled is None:
          sequence_length_fw = sequence_length
      else:
          latency_controlled_t = sequence_length-sequence_length+latency_controlled
          copy_cond = latency_controlled > sequence_length
          sequence_length_fw = tf.where(copy_cond, sequence_length, latency_controlled_t)

      output_fw, output_state_fw = tf.nn.dynamic_rnn(
          cell=cell_fw, inputs=inputs, sequence_length=sequence_length_fw,
          initial_state=initial_state_fw,
          dtype=dtype,
          parallel_iterations=parallel_iterations, swap_memory=swap_memory,
          time_major=time_major, scope=fw_scope)

    # Backward direction
    if not time_major:
      time_dim = 1
      batch_dim = 0
    else:
      time_dim = 0
      batch_dim = 1

    def _reverse(input_, seq_lengths, seq_dim, batch_dim):
      if seq_lengths is not None:
        return tf.reverse_sequence(
            input=input_, seq_lengths=seq_lengths,
            seq_axis=seq_dim, batch_axis=batch_dim)
      else:
        return tf.reverse(input_, axis=[seq_dim])

    with tf.variable_scope("bw") as bw_scope:
      inputs_reverse = _reverse(
          inputs, seq_lengths=sequence_length,
          seq_dim=time_dim, batch_dim=batch_dim)
      tmp, output_state_bw = tf.nn.dynamic_rnn(
          cell=cell_bw, inputs=inputs_reverse, sequence_length=sequence_length,
          initial_state=initial_state_bw, 
          dtype=dtype,
          parallel_iterations=parallel_iterations, swap_memory=swap_memory,
          time_major=time_major, scope=bw_scope)

  output_bw = _reverse(
      tmp, seq_lengths=sequence_length,
      seq_dim=time_dim, batch_dim=batch_dim)

  outputs = (output_fw, output_bw)
  output_states = (output_state_fw, output_state_bw)

  return (outputs, output_states)




def stack_bidirectional_dynamic_rnn(cells_fw,
                                    cells_bw,
                                    inputs,
                                    initial_states_fw=None,
                                    initial_states_bw=None,
                                    latency_controlled=None,
                                    dtype=None,
                                    sequence_length=None,
                                    parallel_iterations=None,
                                    time_major=False,
                                    scope=None):
  """Creates a dynamic bidirectional recurrent neural network.

  Stacks several bidirectional rnn layers. The combined forward and backward
  layer outputs are used as input of the next layer. tf.bidirectional_rnn
  does not allow to share forward and backward information between layers.
  The input_size of the first forward and backward cells must match.
  The initial state for both directions is zero and no intermediate states
  are returned.

  Args:
    cells_fw: List of instances of RNNCell, one per layer,
      to be used for forward direction.
    cells_bw: List of instances of RNNCell, one per layer,
      to be used for backward direction.
    inputs: The RNN inputs. this must be a tensor of shape:
      `[batch_size, max_time, ...]`, or a nested tuple of such elements.
    initial_states_fw: (optional) A list of the initial states (one per layer)
      for the forward RNN.
      Each tensor must has an appropriate type and shape
      `[batch_size, cell_fw.state_size]`.
    initial_states_bw: (optional) Same as for `initial_states_fw`, but using
      the corresponding properties of `cells_bw`.
    dtype: (optional) The data type for the initial state.  Required if
      either of the initial states are not provided.
    sequence_length: (optional) An int32/int64 vector, size `[batch_size]`,
      containing the actual lengths for each of the sequences.
    parallel_iterations: (Default: 32).  The number of iterations to run in
      parallel.  Those operations which do not have any temporal dependency
      and can be run in parallel, will be.  This parameter trades off
      time for space.  Values >> 1 use more memory but take less time,
      while smaller values use less memory but computations take longer.
    time_major: The shape format of the inputs and outputs Tensors. If true,
      these Tensors must be shaped [max_time, batch_size, depth]. If false,
      these Tensors must be shaped [batch_size, max_time, depth]. Using
      time_major = True is a bit more efficient because it avoids transposes at
      the beginning and end of the RNN calculation. However, most TensorFlow
      data is batch-major, so by default this function accepts input and emits
      output in batch-major form.
    scope: VariableScope for the created subgraph; defaults to None.

  Returns:
    A tuple (outputs, output_state_fw, output_state_bw) where:
      outputs: Output `Tensor` shaped:
        `batch_size, max_time, layers_output]`. Where layers_output
        are depth-concatenated forward and backward outputs.
      output_states_fw is the final states, one tensor per layer,
        of the forward rnn.
      output_states_bw is the final states, one tensor per layer,
        of the backward rnn.

  Raises:
    TypeError: If `cell_fw` or `cell_bw` is not an instance of `RNNCell`.
    ValueError: If inputs is `None`.
  """
  if not cells_fw:
    raise ValueError("Must specify at least one fw cell for BidirectionalRNN.")
  if not cells_bw:
    raise ValueError("Must specify at least one bw cell for BidirectionalRNN.")
  if not isinstance(cells_fw, list):
    raise ValueError("cells_fw must be a list of RNNCells (one per layer).")
  if not isinstance(cells_bw, list):
    raise ValueError("cells_bw must be a list of RNNCells (one per layer).")
  if len(cells_fw) != len(cells_bw):
    raise ValueError("Forward and Backward cells must have the same depth.")
  if (initial_states_fw is not None and
      (not isinstance(initial_states_fw, list) or
       len(initial_states_fw) != len(cells_fw))):
    raise ValueError(
        "initial_states_fw must be a list of state tensors (one per layer).")
  if (initial_states_bw is not None and
      (not isinstance(initial_states_bw, list) or
       len(initial_states_bw) != len(cells_bw))):
    raise ValueError(
        "initial_states_bw must be a list of state tensors (one per layer).")

  states_fw = []
  states_bw = []
  prev_layer = inputs
  
  with tf.variable_scope(scope or "stack_bidirectional_rnn"):
    for i, (cell_fw, cell_bw) in enumerate(zip(cells_fw, cells_bw)):
      initial_state_fw = None
      initial_state_bw = None
      if initial_states_fw:
        initial_state_fw = initial_states_fw[i]
      if initial_states_bw:
        initial_state_bw = initial_states_bw[i]

      with tf.variable_scope("cell_%d" % i):
        outputs, (state_fw, state_bw) = lc_bidirectional_dynamic_rnn(
            cell_fw,
            cell_bw,
            prev_layer,
            initial_state_fw=initial_state_fw,
            initial_state_bw=initial_state_bw,
            latency_controlled=latency_controlled,
            sequence_length=sequence_length,
            parallel_iterations=parallel_iterations,
            dtype=dtype,
            time_major=time_major)
        # Concat the outputs to create the new input.
        prev_layer = tf.concat(outputs, 2)
      states_fw.append(state_fw)
      states_bw.append(state_bw)

    if latency_controlled is not None:
        prev_layer = prev_layer[:latency_controlled]

  return prev_layer, tuple(states_fw), tuple(states_bw)

