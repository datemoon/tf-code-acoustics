
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

def as_shape(shape):
    """Converts the given object to a TensorShape."""
    if isinstance(shape, tf.TensorShape):
        return shape
    else:
        return tf.TensorShape(shape)


def _concat(prefix, suffix, static=False):
    """Concat that enables int, Tensor, or TensorShape values.
       
       This function takes a size specification, which can be an integer, a
       TensorShape, or a Tensor, and converts it into a concatenated Tensor
       (if static = False) or a list of integers (if static = True).
       
       Args:
           prefix: The prefix; usually the batch size (and/or time step size).
                   (TensorShape, int, or Tensor.)
           suffix: TensorShape, int, or Tensor.
           static: If `True`, return a python list with possibly unknown dimensions.
                   Otherwise return a `Tensor`.
       
       Returns:
           shape: the concatenation of prefix and suffix.
       
       Raises:
           ValueError: if `suffix` is not a scalar or vector (or TensorShape).
           ValueError: if prefix or suffix was `None` and asked for dynamic
                       Tensors out.
    """
    if isinstance(prefix, tf.Tensor):
        p = prefix
        p_static = tf.contrib.util.constant_value(prefix)
        if p.shape.ndims == 0:
            p = tf.expand_dims(p, 0)
        elif p.shape.ndims != 1:
            raise ValueError("prefix tensor must be either a scalar or vector, "
                    "but saw tensor: %s" % p)
    else:
        p = as_shape(prefix)
        p_static = p.as_list() if p.ndims is not None else None
        p = (
                tf.constant(p.as_list(), dtype=tf.int32)
                if p.is_fully_defined() else None)
    if isinstance(suffix, tf.Tensor):
        s = suffix
        s_static = tf.contrib.util.constant_value(suffix)
        if s.shape.ndims == 0:
            s = tf.expand_dims(s, 0)
        elif s.shape.ndims != 1:
            raise ValueError("suffix tensor must be either a scalar or vector, "
                    "but saw tensor: %s" % s)
    else:
        s = as_shape(suffix)
        s_static = s.as_list() if s.ndims is not None else None
        s = (
                tf.constant(s.as_list(), dtype=tf.int32)
                if s.is_fully_defined() else None)

    if static:
        shape = as_shape(p_static).concatenate(s_static)
        shape = shape.as_list() if shape.ndims is not None else None
    else:
        if p is None or s is None:
            raise ValueError("Provided a prefix or suffix of None: %s and %s" %
                    (prefix, suffix))
        shape = tf.concat((p, s), 0)
    return shape


def _transpose_batch_time(x):
  """Transposes the batch and time dimensions of a Tensor.

  If the input tensor has rank < 2 it returns the original tensor. Retains as
  much of the static shape information as possible.

  Args:
    x: A Tensor.

  Returns:
    x transposed along the first two dimensions.
  """
  x_static_shape = x.get_shape()
  if x_static_shape.ndims is not None and x_static_shape.ndims < 2:
    return x

  x_rank = tf.rank(x)
  x_t = tf.transpose(
      x, tf.concat(
          ([1, 0], tf.range(2, x_rank)), axis=0))
  x_t.set_shape(
      tf.TensorShape([
          x_static_shape[1].value, x_static_shape[0].value
      ]).concatenate(x_static_shape[2:]))
  return x_t


def _best_effort_input_batch_size(flat_input):
  """Get static input batch size if available, with fallback to the dynamic one.

  Args:
    flat_input: An iterable of time major input Tensors of shape
      `[max_time, batch_size, ...]`.
    All inputs should have compatible batch sizes.

  Returns:
    The batch size in Python integer if available, or a scalar Tensor otherwise.

  Raises:
    ValueError: if there is any input with an invalid shape.
  """
  for input_ in flat_input:
    shape = input_.shape
    if shape.ndims is None:
      continue
    if shape.ndims < 2:
      raise ValueError(
          "Expected input tensor %s to have rank at least 2" % input_)
    batch_size = shape[1].value
    if batch_size is not None:
      return batch_size
  # Fallback to the dynamic batch size of the first input.
  return tf.shape(flat_input[0])[1]


def _infer_state_dtype(explicit_dtype, state):
  """Infer the dtype of an RNN state.

  Args:
    explicit_dtype: explicitly declared dtype or None.
    state: RNN's hidden state. Must be a Tensor or a nested iterable containing
      Tensors.

  Returns:
    dtype: inferred dtype of hidden state.

  Raises:
    ValueError: if `state` has heterogeneous dtypes or is empty.
  """
  if explicit_dtype is not None:
    return explicit_dtype
  elif tf.contrib.framework.nest.is_sequence(state):
    inferred_dtypes = [element.dtype for element in tf.contrib.framework.nest.flatten(state)]
    if not inferred_dtypes:
      raise ValueError("Unable to infer dtype from empty state.")
    all_same = all([x == inferred_dtypes[0] for x in inferred_dtypes])
    if not all_same:
      raise ValueError(
          "State has tensors of different inferred_dtypes. Unable to infer a "
          "single representative dtype.")
    return inferred_dtypes[0]
  else:
    return state.dtype


def _maybe_tensor_shape_from_tensor(shape):
  if isinstance(shape, tf.Tensor):
    return as_shape(tf.contrib.util.constant_value(shape))
  else:
    return shape


# pylint: disable=unused-argument
def _rnn_step(
    time, sequence_length, min_sequence_length, max_sequence_length,
    zero_output, state, call_cell, state_size, 
    latency_controlled=None, last_layer=False,
    skip_conditionals=False):
  """Calculate one step of a dynamic RNN minibatch.

  Returns an (output, state) pair conditioned on `sequence_length`.
  When skip_conditionals=False, the pseudocode is something like:

  if t >= max_sequence_length:
    return (zero_output, state)
  if t < min_sequence_length:
    return call_cell()

  # Selectively output zeros or output, old state or new state depending
  # on whether we've finished calculating each row.
  new_output, new_state = call_cell()
  final_output = np.vstack([
    zero_output if time >= sequence_length[r] else new_output_r
    for r, new_output_r in enumerate(new_output)
  ])
  final_state = np.vstack([
    state[r] if time >= sequence_length[r] else new_state_r
    for r, new_state_r in enumerate(new_state)
  ])
  return (final_output, final_state)

  Args:
    time: int32 `Tensor` scalar.
    sequence_length: int32 `Tensor` vector of size [batch_size].
    min_sequence_length: int32 `Tensor` scalar, min of sequence_length.
    max_sequence_length: int32 `Tensor` scalar, max of sequence_length.
    zero_output: `Tensor` vector of shape [output_size].
    state: Either a single `Tensor` matrix of shape `[batch_size, state_size]`,
      or a list/tuple of such tensors.
    call_cell: lambda returning tuple of (new_output, new_state) where
      new_output is a `Tensor` matrix of shape `[batch_size, output_size]`.
      new_state is a `Tensor` matrix of shape `[batch_size, state_size]`.
    state_size: The `cell.state_size` associated with the state.
    skip_conditionals: Python bool, whether to skip using the conditional
      calculations.  This is useful for `dynamic_rnn`, where the input tensor
      matches `max_sequence_length`, and using conditionals just slows
      everything down.

  Returns:
    A tuple of (`final_output`, `final_state`) as given by the pseudocode above:
      final_output is a `Tensor` matrix of shape [batch_size, output_size]
      final_state is either a single `Tensor` matrix, or a tuple of such
        matrices (matching length and shapes of input `state`).

  Raises:
    ValueError: If the cell returns a state tuple whose length does not match
      that returned by `state_size`.
  """

  # Convert state to a list for ease of use
  flat_state = tf.contrib.framework.nest.flatten(state)
  flat_zero_output = tf.contrib.framework.nest.flatten(zero_output)

  # Vector describing which batch entries are finished.
  if latency_controlled is None:
    copy_cond = time >= sequence_length
    copy_cond_state = time >= sequence_length
  else:
    latency_controlled_t = tf.to_int32([ latency_controlled for _ in range(zero_output.shape[0].value)])
    #latency_controlled_t = tf.constant(latency_controlled, dtype=tf.int32, name="latency_controlled")
    # output save
    if last_layer:
      #if tf.less_equal(latency_controlled_t, time):
      #if time >= latency_controlled_t:
      #  copy_cond = 0 <= sequence_length
      #else:
      #  copy_cond = time >= sequence_length
      copy_cond = (time >= sequence_length) | (time >= latency_controlled_t)
    else:
      copy_cond = time >= sequence_length

    #if tf.less_equal(latency_controlled_t, time):
    copy_cond_state = (time >= sequence_length) | (time >= latency_controlled_t)

  def _copy_one_through(output, new_output):
    # TensorArray and scalar get passed through.
    if isinstance(output, tf.TensorArray):
      return new_output
    if output.shape.ndims == 0:
      return new_output
    # Otherwise propagate the old or the new value.
    with tf.colocate_with(new_output):
      return tf.where(copy_cond, output, new_output)

  def _copy_one_through_state(output, new_output):
    # TensorArray and scalar get passed through.
    if isinstance(output, tf.TensorArray):
      return new_output
    if output.shape.ndims == 0:
      return new_output
    # Otherwise propagate the old or the new value.
    with tf.colocate_with(new_output):
      return tf.where(copy_cond_state, output, new_output)

  def _copy_some_through(flat_new_output, flat_new_state):
    # Use broadcasting select to determine which values should get
    # the previous state & zero output, and which values should get
    # a calculated state & output.
    flat_new_output = [
        _copy_one_through(zero_output, new_output)
        for zero_output, new_output in zip(flat_zero_output, flat_new_output)]
    flat_new_state = [
        _copy_one_through_state(state, new_state)
        for state, new_state in zip(flat_state, flat_new_state)]
    return flat_new_output + flat_new_state

  def _maybe_copy_some_through():
    """Run RNN step.  Pass through either no or some past state."""
    new_output, new_state = call_cell()

    tf.contrib.framework.nest.assert_same_structure(state, new_state)

    flat_new_state = tf.contrib.framework.nest.flatten(new_state)
    flat_new_output = tf.contrib.framework.nest.flatten(new_output)
    return tf.cond(
        # if t < min_seq_len: calculate and return everything
        time < min_sequence_length, lambda: flat_new_output + flat_new_state,
        # else copy some of it through
        lambda: _copy_some_through(flat_new_output, flat_new_state))

  # TODO(ebrevdo): skipping these conditionals may cause a slowdown,
  # but benefits from removing cond() and its gradient.  We should
  # profile with and without this switch here.
  if skip_conditionals:
    # Instead of using conditionals, perform the selective copy at all time
    # steps.  This is faster when max_seq_len is equal to the number of unrolls
    # (which is typical for dynamic_rnn).
    new_output, new_state = call_cell()
    tf.contrib.framework.nest.assert_same_structure(state, new_state)
    new_state = tf.contrib.framework.nest.flatten(new_state)
    new_output = tf.contrib.framework.nest.flatten(new_output)
    final_output_and_state = _copy_some_through(new_output, new_state)
  else:
    empty_update = lambda: flat_zero_output + flat_state
    final_output_and_state = tf.cond(
        # if t >= max_seq_len: copy all state through, output zeros
        time >= max_sequence_length, empty_update,
        # otherwise calculation is required: copy some or all of it through
        _maybe_copy_some_through)

  if len(final_output_and_state) != len(flat_zero_output) + len(flat_state):
    raise ValueError("Internal error: state and output were not concatenated "
                     "correctly.")
  final_output = final_output_and_state[:len(flat_zero_output)]
  final_state = final_output_and_state[len(flat_zero_output):]

  for output, flat_output in zip(final_output, flat_zero_output):
    output.set_shape(flat_output.get_shape())
  for substate, flat_substate in zip(final_state, flat_state):
    if not isinstance(substate, tf.TensorArray):
      substate.set_shape(flat_substate.get_shape())

  final_output = tf.contrib.framework.nest.pack_sequence_as(
      structure=zero_output, flat_sequence=final_output)
  final_state = tf.contrib.framework.nest.pack_sequence_as(
      structure=state, flat_sequence=final_state)

  return final_output, final_state

def bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs, sequence_length=None,
                              initial_state_fw=None, initial_state_bw=None, 
                              latency_controlled=None, last_layer=False,
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
      latency_controlled_t = sequence_length-sequence_length+latency_controlled
      copy_cond = latency_controlled > sequence_length
      sequence_length_fw = tf.where(copy_cond, sequence_length, latency_controlled_t)

      output_fw, output_state_fw = dynamic_rnn(
          cell=cell_fw, inputs=inputs, sequence_length=sequence_length_fw,
          initial_state=initial_state_fw,
          latency_controlled=None, last_layer=False,
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
      tmp, output_state_bw = dynamic_rnn(
          cell=cell_bw, inputs=inputs_reverse, sequence_length=sequence_length,
          initial_state=initial_state_bw, 
          latency_controlled=None,last_layer=False,
          dtype=dtype,
          parallel_iterations=parallel_iterations, swap_memory=swap_memory,
          time_major=time_major, scope=bw_scope)

  output_bw = _reverse(
      tmp, seq_lengths=sequence_length,
      seq_dim=time_dim, batch_dim=batch_dim)

  outputs = (output_fw, output_bw)
  output_states = (output_state_fw, output_state_bw)

  return (outputs, output_states)

def dynamic_rnn(cell, inputs, sequence_length=None, initial_state=None, 
                latency_controlled=None,last_layer=False,
                dtype=None, parallel_iterations=None, swap_memory=False,
                time_major=False, scope=None):
  """Creates a recurrent neural network specified by RNNCell `cell`.

  Performs fully dynamic unrolling of `inputs`.

  Example:

  ```python
  # create a BasicRNNCell
  rnn_cell = tf.nn.rnn_cell.BasicRNNCell(hidden_size)

  # 'outputs' is a tensor of shape [batch_size, max_time, cell_state_size]

  # defining initial state
  initial_state = rnn_cell.zero_state(batch_size, dtype=tf.float32)

  # 'state' is a tensor of shape [batch_size, cell_state_size]
  outputs, state = tf.nn.dynamic_rnn(rnn_cell, input_data,
                                     initial_state=initial_state,
                                     dtype=tf.float32)
  ```

  ```python
  # create 2 LSTMCells
  rnn_layers = [tf.nn.rnn_cell.LSTMCell(size) for size in [128, 256]]

  # create a RNN cell composed sequentially of a number of RNNCells
  multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)

  # 'outputs' is a tensor of shape [batch_size, max_time, 256]
  # 'state' is a N-tuple where N is the number of LSTMCells containing a
  # tf.contrib.rnn.LSTMStateTuple for each cell
  outputs, state = tf.nn.dynamic_rnn(cell=multi_rnn_cell,
                                     inputs=data,
                                     dtype=tf.float32)
  ```


  Args:
    cell: An instance of RNNCell.
    inputs: The RNN inputs.
      If `time_major == False` (default), this must be a `Tensor` of shape:
        `[batch_size, max_time, ...]`, or a nested tuple of such
        elements.
      If `time_major == True`, this must be a `Tensor` of shape:
        `[max_time, batch_size, ...]`, or a nested tuple of such
        elements.
      This may also be a (possibly nested) tuple of Tensors satisfying
      this property.  The first two dimensions must match across all the inputs,
      but otherwise the ranks and other shape components may differ.
      In this case, input to `cell` at each time-step will replicate the
      structure of these tuples, except for the time dimension (from which the
      time is taken).
      The input to `cell` at each time step will be a `Tensor` or (possibly
      nested) tuple of Tensors each with dimensions `[batch_size, ...]`.
    sequence_length: (optional) An int32/int64 vector sized `[batch_size]`.
      Used to copy-through state and zero-out outputs when past a batch
      element's sequence length.  So it's more for performance than correctness.
    initial_state: (optional) An initial state for the RNN.
      If `cell.state_size` is an integer, this must be
      a `Tensor` of appropriate type and shape `[batch_size, cell.state_size]`.
      If `cell.state_size` is a tuple, this should be a tuple of
      tensors having shapes `[batch_size, s] for s in cell.state_size`.
    dtype: (optional) The data type for the initial state and expected output.
      Required if initial_state is not provided or RNN state has a heterogeneous
      dtype.
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
    scope: VariableScope for the created subgraph; defaults to "rnn".

  Returns:
    A pair (outputs, state) where:

    outputs: The RNN output `Tensor`.

      If time_major == False (default), this will be a `Tensor` shaped:
        `[batch_size, max_time, cell.output_size]`.

      If time_major == True, this will be a `Tensor` shaped:
        `[max_time, batch_size, cell.output_size]`.

      Note, if `cell.output_size` is a (possibly nested) tuple of integers
      or `TensorShape` objects, then `outputs` will be a tuple having the
      same structure as `cell.output_size`, containing Tensors having shapes
      corresponding to the shape data in `cell.output_size`.

    state: The final state.  If `cell.state_size` is an int, this
      will be shaped `[batch_size, cell.state_size]`.  If it is a
      `TensorShape`, this will be shaped `[batch_size] + cell.state_size`.
      If it is a (possibly nested) tuple of ints or `TensorShape`, this will
      be a tuple having the corresponding shapes. If cells are `LSTMCells`
      `state` will be a tuple containing a `LSTMStateTuple` for each cell.

  Raises:
    TypeError: If `cell` is not an instance of RNNCell.
    ValueError: If inputs is None or an empty list.
  """
  #rnn_cell_impl.assert_like_rnncell("cell", cell)

  with tf.variable_scope(scope or "rnn") as varscope:
    # Create a new scope in which the caching device is either
    # determined by the parent scope, or is set to place the cached
    # Variable using the same placement as for the rest of the RNN.
    if not tf.executing_eagerly():
      if varscope.caching_device is None:
        varscope.set_caching_device(lambda op: op.device)

    # By default, time_major==False and inputs are batch-major: shaped
    #   [batch, time, depth]
    # For internal calculations, we transpose to [time, batch, depth]
    flat_input = tf.contrib.framework.nest.flatten(inputs)

    if not time_major:
      # (B,T,D) => (T,B,D)
      flat_input = [tf.convert_to_tensor(input_) for input_ in flat_input]
      flat_input = tuple(_transpose_batch_time(input_) for input_ in flat_input)

    parallel_iterations = parallel_iterations or 32
    if sequence_length is not None:
      sequence_length = tf.to_int32(sequence_length)
      if sequence_length.get_shape().ndims not in (None, 1):
        raise ValueError(
            "sequence_length must be a vector of length batch_size, "
            "but saw shape: %s" % sequence_length.get_shape())
      sequence_length = tf.identity(  # Just to find it in the graph.
          sequence_length, name="sequence_length")

    batch_size = _best_effort_input_batch_size(flat_input)

    if initial_state is not None:
      state = initial_state
    else:
      if not dtype:
        raise ValueError("If there is no initial_state, you must give a dtype.")
      state = cell.zero_state(batch_size, dtype)

    def _assert_has_shape(x, shape):
      x_shape = tf.shape(x)
      packed_shape = tf.stack(shape)
      return tf.Assert(
          tf.reduce_all(tf.equal(x_shape, packed_shape)),
          ["Expected shape for Tensor %s is " % x.name,
           packed_shape, " but saw shape: ", x_shape])

    if not tf.executing_eagerly() and sequence_length is not None:
      # Perform some shape validation
      with tf.control_dependencies(
          [_assert_has_shape(sequence_length, [batch_size])]):
        sequence_length = tf.identity(
            sequence_length, name="CheckSeqLen")

    inputs = tf.contrib.framework.nest.pack_sequence_as(structure=inputs, flat_sequence=flat_input)

    (outputs, final_state) = _dynamic_rnn_loop(
        cell,
        inputs,
        state,
        parallel_iterations=parallel_iterations,
        swap_memory=swap_memory,
        sequence_length=sequence_length,
        latency_controlled=latency_controlled,last_layer=last_layer,
        dtype=dtype)

    # Outputs of _dynamic_rnn_loop are always shaped [time, batch, depth].
    # If we are performing batch-major calculations, transpose output back
    # to shape [batch, time, depth]
    if not time_major:
      # (T,B,D) => (B,T,D)
      outputs = tf.contrib.framework.nest.map_structure(_transpose_batch_time, outputs)

    return (outputs, final_state)


def _dynamic_rnn_loop(cell,
                      inputs,
                      initial_state,
                      parallel_iterations,
                      swap_memory,
                      sequence_length=None,
                      latency_controlled=None,last_layer=False,
                      dtype=None):
  """Internal implementation of Dynamic RNN.

  Args:
    cell: An instance of RNNCell.
    inputs: A `Tensor` of shape [time, batch_size, input_size], or a nested
      tuple of such elements.
    initial_state: A `Tensor` of shape `[batch_size, state_size]`, or if
      `cell.state_size` is a tuple, then this should be a tuple of
      tensors having shapes `[batch_size, s] for s in cell.state_size`.
    parallel_iterations: Positive Python int.
    swap_memory: A Python boolean
    sequence_length: (optional) An `int32` `Tensor` of shape [batch_size].
    dtype: (optional) Expected dtype of output. If not specified, inferred from
      initial_state.

  Returns:
    Tuple `(final_outputs, final_state)`.
    final_outputs:
      A `Tensor` of shape `[time, batch_size, cell.output_size]`.  If
      `cell.output_size` is a (possibly nested) tuple of ints or `TensorShape`
      objects, then this returns a (possibly nested) tuple of Tensors matching
      the corresponding shapes.
    final_state:
      A `Tensor`, or possibly nested tuple of Tensors, matching in length
      and shapes to `initial_state`.

  Raises:
    ValueError: If the input depth cannot be inferred via shape inference
      from the inputs.
  """
  state = initial_state
  assert isinstance(parallel_iterations, int), "parallel_iterations must be int"

  state_size = cell.state_size

  flat_input = tf.contrib.framework.nest.flatten(inputs)
  flat_output_size = tf.contrib.framework.nest.flatten(cell.output_size)

  # Construct an initial output
  input_shape = tf.shape(flat_input[0])
  time_steps = input_shape[0]
  batch_size = _best_effort_input_batch_size(flat_input)

  inputs_got_shape = tuple(input_.get_shape().with_rank_at_least(3)
                           for input_ in flat_input)

  const_time_steps, const_batch_size = inputs_got_shape[0].as_list()[:2]

  for shape in inputs_got_shape:
    if not shape[2:].is_fully_defined():
      raise ValueError(
          "Input size (depth of inputs) must be accessible via shape inference,"
          " but saw value None.")
    got_time_steps = shape[0].value
    got_batch_size = shape[1].value
    if const_time_steps != got_time_steps:
      raise ValueError(
          "Time steps is not the same for all the elements in the input in a "
          "batch.")
    if const_batch_size != got_batch_size:
      raise ValueError(
          "Batch_size is not the same for all the elements in the input.")

  # Prepare dynamic conditional copying of state & output
  def _create_zero_arrays(size):
    size = _concat(batch_size, size)
    return tf.zeros(
        tf.stack(size), _infer_state_dtype(dtype, state))

  flat_zero_output = tuple(_create_zero_arrays(output)
                           for output in flat_output_size)
  zero_output = tf.contrib.framework.nest.pack_sequence_as(structure=cell.output_size,
                                      flat_sequence=flat_zero_output)

  if sequence_length is not None:
    min_sequence_length = tf.reduce_min(sequence_length)
    max_sequence_length = tf.reduce_max(sequence_length)
  else:
    max_sequence_length = time_steps

  time = tf.constant(0, dtype=tf.int32, name="time")

  with tf.name_scope("dynamic_rnn") as scope:
    base_name = scope

  def _create_ta(name, element_shape, dtype):
    return tf.TensorArray(dtype=dtype,
                                        size=time_steps,
                                        element_shape=element_shape,
                                        tensor_array_name=base_name + name)

  in_graph_mode = not tf.executing_eagerly()
  if in_graph_mode:
    output_ta = tuple(
        _create_ta(
            "output_%d" % i,
            element_shape=(tf.TensorShape([const_batch_size])
                           .concatenate(
                               _maybe_tensor_shape_from_tensor(out_size))),
            dtype=_infer_state_dtype(dtype, state))
        for i, out_size in enumerate(flat_output_size))
    input_ta = tuple(
        _create_ta(
            "input_%d" % i,
            element_shape=flat_input_i.shape[1:],
            dtype=flat_input_i.dtype)
        for i, flat_input_i in enumerate(flat_input))
    input_ta = tuple(ta.unstack(input_)
                     for ta, input_ in zip(input_ta, flat_input))
  else:
    output_ta = tuple([0 for _ in range(time_steps.numpy())]
                      for i in range(len(flat_output_size)))
    input_ta = flat_input

  def _time_step(time, output_ta_t, state):
    """Take a time step of the dynamic RNN.

    Args:
      time: int32 scalar Tensor.
      output_ta_t: List of `TensorArray`s that represent the output.
      state: nested tuple of vector tensors that represent the state.

    Returns:
      The tuple (time + 1, output_ta_t with updated flow, new_state).
    """

    if in_graph_mode:
      input_t = tuple(ta.read(time) for ta in input_ta)
      # Restore some shape information
      for input_, shape in zip(input_t, inputs_got_shape):
        input_.set_shape(shape[1:])
    else:
      input_t = tuple(ta[time.numpy()] for ta in input_ta)

    input_t = tf.contrib.framework.nest.pack_sequence_as(structure=inputs, flat_sequence=input_t)
    call_cell = lambda: cell(input_t, state)

    if sequence_length is not None:
      (output, new_state) = _rnn_step(
          time=time,
          sequence_length=sequence_length,
          min_sequence_length=min_sequence_length,
          max_sequence_length=max_sequence_length,
          zero_output=zero_output,
          state=state,
          call_cell=call_cell,
          state_size=state_size,
          latency_controlled=latency_controlled,last_layer=last_layer,
          skip_conditionals=True)
    else:
      (output, new_state) = call_cell()

    # Pack state if using state tuples
    output = tf.contrib.framework.nest.flatten(output)

    if in_graph_mode:
      output_ta_t = tuple(
          ta.write(time, out) for ta, out in zip(output_ta_t, output))
    else:
      for ta, out in zip(output_ta_t, output):
        ta[time.numpy()] = out

    return (time + 1, output_ta_t, new_state)

  if in_graph_mode:
    # Make sure that we run at least 1 step, if necessary, to ensure
    # the TensorArrays pick up the dynamic shape.
    loop_bound = tf.minimum(
        time_steps, tf.maximum(1, max_sequence_length))
  else:
    # Using max_sequence_length isn't currently supported in the Eager branch.
    loop_bound = time_steps

  _, output_final_ta, final_state = tf.while_loop(
      cond=lambda time, *_: time < loop_bound,
      body=_time_step,
      loop_vars=(time, output_ta, state),
      parallel_iterations=parallel_iterations,
      maximum_iterations=time_steps,
      swap_memory=swap_memory)

  # Unpack final output if not using output tuples.
  if in_graph_mode:
    final_outputs = tuple(ta.stack() for ta in output_final_ta)
    # Restore some shape information
    for output, output_size in zip(final_outputs, flat_output_size):
      shape = _concat(
          [const_time_steps, const_batch_size], output_size, static=True)
      output.set_shape(shape)
  else:
    final_outputs = output_final_ta

  final_outputs = tf.contrib.framework.nest.pack_sequence_as(
      structure=cell.output_size, flat_sequence=final_outputs)
  if not in_graph_mode:
    final_outputs = tf.stack(final_outputs, axis=0)

  return (final_outputs, final_state)


