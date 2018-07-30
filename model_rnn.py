from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.python.ops import rnn_cell
import collections

from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope as vs

from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops.math_ops import tanh

from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest
import tensorflow as tf
from tensorflow.python.ops.rnn_cell import RNNCell


def _state_size_with_prefix(state_size, prefix=None):
    """Helper function that enables int or TensorShape shape specification.

    This function takes a size specification, which can be an integer or a
    TensorShape, and converts it into a list of integers. One may specify any
    additional dimensions that precede the final state size specification.

    Args:
      state_size: TensorShape or int that specifies the size of a tensor.
      prefix: optional additional list of dimensions to prepend.

    Returns:
      result_state_size: list of dimensions the resulting tensor size.
    """
    result_state_size = tensor_shape.as_shape(state_size).as_list()
    if prefix is not None:
        if not isinstance(prefix, list):
            raise TypeError("prefix of _state_size_with_prefix should be a list.")
        result_state_size = prefix + result_state_size
    return result_state_size


_LSTMStateTuple = collections.namedtuple("LSTMStateTuple", ("c", "h"))


class LSTMStateTuple(_LSTMStateTuple):
    """Tuple used by LSTM Cells for `state_size`, `zero_state`, and output state.

    Stores two elements: `(c, h)`, in that order.

    Only used when `state_is_tuple=True`.
    """
    __slots__ = ()

    @property
    def dtype(self):
        (c, h) = self
        if not c.dtype == h.dtype:
            raise TypeError("Inconsistent internal state: %s vs %s" %
                            (str(c.dtype), str(h.dtype)))
        return c.dtype


class BasicLSTMCell(RNNCell):
    """Basic LSTM recurrent network cell.

    The implementation is based on: http://arxiv.org/abs/1409.2329.

    We add forget_bias (default: 1) to the biases of the forget gate in order to
    reduce the scale of forgetting in the beginning of the training.

    It does not allow cell clipping, a projection layer, and does not
    use peep-hole connections: it is the basic baseline.

    For advanced models, please use the full LSTMCell that follows.
    """

    def __init__(self, num_units, num_cand=10, forget_bias=1.0,
                 state_is_tuple=False, activation=tanh):
        """Initialize the basic LSTM cell.

        Args:
          num_units: int, The number of units in the LSTM cell.
          forget_bias: float, The bias added to forget gates (see above).
          input_size: Deprecated and unused.
          state_is_tuple: If True, accepted and returned states are 2-tuples of
            the `c_state` and `m_state`.  By default (False), they are concatenated
            along the column axis.  This default behavior will soon be deprecated.
          activation: Activation function of the inner states.
        """
        if not state_is_tuple:
            logging.warn("%s: Using a concatenated state is slower and will soon be "
                         "deprecated.  Use state_is_tuple=True.", self)
        self._num_units = num_units
        self._forget_bias = forget_bias
        self._state_is_tuple = state_is_tuple
        self._activation = activation
        self._num_cand = num_cand

    @property
    def state_size(self):
        return (LSTMStateTuple(self._num_units, self._num_units)
                if self._state_is_tuple else 2 * self._num_units)

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        """Long short-term memory cell (LSTM)."""
        with vs.variable_scope(scope or type(self).__name__):  # "BasicLSTMCell"
            # Parameters of gates are concatenated into one multiply for efficiency.
            if self._state_is_tuple:
                # batch_size X num_cand * hidden_size, batch_size X num_cand * hidden_size
                c, h = state
            else:
                c, h = array_ops.split(1, 2, state)
            hidden_size = int(self._num_units / self._num_cand)
            # batch_size X num_cand X hidden_size
            h_flat = array_ops.reshape(h, [-1, self._num_cand, hidden_size])
            # batch_size X hidden_size
            h_sum = tf.reduce_mean(h_flat, reduction_indices=1)
            # batch_size * num_cand X hidden_size
            with vs.variable_scope('forget_gate'):
                W_f = tf.get_variable("W_f", [hidden_size, hidden_size])
                U_f = tf.get_variable("U_f", [hidden_size, hidden_size])
                b_f = tf.get_variable("b_f", [hidden_size],
                                      initializer=tf.constant_initializer(1.0))
                f_x = tf.matmul(tf.reshape(inputs, [-1, hidden_size]), W_f)
                f_h = tf.reshape(tf.matmul(tf.reshape(h_flat,
                                                      [-1, hidden_size]),
                                           U_f),
                                 [-1, 1, self._num_cand, hidden_size])
                f_x = tf.tile(tf.reshape(f_x,
                                         [-1, self._num_cand, 1, hidden_size]),
                              [1, 1, self._num_cand, 1])
                # batch_size * num_cand * num_cand * hidden_size
                f_xh = sigmoid(f_x + f_h + b_f)
            # f_x = _linear(array_ops.reshape(inputs, [-1, hidden_size]),
            #               hidden_size, True, 1.0, 'InputGate1')
            # batch_size * num_cand X hidden_size
            # f_h = array_ops.reshape(_linear(array_ops.reshape(h,
            #                                            [-1, hidden_size]),
            #                                 hidden_size, True, 1.0, 'InputGate2'),
            #                             [-1, self._num_cand, hidden_size])
            #
            # f_x = array_ops.reshape(array_ops.tile(f_x, [1, self._num_cand]),
            #                         [-1, self._num_cand, self._num_cand, hidden_size])
            # f_h = array_ops.reshape(array_ops.tile(f_h, [1, self._num_cand, 1]),
            #                         [-1, self._num_cand, self._num_cand, hidden_size])
            # f_xh = array_ops.transpose(f_x + f_h, [1, 0, 2, 3])

            with vs.variable_scope('update'):
                W_a = tf.get_variable("W_a", [hidden_size, hidden_size * 3])
                U_a = tf.get_variable("U_a", [hidden_size, hidden_size * 3])
                b_in = tf.get_variable("b_in", [hidden_size])
                b_o = tf.get_variable("b_o", [hidden_size])
                b_u = tf.get_variable("b_u", [hidden_size])
                a_x = tf.matmul(tf.reshape(inputs, [-1, hidden_size]), W_a)
                a_h = tf.matmul(h_sum, U_a)
                i_x, o_x, u_x = tf.split(1, 3, a_x)
                i_h, o_h, u_h = tf.split(1, 3, a_h)
                # batch_size X num_cand X hidden_size
                i_xh = sigmoid(tf.reshape(i_x, [-1, self._num_cand, hidden_size]) \
                        + tf.reshape(i_h, [-1, 1, hidden_size]) + b_in)
                o_xh = sigmoid(tf.reshape(o_x, [-1, self._num_cand, hidden_size]) \
                       + tf.reshape(o_h, [-1, 1, hidden_size]) + b_o)
                u_xh = tanh(tf.reshape(u_x, [-1, self._num_cand, hidden_size]) \
                       + tf.reshape(u_h, [-1, 1, hidden_size]) + b_u)

            #     o_x = _linear(array_ops.reshape(inputs, [-1, hidden_size]), 3 * hidden_size, True)
            # with vs.variable_scope('UpdateGates2'):
            #     o_h = _linear(array_ops.reshape(h_sum, [-1, hidden_size]), 3 * hidden_size, True)
                #  batch_size * num_cand * hidden_size
            # o = array_ops.reshape(o_x, [-1, self._num_cand, 3 * hidden_size]) \
            #       + array_ops.reshape(tf.tile(o_h, [1, self._num_cand]),
            #                    [-1, self._num_cand, 3 * hidden_size])
            # i_xh, o_xh, u_xh = array_ops.split(2, 3, o)
            # i = input_gate, j = new_input, f = forget_gate, o = output_gate
            new_c =tf.reduce_sum(
                array_ops.reshape(c,[-1, 1, self._num_cand, hidden_size])
                * f_xh, reduction_indices=2) + i_xh * u_xh
            new_h = tanh(new_c) * o_xh
            new_c = array_ops.reshape(new_c, [-1, hidden_size * self._num_cand])
            new_h = array_ops.reshape(new_h, [-1, hidden_size * self._num_cand])
            if self._state_is_tuple:
                new_state = LSTMStateTuple(new_c, new_h)
            else:
                new_state = array_ops.concat(1, [new_c, new_h])
            return new_h, new_state


def _linear(args, output_size, bias, bias_start=0.0, scope=None):
    """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.

    Args:
      args: a 2D Tensor or a list of 2D, batch x n, Tensors.
      output_size: int, second dimension of W[i].
      bias: boolean, whether to add a bias term or not.
      bias_start: starting value to initialize the bias; 0 by default.
      scope: VariableScope for the created subgraph; defaults to "Linear".

    Returns:
      A 2D Tensor with shape [batch x output_size] equal to
      sum_i(args[i] * W[i]), where W[i]s are newly created matrices.

    Raises:
      ValueError: if some of the arguments has unspecified or wrong shape.
    """
    if args is None or (nest.is_sequence(args) and not args):
        raise ValueError("`args` must be specified")
    if not nest.is_sequence(args):
        args = [args]

    # Calculate the total size of arguments on dimension 1.
    total_arg_size = 0
    shapes = [a.get_shape().as_list() for a in args]
    for shape in shapes:
        if len(shape) != 2:
            raise ValueError("Linear is expecting 2D arguments: %s" % str(shapes))
        if not shape[1]:
            raise ValueError("Linear expects shape[1] of arguments: %s" % str(shapes))
        else:
            total_arg_size += shape[1]

    dtype = [a.dtype for a in args][0]

    # Now the computation.
    with vs.variable_scope(scope or "Linear"):
        matrix = vs.get_variable(
            "Matrix", [total_arg_size, output_size], dtype=dtype)
        if len(args) == 1:
            res = math_ops.matmul(args[0], matrix)
        else:
            res = math_ops.matmul(array_ops.concat(1, args), matrix)
        if not bias:
            return res
        bias_term = vs.get_variable(
            "Bias", [output_size],
            dtype=dtype,
            initializer=init_ops.constant_initializer(
                bias_start, dtype=dtype))
    return res + bias_term


class GRUCellAttn(rnn_cell.GRUCell):
    def __init__(self, num_units, encoder_len, encoder_output, encoder_mask, scope=None):
        self.hs = encoder_output
        self.mask = tf.cast(encoder_mask, tf.bool)
        self.encoder_len = encoder_len
        with vs.variable_scope(scope or type(self).__name__):
            with vs.variable_scope("Attn1"):
                hs2d = array_ops.reshape(self.hs, [-1, 2 * num_units])
                phi_hs2d = tanh(rnn_cell._linear(hs2d, num_units, True, 1.0))
                self.phi_hs = array_ops.reshape(phi_hs2d, [self.encoder_len, -1, num_units])
        super(GRUCellAttn, self).__init__(num_units)

    def __call__(self, inputs, state, scope=None):
        gru_out, gru_state = super(GRUCellAttn, self).__call__(inputs, state, scope)
        with vs.variable_scope(scope or type(self).__name__):
            with vs.variable_scope("Attn2"):
                gamma_h = tanh(rnn_cell._linear(gru_out, self._num_units,
                                                True, 1.0))
            weights = tf.reduce_sum(self.phi_hs * gamma_h,
                                    reduction_indices=2)
            weights = tf.select(self.mask, weights, tf.ones_like(weights) * (-2 ** 32 + 1))
            weights = tf.transpose(tf.nn.softmax(tf.transpose(weights)))
            context = tf.reduce_sum(self.hs * tf.reshape(weights,
                                                         [self.encoder_len, -1, 1]),
                                    reduction_indices=0)
            with vs.variable_scope("AttnConcat"):
                out = tf.nn.relu(rnn_cell._linear([context, gru_out], self._num_units, True, 1.0))
            return (out, out)

    def beam(self, inputs, state, beam_size, batch_size, scope=None):
        gru_out, gru_state = super(GRUCellAttn, self).__call__(inputs, state, scope)
        with vs.variable_scope(scope or type(self).__name__):
            with vs.variable_scope("Attn2"):
                gamma_h = tanh(rnn_cell._linear(gru_out, self._num_units,
                                                True, 1.0))
                gamma_h  = tf.reshape(gamma_h, [batch_size, -1, self._num_units])
            self.phi_hs = tf.transpose(self.phi_hs, [1, 2, 0])
            weights = tf.batch_matmul(gamma_h, self.phi_hs)
            weights = tf.select(self.mask, weights, tf.ones_like(weights) * (-2 ** 32 + 1))
            weights = tf.reshape(tf.nn.softmax(tf.reshape(weights,
                                                          [batch_size * beam_size, -1])),
                                 [batch_size, beam_size, -1])
            context = tf.batch_matmul(weights, tf.transpose(self.hs, [1, 0, 2]))
            context = tf.reshape(context, [batch_size * beam_size, 2 * self._num_units])
            with vs.variable_scope("AttnConcat"):
                out = tf.nn.relu(rnn_cell._linear([context, gru_out], self._num_units, True, 1.0))
            return (out, out)

class GRUCellAttn2(rnn_cell.GRUCell):
    def __init__(self, num_units, encoder_len, encoder_output, encoder_mask, scope=None):
        self.hs = encoder_output
        self.mask = tf.cast(encoder_mask, tf.bool)
        self.encoder_len = encoder_len
        with vs.variable_scope(scope or type(self).__name__):
            with vs.variable_scope("Attn1"):
                hs2d = array_ops.reshape(self.hs, [-1, 2 * num_units])
                phi_hs2d = tanh(rnn_cell._linear(hs2d, num_units, True, 1.0))
                self.phi_hs = array_ops.reshape(phi_hs2d, array_ops.shape(self.hs))
        super(GRUCellAttn2, self).__init__(num_units)

    def __call__(self, inputs, state, scope=None):
        gru_out, gru_state = super(GRUCellAttn2, self).__call__(inputs, state, scope)
        with vs.variable_scope(scope or type(self).__name__):
            with vs.variable_scope("Attn2"):
                gamma_h = tanh(rnn_cell._linear(gru_out, self._num_units,
                                                True, 1.0))
            weights = tf.reduce_sum(self.phi_hs * gamma_h,
                                    reduction_indices=2)
            weights = tf.select(self.mask, weights, tf.ones_like(weights) * (-2 ** 32 + 1))
            weights = tf.transpose(tf.nn.softmax(tf.transpose(weights)))
            context = tf.reduce_sum(self.hs * tf.reshape(weights,
                                                         [self.encoder_len, -1, 1]),
                                    reduction_indices=0)
            with vs.variable_scope("AttnConcat"):
                out = tf.nn.relu(rnn_cell._linear([context, gru_out], self._num_units, True, 1.0))
            return (out, out)

    def beam(self, inputs, state, beam_size, batch_size, scope=None):
        gru_out, gru_state = super(GRUCellAttn2, self).__call__(inputs, state, scope)
        with vs.variable_scope(scope or type(self).__name__):
            with vs.variable_scope("Attn2"):
                gamma_h = tanh(rnn_cell._linear(gru_out, self._num_units,
                                                True, 1.0))
                gamma_h  = tf.reshape(gamma_h, [batch_size, -1, self._num_units])
            self.phi_hs = tf.transpose(self.phi_hs, [1, 2, 0])
            weights = tf.batch_matmul(gamma_h, self.phi_hs)
            weights = tf.select(self.mask, weights, tf.ones_like(weights) * (-2 ** 32 + 1))
            weights = tf.reshape(tf.nn.softmax(tf.reshape(weights,
                                                          [batch_size * beam_size, -1])),
                                 [batch_size, beam_size, -1])
            context = tf.batch_matmul(weights, tf.transpose(self.hs, [1, 0, 2]))
            context = tf.reshape(context, [batch_size * beam_size, self._num_units])
            with vs.variable_scope("AttnConcat"):
                out = tf.nn.relu(rnn_cell._linear([context, gru_out], self._num_units, True, 1.0))
            return (out, out)