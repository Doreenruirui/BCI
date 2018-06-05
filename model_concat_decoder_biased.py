# Copyright 2016 Stanford University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import variable_scope as vs
from data_generation import char2id


class NLCModel(object):
    def __init__(self,vocab_size, size, num_cand, num_layers, num_pred):

        self.size = size
        self.num_cand = num_cand
        self.num_pred = num_pred
        self.vocab_size= vocab_size
        self.num_layers = num_layers
        self.source_probs = tf.placeholder(tf.float32, shape=[None, None, None])
        self.source_mask = tf.placeholder(tf.int32, shape=[None, None])
        self.beam_size = tf.placeholder(tf.int32)
        self.len_seq = tf.shape(self.source_probs)[0]
        self.batch_size = tf.shape(self.source_probs)[1]
        with tf.variable_scope("NLC", initializer=tf.uniform_unit_scaling_initializer(1.0)):
            self.setup_embeddings()
            self.setup_decoder()
            self.setup_beam()
        self.saver = tf.train.Saver(tf.all_variables(), max_to_keep=0)

    def setup_embeddings(self):
        with vs.variable_scope("embeddings"):
            self.L_enc = tf.get_variable("L_enc", [self.vocab_size, self.size])
            self.encoder_inputs = tf.reshape(self.source_probs,
                                             [self.len_seq,
                                              self.batch_size,
                                              self.vocab_size,
                                              -1]) * \
                                  tf.reshape(self.L_enc, [1, 1, self.vocab_size, self.size])
            self.encoder_inputs = tf.reshape(self.encoder_inputs,
                                             [self.len_seq, -1, self.vocab_size * self.size])

    def setup_decoder(self):
        self.encoder_cell = rnn_cell.GRUCell(self.size)
        inp = tf.reshape(tf.slice(self.encoder_inputs, [0, 0, 0], [1, 1, -1]), [1, -1])
        with vs.variable_scope("BasicLSTM"):
            for i in xrange(self.num_layers):
                with vs.variable_scope("EncoderCell%d" % i) as scope:
                    inp, _ = self.encoder_cell(inp, tf.zeros([1, self.size]))
        with vs.variable_scope("Logistic"):
            self.logits2d = rnn_cell._linear(inp, self.vocab_size, True, 1.0)

    def decode_onestep(self, decoder_inputs, state_inputs):
        inp = decoder_inputs
        decoder_state_output = []
        with vs.variable_scope("BasicLSTM", reuse=True):
            for i in xrange(self.num_layers):
                with vs.variable_scope("EncoderCell%d" % i) as scope:
                    inp, _ = self.encoder_cell(inp, state_inputs[i])
                    decoder_state_output.append(inp)
        return inp, decoder_state_output

    def setup_beam(self):
        time_0 = tf.constant(0)
        beam_seqs_0 = tf.constant([[char2id['<zero>']]])
        beam_probs_0 = tf.constant([[0.]])
        top_seqs_0 = tf.zeros([1, self.num_pred], dtype=tf.int32)
        top_probs_0 = tf.zeros([1, self.num_pred], dtype=tf.float32)
        state_0 = tf.zeros([1, self.size])
        states_0 = [state_0] * self.num_layers

        def decode_cond(time, beam_seqs, beam_probs, top_seqs, top_probs, *states):
            return time < self.len_seq

        def decode_step(time, beam_seqs, beam_probs, top_seqs, top_probs, *states):
            batch_size = tf.shape(beam_probs)[0]
            inputs_cand = tf.reshape(tf.slice(self.encoder_inputs, [time, 0, 0], [1, 1, -1]), [1, -1])
            inputs_cand = tf.reshape(tf.tile(inputs_cand, [batch_size, 1]), [batch_size, len(char2id) * self.size])
            decoder_outputs, state_output = self.decode_onestep(inputs_cand, states)

            with vs.variable_scope("Logistic", reuse=True):
                do2d = tf.reshape(decoder_outputs, [-1, self.size])
                logits2d = rnn_cell._linear(do2d, self.vocab_size, True, 1.0)
                logprobs2d = tf.nn.log_softmax(logits2d)

            total_probs = logprobs2d + tf.reshape(beam_probs, [-1, 1])
            flat_total_probs = tf.reshape(total_probs, [-1])

            beam_k = tf.minimum(tf.size(flat_total_probs), self.beam_size)
            next_beam_probs, top_indices = tf.nn.top_k(flat_total_probs, k=beam_k)


            next_bases = tf.floordiv(top_indices, self.vocab_size)
            next_mods = tf.mod(top_indices, self.vocab_size)


            cur_probs = tf.reshape(tf.slice(next_beam_probs, [0], [self.num_pred]), [1, self.num_pred])
            cur_seqs = tf.reshape(tf.slice(next_mods, [0], [self.num_pred]), [1, self.num_pred])
            next_top_seqs = tf.concat(0, [top_seqs, cur_seqs])
            next_top_probs = tf.concat(0, [top_probs, cur_probs])

            next_states = [tf.gather(state, next_bases) for state in state_output]
            next_beam_seqs = tf.concat(1, [tf.gather(beam_seqs, next_bases),
                                           tf.reshape(next_mods, [-1, 1])])

            return [time + 1, next_beam_seqs, next_beam_probs, next_top_seqs, next_top_probs] +  next_states

        var_shape = []
        var_shape.append((time_0, time_0.get_shape()))
        var_shape.append((beam_seqs_0, tf.TensorShape([None, None])))
        var_shape.append((beam_probs_0, tf.TensorShape([None, ])))
        var_shape.append((top_seqs_0, tf.TensorShape([None, None])))
        var_shape.append((top_probs_0, tf.TensorShape([None, None])))
        var_shape.extend([(state_0, tf.TensorShape([None, self.size])) for state_0 in states_0])
        loop_vars, loop_var_shapes = zip(*var_shape)
        self.loop_vars = loop_vars
        self.loop_var_shapes = loop_var_shapes
        ret_vars = tf.while_loop(cond=decode_cond, body=decode_step, loop_vars=loop_vars, back_prop=False)
        self.vars = ret_vars
        self.beam_output = ret_vars[1]
        self.beam_scores = ret_vars[2]
        self.top_outputs = ret_vars[3]
        self.top_scores = ret_vars[4]

    def decode_beam(self, session, source_probs, source_mask, beam_size):
        input_feed = {}
        input_feed[self.source_probs] = source_probs
        input_feed[self.source_mask] = source_mask
        input_feed[self.beam_size] = beam_size
        output_feed = [self.beam_output, self.beam_scores, self.top_outputs, self.top_scores]
        outputs = session.run(output_feed, input_feed)
        return outputs[0], outputs[1], outputs[2], outputs[3]