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

import random
from data_generate import char2id

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import variable_scope as vs
from model_rnn import GRUCellAttn
from module import label_smooth


def get_optimizer(opt):
    if opt == "adam":
        optfn = tf.train.AdamOptimizer
    elif opt == "sgd":
        optfn = tf.train.GradientDescentOptimizer
    else:
        assert(False)
    return optfn


class Model(object):
    def __init__(self, size, num_cand, num_layers, max_gradient_norm, 
                 learning_rate, learning_rate_decay,
                 forward_only=False, optimizer="adam", num_pred=5):
        self.size = size
        self.num_cand = num_cand
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.learning_decay = learning_rate_decay
        self.max_grad_norm = max_gradient_norm
        self.foward_only = forward_only
        self.optimizer = optimizer
        self.num_pred = num_pred

    def _add_place_holders(self, flag_word):
        self.keep_prob = tf.placeholder(tf.float32)
        self.src_toks = tf.placeholder(tf.float32, 
                                           shape=[None, None, None])
        self.tgt_toks = tf.placeholder(tf.int32, shape=[None, None])
        if flag_word:
            self.target_tokens_word = tf.placeholder(tf.int32, 
                                                     shape=[None, None])
        self.src_mask = tf.placeholder(tf.int32, shape=[None, None])
        self.tgt_mask = tf.placeholder(tf.int32, shape=[None, None])
        self.beam_size = tf.placeholder(tf.int32)
        self.batch_size = tf.shape(self.src_toks)[1]
        self.len_inp = tf.shape(self.src_toks)[0]

    def setup_train(self):
        self.lr = tf.Variable(float(self.learning_rate), trainable=False)
        self.lr_decay_op = self.lr.assign(
            self.lr * self.learning_decay)
        self.global_step = tf.Variable(0, trainable=False)
        params = tf.trainable_variables()
        opt = get_optimizer(self.optimizer)(self.lr)
        gradients = tf.gradients(self.losses, params)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, 
                                                      self.max_grad_norm)
        self.gradient_norm = tf.global_norm(gradients)
        self.param_norm = tf.global_norm(params)
        self.updates = opt.apply_gradients(
            zip(clipped_gradients, params), global_step=self.global_step)

    def setup_embeddings(self, vocab_size):
        with vs.variable_scope("embeddings"):
            self.L_enc = tf.get_variable("L_enc", [vocab_size, self.size])
            self.encoder_inputs = tf.reshape(self.src_toks,
                                             [self.len_inp,
                                              self.batch_size,
                                              vocab_size,
                                              -1]) * \
                                  tf.reshape(self.L_enc,
                                             [1, 1, vocab_size, self.size])
            self.encoder_inputs = tf.reshape(self.encoder_inputs,
                                             [self.len_inp, -1, 
                                              vocab_size * self.size])
            self.decoder_inputs = embedding_ops.embedding_lookup(self.L_enc, self.tgt_toks)

    def setup_encoder(self, flag_bidirect):
        with vs.variable_scope("Encoder"):
            inp = tf.nn.dropout(self.encoder_inputs, self.keep_prob)
            srclen = tf.reduce_sum(self.src_mask, reduction_indices=0)
            fw_cell = rnn_cell.GRUCell(self.size)
            fw_cell = rnn_cell.DropoutWrapper(
                fw_cell, output_keep_prob=self.keep_prob)
            self.encoder_fw_cell = rnn_cell.MultiRNNCell(
                [fw_cell] * self.num_layers, state_is_tuple=True)
            if flag_bidirect:
                bw_cell = rnn_cell.GRUCell(self.size)
                bw_cell = rnn_cell.DropoutWrapper(
                    bw_cell, output_keep_prob=self.keep_prob)
                self.encoder_bw_cell = rnn_cell.MultiRNNCell(
                    [bw_cell] * self.num_layers, state_is_tuple=True)
                out, _ = rnn.bidirectional_dynamic_rnn(self.encoder_fw_cell,
                                                       self.encoder_bw_cell,
                                                       inp, tf.cast(srclen, tf.int64),
                                                       dtype=tf.float32,
                                                       time_major=True,
                                                       initial_state_fw=self.encoder_fw_cell.zero_state(
                                                           self.batch_size, dtype=tf.float32),
                                                       initial_state_bw=self.encoder_bw_cell.zero_state(
                                                           self.batch_size, dtype=tf.float32))
                out = out[0] + out[1]
            else:
                out, _ = rnn.dynamic_rnn(self.encoder_fw_cell, inp, srclen,
                                         dtype=tf.float32, time_major=True)
            self.encoder_output = out

    def setup_decoder(self):
        with vs.variable_scope("Decoder"):
            inp = self.decoder_inputs
            tgt_len = tf.reduce_sum(self.tgt_mask, reduction_indices=0)
            if self.num_layers > 1:
                with vs.variable_scope("RNN"):
                    decoder_cell = rnn_cell.GRUCell(self.size)
                    decoder_cell = rnn_cell.DropoutWrapper(decoder_cell,
                                                           output_keep_prob=self.keep_prob)
                    self.decoder_cell = rnn_cell.MultiRNNCell(
                        [decoder_cell] * (self.num_layers - 1), state_is_tuple=True)
                    inp, _ = rnn.dynamic_rnn(self.decoder_cell, inp, tgt_len,
                                             dtype=tf.float32, time_major=True,
                                             initial_state=self.decoder_cell.zero_state(
                                                 self.batch_size, dtype=tf.float32))

            with vs.variable_scope("Attn"):
                self.attn_cell = GRUCellAttn(self.size, self.len_inp,
                                             self.encoder_output, self.src_mask)
                self.decoder_output, _ = rnn.dynamic_rnn(self.attn_cell, inp, tgt_len,
                                                         dtype=tf.float32, time_major=True,
                                                         initial_state=self.attn_cell.zero_state(
                                                             self.batch_size, dtype=tf.float32))

    def setup_loss(self, output, vocab_size):
        with vs.variable_scope("Logistic"):
            logits2d = rnn_cell._linear(tf.reshape(output,
                                                   [-1, self.size]),
                                        vocab_size, True, 1.0)
            outputs2d = tf.nn.log_softmax(logits2d)
            self.outputs = tf.arg_max(tf.reshape(outputs2d,
                                      [-1, self.batch_size, vocab_size]), 2)
            labels = tf.reshape(tf.pad(tf.slice(self.tgt_toks, [1, 0], [-1, -1]),
                                       [[0,1],[0,0]]),
                                [-1])
            self.labels = labels
            labels = label_smooth(labels, vocab_size)
            mask1d = tf.reshape(self.src_mask, [-1])
            losses1d = tf.nn.softmax_cross_entropy_with_logits(logits2d,
                                                                      labels) \
                       * tf.to_float(mask1d)
            losses2d = tf.reshape(losses1d, [self.len_inp, self.batch_size])
            self.losses = tf.reduce_sum(losses2d) / tf.to_float(self.batch_size)

    def build_model(self, vocab_size_char, model='lstm', flag_word=False, vocab_size_word=0, loss="char"):
        self._add_place_holders(flag_word)
        with tf.variable_scope("Model", initializer=tf.uniform_unit_scaling_initializer(1.0)):
            self.setup_embeddings(vocab_size_char)
            if model == 'lstm':
                self.setup_encoder(flag_bidirect=False)
                output = self.encoder_output
            else:
                self.setup_encoder(flag_bidirect=True)
                self.setup_decoder()
                output = self.decoder_output
            if flag_word:
                self.setup_loss(output, vocab_size_word)
            else:
                self.setup_loss(output, vocab_size_char)
            if self.foward_only:
                self.setup_beam("Logistic", vocab_size_char, model)
        if not self.foward_only:
            self.setup_train()

        self.saver = tf.train.Saver(tf.all_variables(), max_to_keep=0)

    def beam_step_lstm(self, time_step, batch_size, state_inputs):
        inp = tf.slice(self.encoder_inputs, [time_step, 0, 0], [1, -1, -1])
        inp = tf.reshape(tf.tile(tf.reshape(inp, [1, -1]), [batch_size, 1]), [batch_size, len(char2id) * self.size])
        with vs.variable_scope("Encoder", reuse=True):
            with vs.variable_scope('RNN', reuse=True):
                out, state_outputs = self.encoder_fw_cell(inp, state_inputs)
        return out, state_outputs

    def beam_step_seq2seq(self, time_step, batch_size, beam_seqs, state_inputs):
        inp = tf.reshape(tf.slice(beam_seqs, [0, time_step], [-1, 1]), [batch_size])
        inputs = embedding_ops.embedding_lookup(self.L_enc, inp)
        with vs.variable_scope("Decoder", reuse=True):
            with vs.variable_scope("RNN", reuse=True):
                with vs.variable_scope("RNN", reuse=True):
                    rnn_out, rnn_outputs = self.decoder_cell(inputs, state_inputs[:self.num_layers-1])
            with vs.variable_scope("Attn", reuse=True):
                with vs.variable_scope("RNN", reuse=True):
                    self.attn_cell.mask = tf.cast(tf.tile(self.src_mask, [1, batch_size]), tf.bool)
                    out, attn_outputs = self.attn_cell(rnn_out, state_inputs[-1])
        state_outputs = rnn_outputs + (attn_outputs, )
        return out, state_outputs

    def beam_loss(self, scope, output, vocab_size):
        with vs.variable_scope(scope, reuse=True):
            do2d = tf.reshape(output, [-1, self.size])
            logits2d = rnn_cell._linear(do2d, vocab_size, True, 1.0)
            logprobs2d = tf.nn.log_softmax(logits2d)
        return logprobs2d

    def setup_beam(self, loss_scope, vocab_size, model='lstm'):
        time_0 = tf.constant(0)
        beam_seqs_0 = tf.constant([[char2id['<sos>']]])
        beam_probs_0 = tf.constant([0.])
        top_seqs_0 = tf.zeros([1, self.num_pred], dtype=tf.int32)
        top_probs_0 = tf.zeros([1, self.num_pred], dtype=tf.float32)
        perplex_0 = tf.constant([0.])
        state_0 = tf.zeros([1, self.size])
        states_0 = (state_0,) * self.num_layers

        def decode_cond(time, beam_seqs, beam_probs, top_seqs, top_probs, perplex, *states):
            return time < self.len_inp

        def decode_step(time, beam_seqs, beam_probs, top_seqs, top_probs, perplex, *states):
            batch_size = tf.shape(beam_probs)[0]
            if model == 'lstm':
                decoder_outputs, state_output = self.beam_step_lstm(time, batch_size, states)
            else:
                decoder_outputs, state_output = self.beam_step_seq2seq(time, batch_size, beam_seqs, states)
            logprobs2d = self.beam_loss(loss_scope, decoder_outputs, vocab_size)

            total_probs = logprobs2d + tf.reshape(beam_probs, [-1, 1])
            flat_total_probs = tf.reshape(total_probs, [-1])

            beam_k = tf.minimum(tf.size(flat_total_probs), self.beam_size)
            next_beam_probs, top_indices = tf.nn.top_k(flat_total_probs, k=beam_k)

            next_bases = tf.floordiv(top_indices, vocab_size)
            next_mods = tf.mod(top_indices, vocab_size)

            cur_probs = tf.reshape(tf.slice(next_beam_probs, [0], [self.num_pred]), [1, self.num_pred])
            cur_seqs = tf.reshape(tf.slice(next_mods, [0], [self.num_pred]), [1, self.num_pred])
            next_top_seqs = tf.concat(0, [top_seqs, cur_seqs])
            next_top_probs = tf.concat(0, [top_probs, cur_probs])

            next_states = [tf.gather(state, next_bases) for state in state_output]
            next_beam_seqs = tf.concat(1, [tf.gather(beam_seqs, next_bases),
                                           tf.reshape(next_mods, [-1, 1])])

            cur_tok = tf.reshape(tf.slice(self.tgt_toks, [time + 1, 0], [1, -1]), ())
            perplex_prob = tf.reduce_max(tf.slice(total_probs, [0, cur_tok], [-1, 1]))
            next_perplex = tf.concat(0, [perplex,tf.reshape(perplex_prob, [-1])])

            return [time + 1, next_beam_seqs, next_beam_probs, next_top_seqs, next_top_probs, next_perplex] + next_states

        var_shape = []
        var_shape.append((time_0, time_0.get_shape()))
        var_shape.append((beam_seqs_0, tf.TensorShape([None, None])))
        var_shape.append((beam_probs_0, tf.TensorShape([None, ])))
        var_shape.append((top_seqs_0, tf.TensorShape([None, None])))
        var_shape.append((top_probs_0, tf.TensorShape([None, None])))
        var_shape.append((perplex_0, tf.TensorShape([None, ])))
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
        self.perplex = ret_vars[5]

    def decode_beam(self, session, src_toks, tgt_toks, src_mask, beam_size, num_cand):
        input_feed = {}
        input_feed[self.src_toks] = src_toks
        input_feed[self.src_mask] = src_mask
        input_feed[self.tgt_toks] = tgt_toks
        input_feed[self.keep_prob] = 1.
        input_feed[self.beam_size] = beam_size
        self.num_cand = num_cand
        output_feed = [self.beam_output, self.beam_scores, self.top_outputs, self.top_scores, self.perplex]
        outputs = session.run(output_feed, input_feed)
        return outputs[0], outputs[1], outputs[2], outputs[3], outputs[4]

    def train(self, session, src_toks, src_mask, tgt_toks, keep_prob):
        input_feed = {}
        input_feed[self.src_toks] = src_toks
        input_feed[self.tgt_toks] = tgt_toks
        input_feed[self.src_mask] = src_mask
        input_feed[self.tgt_mask] = src_mask
        input_feed[self.keep_prob] = keep_prob

        output_feed = [self.updates, self.gradient_norm, self.losses, self.param_norm]

        outputs = session.run(output_feed, input_feed)

        return outputs[1], outputs[2], outputs[3]

    def test(self, session, src_toks, src_mask, tgt_toks):
        input_feed = {}
        input_feed[self.src_toks] = src_toks
        input_feed[self.tgt_toks] = tgt_toks
        input_feed[self.src_mask] = src_mask
        input_feed[self.tgt_mask] = src_mask
        input_feed[self.keep_prob] = 1.
        output_feed = [self.losses]
        outputs = session.run(output_feed, input_feed)
        return outputs[0]
