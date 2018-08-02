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
        self.src_len = tf.cast(tf.reduce_sum(self.src_mask, reduction_indices=0), tf.int64)

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

    def setup_embeddings(self, vocab_size, model_sum=0, model='seq2seq'):
        with vs.variable_scope("embeddings"):
            zeros = tf.zeros([1, self.size])
            enc = tf.get_variable("L_enc", [vocab_size - 1, self.size])
            self.L_enc = tf.concat(0, [zeros, enc])
            dec = tf.get_variable("L_dec", [vocab_size - 1, self.size])
            self.L_dec = tf.concat(0, [zeros, dec])

            self.encoder_inputs = tf.reshape(self.src_toks,
                                             [self.len_inp,
                                              self.batch_size,
                                              vocab_size,
                                              -1]) * \
                                  tf.reshape(self.L_enc,
                                             [1, 1, vocab_size, self.size])
            embed_sos = embedding_ops.embedding_lookup(self.L_dec, char2id['<sos>'])
            embed_sos = tf.tile(tf.reshape(embed_sos, [1, 1, -1]),
                                [1, self.batch_size, 1])
            if model_sum == 1:
                self.encoder_inputs = tf.reduce_sum(self.encoder_inputs, reduction_indices=2)
            else:
                self.encoder_inputs = tf.reshape(self.encoder_inputs,
                                                 [self.len_inp, -1,
                                                  vocab_size * self.size])
                if model_sum == 2:
                    self.encoder_inputs = tf.nn.relu(rnn_cell._linear(tf.reshape(self.encoder_inputs,
                                                                      [-1, vocab_size * self.size]),
                                                           self.size, True, 1.0))
                    self.encoder_inputs = tf.reshape(self.encoder_inputs,
                                                     [self.len_inp, -1, self.size])
            self.decoder_inputs = embedding_ops.embedding_lookup(self.L_dec,
                                                                 tf.slice(self.tgt_toks,
                                                                          [0, 0],
                                                                          [self.len_inp -1, -1]))
            if model == 'seq2seq':
                self.decoder_inputs = tf.concat(0, [embed_sos, self.decoder_inputs])

    def setup_encoder(self, flag_bidirect):
        with vs.variable_scope("Encoder"):
            inp = tf.nn.dropout(self.encoder_inputs, self.keep_prob)
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
                                                       inp, self.src_len,
                                                       dtype=tf.float32,
                                                       time_major=True,
                                                       initial_state_fw=self.encoder_fw_cell.zero_state(
                                                           self.batch_size, dtype=tf.float32),
                                                       initial_state_bw=self.encoder_bw_cell.zero_state(
                                                           self.batch_size, dtype=tf.float32))
                # out = out[0] + out[1]
                out = tf.concat(2, [out[0], out[1]])
            else:
                out, _ = rnn.dynamic_rnn(self.encoder_fw_cell, inp, self.src_len,
                                         dtype=tf.float32, time_major=True)
            self.encoder_output = out

    def setup_encoder_forward(self):
        with vs.variable_scope("Encoder", reuse=None):
            inp = tf.nn.dropout(self.encoder_inputs, self.keep_prob)
            fw_cell = rnn_cell.GRUCell(self.size)
            fw_cell = rnn_cell.DropoutWrapper(
                fw_cell, output_keep_prob=self.keep_prob)
            self.encoder_fw_cell = rnn_cell.MultiRNNCell(
                [fw_cell] * self.num_layers, state_is_tuple=True)
            self.enc_fw_out, _ = rnn.dynamic_rnn(
                self.encoder_fw_cell, inp, self.src_len,
                dtype=tf.float32, time_major=True,
                initial_state=self.encoder_fw_cell.zero_state(self.batch_size,
                                                              dtype=tf.float32),
                scope='BiRNN_FW')

    def setup_encoder_backward(self):
        with vs.variable_scope("Encoder", reuse=None):
            inp = tf.nn.dropout(self.encoder_inputs, self.keep_prob)
            inp = tf.transpose(inp, [1, 0, 2])
            inp = tf.reverse_sequence(inp, self.src_len, seq_dim=1, batch_dim=0)
            inp = tf.transpose(inp, [1, 0, 2])
            bw_cell = rnn_cell.GRUCell(self.size)
            bw_cell = rnn_cell.DropoutWrapper(
                bw_cell, output_keep_prob=self.keep_prob)
            self.encoder_bw_cell = rnn_cell.MultiRNNCell(
                [bw_cell] * self.num_layers, state_is_tuple=True)
            self.enc_bw_out, _ = rnn.dynamic_rnn(
                self.encoder_bw_cell,
                inp,
                self.src_len,
                dtype=tf.float32, time_major=True,
                initial_state=self.encoder_fw_cell.zero_state(self.batch_size,
                                                              dtype=tf.float32),
                scope='BiRNN_BW')
            self.enc_bw_out = tf.transpose(self.enc_bw_out, [1, 0, 2])
            self.enc_bw_out = tf.reverse_sequence(self.enc_bw_out, self.src_len, seq_dim=1, batch_dim=0)
            self.enc_bw_out = tf.transpose(self.enc_bw_out, [1, 0, 2])

    def setup_decoder(self):
        with vs.variable_scope("Decoder"):
            inp =  tf.nn.dropout(self.decoder_inputs, self.keep_prob)
            if self.num_layers > 1:
                with vs.variable_scope("RNN"):
                    decoder_cell = rnn_cell.GRUCell(self.size)
                    decoder_cell = rnn_cell.DropoutWrapper(decoder_cell,
                                                           output_keep_prob=self.keep_prob)
                    self.decoder_cell = rnn_cell.MultiRNNCell(
                        [decoder_cell] * (self.num_layers - 1), state_is_tuple=True)
                    inp, _ = rnn.dynamic_rnn(self.decoder_cell, inp, self.src_len,
                                             dtype=tf.float32, time_major=True,
                                             initial_state=self.decoder_cell.zero_state(
                                                 self.batch_size, dtype=tf.float32))


            with vs.variable_scope("Attn"):
                self.attn_cell = GRUCellAttn(self.size, self.len_inp,
                                             self.encoder_output, self.src_mask)
                self.decoder_output, _ = rnn.dynamic_rnn(self.attn_cell, inp, self.src_len,
                                                         dtype=tf.float32, time_major=True,
                                                         initial_state=self.attn_cell.zero_state(
                                                             self.batch_size, dtype=tf.float32))

    def setup_loss(self, output, vocab_size, model):
        with vs.variable_scope("Logistic"):
            logits2d = rnn_cell._linear(tf.reshape(output,
                                                   [-1, self.size]),
                                        vocab_size, True, 1.0)
            self.outputs2d = tf.nn.log_softmax(logits2d)
            self.mask = self.src_mask
            labels = tf.reshape(self.tgt_toks, [-1])
            self.labels = labels

            if self.foward_only:
                labels = tf.one_hot(labels, depth=vocab_size)
            else:
                labels = label_smooth(labels, vocab_size)

            mask1d = tf.reshape(self.mask, [-1])
            losses1d = tf.nn.softmax_cross_entropy_with_logits(logits2d,
                                                                      labels) \
                       * tf.to_float(mask1d)
            self.losses2d = tf.reshape(losses1d, [self.len_inp, self.batch_size])
            self.losses = tf.reduce_sum(self.losses2d) / tf.to_float(self.batch_size)


    def build_model(self, vocab_size_char, model='lstm', model_sum=0, flag_bidirect=False, flag_word=False, vocab_size_word=0, loss="char"):
        self._add_place_holders(flag_word)
        with tf.variable_scope("Model", initializer=tf.uniform_unit_scaling_initializer(1.0)):
            self.setup_embeddings(vocab_size_char, model_sum=model_sum, model=model)
            if flag_bidirect:
                if not self.foward_only:
                    self.setup_encoder(flag_bidirect=True)
                else:
                    self.setup_encoder_forward()
                    self.setup_encoder_backward()
                    self.encoder_output = tf.concat(2, [self.enc_fw_out, self.enc_bw_out])
            else:
                self.setup_encoder(flag_bidirect=False)
            if model == 'lstm':
                output = self.encoder_output
            else:
                self.setup_decoder()
                output = self.decoder_output
            if flag_word:
                self.setup_loss(output, vocab_size_word, model)
            else:
                self.setup_loss(output, vocab_size_char, model)
            if self.foward_only:
                if model == 'seq2seq':
                    self.setup_beam("Logistic", vocab_size_char, model)
                else:
                    self.setup_lstm(vocab_size_char)
        if not self.foward_only:
            self.setup_train()

        self.saver = tf.train.Saver(tf.all_variables(), max_to_keep=0)

    def beam_step_seq2seq(self, time_step, beam_seqs, state_inputs):
        beam_size = tf.shape(beam_seqs)[1]
        self.attn_cell.mask = tf.cast(tf.tile(tf.reshape(tf.transpose(self.src_mask, [1, 0]),
                                                         [self.batch_size, 1, -1]),
                                              [1, beam_size, 1]),
                                      tf.bool)
        inp = tf.reshape(tf.slice(beam_seqs, [0, 0, time_step], [-1, -1, 1]),
                         [self.batch_size * beam_size])
        inputs = embedding_ops.embedding_lookup(self.L_dec, inp)
        with vs.variable_scope("Decoder", reuse=True):
            with vs.variable_scope("RNN", reuse=True):
                with vs.variable_scope("RNN", reuse=True):
                    rnn_out, rnn_outputs = self.decoder_cell(inputs, state_inputs[:self.num_layers-1])
            with vs.variable_scope("Attn", reuse=True):
                with vs.variable_scope("RNN", reuse=True):
                    out, attn_outputs = self.attn_cell.beam(rnn_out, state_inputs[-1], beam_size, self.batch_size)
        state_outputs = rnn_outputs + (attn_outputs, )
        return out, state_outputs

    def beam_loss(self, scope, output, vocab_size):
        with vs.variable_scope(scope, reuse=True):
            do2d = tf.reshape(output, [-1, self.size])
            logits2d = rnn_cell._linear(do2d, vocab_size, True, 1.0)
            logprobs3d = tf.reshape(tf.nn.log_softmax(logits2d), [self.batch_size, -1, vocab_size])
        return logprobs3d


    def setup_lstm(self, vocab_size):
        probs, index = tf.nn.top_k(self.outputs2d, k=self.num_pred)
        self.top_seqs = tf.reshape(index, [-1, self.batch_size, self.num_pred])

    def setup_beam(self, loss_scope, vocab_size, model='lstm'):
        time_0 = tf.constant(0)
        beam_seqs_0 = tf.ones([self.batch_size, 1, 1], dtype=tf.int32) * char2id['<sos>']
        beam_probs_0 = tf.zeros(shape=[self.batch_size, 1], dtype=tf.float32)
        # top_seqs_0 = tf.zeros([self.batch_size, self.num_pred, 1], dtype=tf.int32)
        top_seqs_1 = tf.zeros([self.batch_size, self.num_pred, 1], dtype=tf.int32)
        top_seqs_2 = tf.zeros([self.batch_size, self.num_pred, 1], dtype=tf.int32)
        top_seqs_3 = tf.zeros([self.batch_size, self.num_pred, 1], dtype=tf.int32)
        top_seqs_4 = tf.zeros([self.batch_size, self.num_pred, 1], dtype=tf.int32)
        perplex_0 = tf.zeros([self.batch_size, 1], dtype=tf.float32)
        state_0 = tf.zeros([self.batch_size, self.size])
        states_0 = (state_0,) * self.num_layers

        def decode_cond(time, beam_seqs, beam_probs, perplex, top_seqs1, top_seqs2, top_seqs3, top_seqs4, *states):
            return time < self.len_inp

        def decode_step(time, beam_seqs, beam_probs, perplex, top_seqs1, top_seqs2, top_seqs3, top_seqs4, *states):

            decoder_outputs, state_output = self.beam_step_seq2seq(time, beam_seqs, states)
            old_beam_size = tf.shape(beam_seqs)[1]
            logprobs3d = self.beam_loss(loss_scope, decoder_outputs, vocab_size)
            # batch_size * old_beam_size * vocab_size
            total_probs = logprobs3d + tf.reshape(beam_probs, [self.batch_size, -1, 1])
            # batch_size * (old_beam_size * vocab_size)
            flat_total_probs = tf.reshape(total_probs, [self.batch_size, -1])
            # new_beam_size
            beam_size = tf.minimum(tf.shape(flat_total_probs)[1], self.beam_size)
            # batch_size * new_beam_size, batch_size * new_beam_size
            next_beam_probs, top_indices = tf.nn.top_k(flat_total_probs, k=beam_size)
            # batch_size * new_beam_size
            next_bases = tf.floordiv(top_indices, vocab_size)
            # batch_size * new_beam_size
            next_mods = tf.mod(top_indices, vocab_size)
            # batch_size * new_beam_size * 1
            batch_index = tf.tile(tf.reshape(tf.range(self.batch_size),
                                             [-1, 1, 1]),
                                  [1, beam_size, 1])
            # batch_size * new_beam_size * 1
            base_index = tf.reshape(next_bases, [self.batch_size, -1, 1])
            # batch_size * new_beam_size * 2
            fetch_index = tf.concat(2, [batch_index, base_index])
            # batch_size * new_beam_size * len_output
            gather_seqs = tf.gather_nd(beam_seqs, fetch_index)
            # batch_size * new_beam_size * (len_output + 1)
            next_beam_seqs = tf.concat(2, [gather_seqs,
                                           tf.expand_dims(next_mods, -1)])
            # [(batch_size * new_beam_size) * num_units] * 3
            next_states = [tf.reshape(tf.gather_nd(tf.reshape(state,
                                                   [self.batch_size, -1, self.size]),
                                        fetch_index),
                                      [self.batch_size * beam_size, -1])
                           for state in state_output]


            # cur_seqs = tf.expand_dims(tf.slice(next_mods, [0, 0], [-1, self.num_pred]), -1)
            # next_top_seqs = tf.concat(2, [top_seqs, cur_seqs])

            # batch_size * vocab_size
            sum_probs = tf.reduce_sum(total_probs, reduction_indices=1)
            # batch_size * num_pred * 1
            cur_seqs1 = tf.expand_dims(tf.nn.top_k(sum_probs, self.num_pred)[1], -1)
            # batch_szie * num_pred * (len_output + 1)
            next_top_seqs1 = tf.concat(2, [top_seqs1, cur_seqs1])


            max_probs = tf.reduce_max(total_probs, reduction_indices=1)
            cur_seqs2 = tf.expand_dims(tf.nn.top_k(max_probs, self.num_pred)[1], -1)
            next_top_seqs2 = tf.concat(2, [top_seqs2, cur_seqs2])


            sum_probs_1 = tf.reduce_sum(logprobs3d, reduction_indices=1)
            cur_seqs3 = tf.expand_dims(tf.nn.top_k(sum_probs_1, self.num_pred)[1], -1)
            next_top_seqs3= tf.concat(2, [top_seqs3, cur_seqs3])


            max_probs_1 = tf.reduce_max(logprobs3d, reduction_indices=1)
            cur_seqs4 = tf.expand_dims(tf.nn.top_k(max_probs_1, self.num_pred)[1], -1)
            next_top_seqs4 = tf.concat(2, [top_seqs4, cur_seqs4])

            cur_tok = tf.tile(tf.reshape(tf.slice(self.tgt_toks,
                                                  [time, 0], [1, -1]),
                                         [-1, 1, 1]),
                              [1, old_beam_size, 1])
            # batch_size * old_beam_size * 1
            batch_index = tf.tile(tf.reshape(tf.range(self.batch_size), [-1, 1, 1]),
                                  [1, old_beam_size, 1])
            beam_index = tf.tile(tf.reshape(tf.range(old_beam_size), [1, -1, 1]), [self.batch_size, 1, 1])
            toks_index = tf.concat(2, [batch_index, beam_index, cur_tok])
            perplex_prob = tf.reduce_max(tf.gather_nd(logprobs3d, toks_index),
                                         reduction_indices=-1)
            next_perplex = tf.concat(1, [perplex, tf.reshape(perplex_prob, [-1, 1])])

            return [time + 1, next_beam_seqs, next_beam_probs, next_perplex, next_top_seqs1, next_top_seqs2, next_top_seqs3, next_top_seqs4] + next_states

        var_shape = []
        var_shape.append((time_0, time_0.get_shape()))
        var_shape.append((beam_seqs_0, tf.TensorShape([None, None, None])))
        var_shape.append((beam_probs_0, tf.TensorShape([None, None])))
        var_shape.append((perplex_0, tf.TensorShape([None, None])))
        # var_shape.append((top_seqs_0, tf.TensorShape([None, None, None])))
        var_shape.append((top_seqs_1, tf.TensorShape([None, None, None])))
        var_shape.append((top_seqs_2, tf.TensorShape([None, None, None])))
        var_shape.append((top_seqs_3, tf.TensorShape([None, None, None])))
        var_shape.append((top_seqs_4, tf.TensorShape([None, None, None])))
        # var_shape.append((top_probs_0, tf.TensorShape([None, None])))
        var_shape.extend([(state_0, tf.TensorShape([None, self.size])) for state_0 in states_0])
        loop_vars, loop_var_shapes = zip(*var_shape)
        self.loop_vars = loop_vars
        self.loop_var_shapes = loop_var_shapes
        ret_vars = tf.while_loop(cond=decode_cond, body=decode_step, loop_vars=loop_vars, back_prop=False)
        self.vars = ret_vars
        self.beam_output = ret_vars[1]
        self.beam_scores = ret_vars[2]
        self.perplex = ret_vars[3]
        self.top_seqs_1 = ret_vars[4]
        self.top_seqs_2 = ret_vars[5]
        self.top_seqs_3 = ret_vars[6]
        self.top_seqs_4 = ret_vars[7]

    def decode_beam(self, session, src_toks, tgt_toks, src_mask, beam_size=128):
        input_feed = {}
        input_feed[self.src_toks] = src_toks
        input_feed[self.src_mask] = src_mask
        input_feed[self.tgt_toks] = tgt_toks
        input_feed[self.keep_prob] = 1.
        input_feed[self.beam_size] = beam_size
        output_feed = [self.beam_output, self.beam_scores, self.perplex,  self.top_seqs_1, self.top_seqs_2, self.top_seqs_3, self.top_seqs_4]
        outputs = session.run(output_feed, input_feed)
        return outputs[0], outputs[1], outputs[2], [outputs[3], outputs[4], outputs[5], outputs[6]]

    def encode_fw(self, session, src_toks, src_mask):
        input_feed = {}
        input_feed[self.src_toks] = src_toks
        input_feed[self.src_mask] = src_mask
        input_feed[self.keep_prob] = 1.
        output_feed = [self.enc_fw_out]
        outputs = session.run(output_feed, input_feed)
        return outputs[0]

    def encode_bw(self, session, src_toks, src_mask):
        input_feed = {}
        input_feed[self.src_toks] = src_toks
        input_feed[self.src_mask] = src_mask
        input_feed[self.keep_prob] = 1.
        output_feed = [self.enc_bw_out]
        outputs = session.run(output_feed, input_feed)
        return outputs[0]

    def decode_lstm(self, session, src_toks, tgt_toks, src_mask):
        input_feed = {}
        input_feed[self.src_toks] = src_toks
        input_feed[self.src_mask] = src_mask
        input_feed[self.tgt_toks] = tgt_toks
        input_feed[self.keep_prob] = 1.
        output_feed = [self.losses2d,self.top_seqs]
        outputs = session.run(output_feed, input_feed)
        return outputs[0], outputs[1]

    def encode(self, session, src_toks, src_mask):
        input_feed = {}
        input_feed[self.src_toks] = src_toks
        input_feed[self.src_mask] = src_mask
        input_feed[self.keep_prob] = 1.
        output_feed = [self.encoder_output]
        outputs = session.run(output_feed, input_feed)
        return outputs[0]

    def decode(self, session, encoder_output, src_mask, tgt_toks, beam_size):
        input_feed = {}
        input_feed[self.encoder_output] = encoder_output
        input_feed[self.src_mask] = src_mask
        input_feed[self.tgt_toks] = tgt_toks
        input_feed[self.keep_prob] = 1.
        input_feed[self.beam_size] = beam_size
        input_feed[self.batch_size] = encoder_output.shape[1]
        input_feed[self.len_inp] = encoder_output.shape[0]
        output_feed = [self.beam_output, self.beam_scores, self.perplex,  self.top_seqs_1, self.top_seqs_2, self.top_seqs_3, self.top_seqs_4]
        outputs = session.run(output_feed, input_feed)
        return outputs[0], outputs[1], outputs[2], [outputs[3], outputs[4], outputs[5], outputs[6]]

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
