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
                #out = tf.concat(2, [out[0], out[1]])
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
            self.outputs2d = tf.nn.log_softmax(logits2d)
            # self.outputs = tf.argmax(tf.reshape(outputs2d,
            #                           [-1, self.batch_size, vocab_size]), 2)
            labels = tf.reshape(tf.pad(tf.slice(self.tgt_toks, [1, 0], [-1, -1]),
                                       [[0,1],[0,0]]),
                                [-1])
            self.labels = labels
            if self.foward_only:
                labels = tf.one_hot(labels, depth=vocab_size)
            else:
                labels = label_smooth(labels, vocab_size)
            mask1d = tf.reshape(self.src_mask, [-1])
            losses1d = tf.nn.softmax_cross_entropy_with_logits(logits2d,
                                                                      labels) \
                       * tf.to_float(mask1d)
            self.losses2d = tf.reshape(losses1d, [self.len_inp, self.batch_size])
            self.losses = tf.reduce_sum(self.losses2d) / tf.to_float(self.batch_size)

    def build_model(self, vocab_size_char, model='lstm', flag_bidirect=False, flag_word=False, vocab_size_word=0, loss="char"):
        self._add_place_holders(flag_word)
        with tf.variable_scope("Model", initializer=tf.uniform_unit_scaling_initializer(1.0)):
            self.setup_embeddings(vocab_size_char)
            if flag_bidirect:
                self.setup_encoder(flag_bidirect=True)
            else:
                self.setup_encoder(flag_bidirect=False)
            if model == 'lstm':
                output = self.encoder_output
            else:
                self.setup_decoder()
                output = self.decoder_output
            if flag_word:
                self.setup_loss(output, vocab_size_word)
            else:
                self.setup_loss(output, vocab_size_char)
            if self.foward_only:
                if model == 'seq2seq':
                    self.setup_beam("Logistic", vocab_size_char, model)
                else:
                    self.setup_lstm(vocab_size_char)
        if not self.foward_only:
            self.setup_train()

        self.saver = tf.train.Saver(tf.all_variables(), max_to_keep=0)


    def beam_loss(self, scope, output, vocab_size):
        with vs.variable_scope(scope, reuse=True):
            do2d = tf.reshape(output, [-1, self.size])
            logits2d = rnn_cell._linear(do2d, vocab_size, True, 1.0)
            logprobs3d = tf.reshape(tf.nn.log_softmax(logits2d), [self.batch_size, -1, vocab_size])
        return logprobs3d


    def setup_lstm(self, vocab_size):
        probs, index = tf.nn.top_k(self.outputs2d, k=self.num_pred)
        self.top_seqs = tf.reshape(index, [-1, self.batch_size, self.num_pred])

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
