from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import variable_scope as vs
from model_rnn import GRUCellAttn, linear
from module import label_smooth


def get_optimizer(opt):
    if opt == "adam":
        optfn = tf.train.AdamOptimizer
    elif opt == "sgd":
        optfn = tf.train.GradientDescentOptimizer
    else:
        assert (False)
    return optfn


class Model(object):
    def __init__(self, size, num_layers, max_gradient_norm,
                 learning_rate, learning_rate_decay, num_wit,
                 forward_only=False, optimizer="adam", num_pred=5):
        self.hidden_size = size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.learning_decay = learning_rate_decay
        self.max_grad_norm = max_gradient_norm
        self.foward_only = forward_only
        self.optimizer = optimizer
        self.num_pred = num_pred
        self.num_wit = num_wit

    def _add_place_holders(self):
        self.keep_prob = tf.placeholder(tf.float32)
        self.src_toks = tf.placeholder(tf.int32, shape=[None, None, None])
        self.src_probs = tf.placeholder(tf.float32, shape=[None, None, None])
        self.tgt_toks = tf.placeholder(tf.int32, shape=[None, None])
        self.src_mask = tf.placeholder(tf.int32, shape=[None, None])
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

    def setup_embeddings(self, vocab_size, model_sum=0):
        with vs.variable_scope("embeddings"):
            zeros = tf.zeros([1, self.hidden_size])
            enc = tf.get_variable("L_enc", [vocab_size - 1, self.hidden_size])
            self.L_enc = tf.concat([zeros, enc], axis=0)
            if model_sum == 1:
                self.encoder_inputs = embedding_ops.embedding_lookup(
                    self.L_enc, self.src_toks)                              # len_inp * batch_size * num_wit * embedding_size
                self.encoder_inputs = self.encoder_inputs * \
                                      tf.reshape(self.src_probs,
                                                 [self.len_inp, self.batch_size,
                                                  self.num_wit, 1])
                self.encoder_inputs = tf.reduce_sum(self.encoder_inputs,
                                                    axis=2)
            else:
                self.encoder_inputs = tf.reshape(self.src_probs,
                                                 [self.len_inp, self.batch_size,
                                                  vocab_size, -1]) * \
                                      tf.reshape(self.L_enc,
                                                 [1, 1, vocab_size, self.hidden_size])

                self.encoder_inputs = tf.reshape(self.encoder_inputs,
                                                 [self.len_inp, -1,
                                                  vocab_size * self.hidden_size])

    def lstm_cell(self):
        lstm = rnn_cell.DropoutWrapper(tf.contrib.rnn.GRUCell(self.hidden_size),
                                       output_keep_prob=self.keep_prob)
        return lstm

    def setup_encoder(self):
        with vs.variable_scope("Encoder"):
            inp = tf.nn.dropout(self.encoder_inputs, self.keep_prob)
            self.encoder_fw_cell = rnn_cell.MultiRNNCell(
                [self.lstm_cell() for _ in range(self.num_layers)],
                state_is_tuple=True)
            out, _ = rnn.dynamic_rnn(self.encoder_fw_cell, inp, self.src_len,
                                         dtype=tf.float32, time_major=True)
            self.encoder_output = out

    def setup_loss(self, output, vocab_size, size):
        with vs.variable_scope("Logistic"):
            logits2d = linear(tf.reshape(output, [-1, size]),
                               vocab_size, True, 1.0)
            self.outputs2d = tf.nn.log_softmax(logits2d)
            labels = tf.reshape(self.tgt_toks, [-1])
            if self.foward_only:
                labels = tf.one_hot(labels, depth=vocab_size)
            else:
                labels = label_smooth(labels, vocab_size)
            mask1d = tf.reshape(self.src_mask, [-1])
            losses1d = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits2d,
                                                               labels=labels) \
                       * tf.to_float(mask1d)
            self.losses2d = tf.reshape(losses1d, [self.len_inp,
                                                  self.batch_size])
            self.losses = tf.reduce_sum(self.losses2d) / \
                          tf.to_float(self.batch_size)

    def build_lstm(self, vocab_size_char, model_sum=0):
        self._add_place_holders()
        with tf.variable_scope("Model", initializer=tf.uniform_unit_scaling_initializer(1.0)):
            self.setup_embeddings(vocab_size_char, model_sum=model_sum)
            self.setup_encoder()
            output = self.encoder_output
            self.setup_loss(output, vocab_size_char, self.hidden_size)
            if self.foward_only:
                self.setup_lstm()
        if not self.foward_only:
            self.setup_train()
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=0)

    def setup_lstm(self):
        probs, index = tf.nn.top_k(self.outputs2d, k=self.num_pred)
        self.top_seqs = tf.reshape(index, [-1, self.batch_size, self.num_pred])

    def decode_lstm(self, session, src_toks, src_probs, tgt_toks, src_mask, flag_sum):
        input_feed = {}
        if flag_sum:
            input_feed[self.src_toks] = src_toks
        input_feed[self.src_probs] = src_probs
        input_feed[self.src_mask] = src_mask
        input_feed[self.tgt_toks] = tgt_toks
        input_feed[self.keep_prob] = 1.
        output_feed = [self.losses2d, self.top_seqs]
        outputs = session.run(output_feed, input_feed)
        return outputs[0], outputs[1]

    def train(self, session, src_toks, src_probs, src_mask, tgt_toks, keep_prob, flag_sum):
        input_feed = {}
        if flag_sum:
            input_feed[self.src_toks] = src_toks
        input_feed[self.src_probs] = src_probs
        input_feed[self.tgt_toks] = tgt_toks
        input_feed[self.src_mask] = src_mask
        input_feed[self.keep_prob] = keep_prob

        output_feed = [self.updates, self.gradient_norm, self.losses, self.param_norm]

        outputs = session.run(output_feed, input_feed)

        return outputs[1], outputs[2], outputs[3]

    def test(self, session, src_toks, src_probs, src_mask, tgt_toks, flag_sum):
        input_feed = {}
        if flag_sum:
            input_feed[self.src_toks] = src_toks
        input_feed[self.src_probs] = src_probs
        input_feed[self.tgt_toks] = tgt_toks
        input_feed[self.src_mask] = src_mask
        input_feed[self.keep_prob] = 1.
        output_feed = [self.losses]
        outputs = session.run(output_feed, input_feed)
        return outputs[0]
