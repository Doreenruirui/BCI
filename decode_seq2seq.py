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

import re
from os.path import join as pjoin
import os
import numpy as np
from six.moves import xrange
import tensorflow as tf
import model_concat as model_concat
from flag import FLAGS
from data_generate import id2char, char2id, load_vocabulary, padded
from evaluate import batch_mrr, batch_recall_at_k
from data_generate import pair_iter

import logging
logging.basicConfig(level=logging.INFO)


model = None
line_x = None
len_x = None
data = None
index_x = None


def create_model(session):
    global model, word2id
    vocab_size_char = len(char2id)
    model = model_concat.Model(FLAGS.size, FLAGS.num_cand, FLAGS.num_layers,
                        FLAGS.max_gradient_norm, FLAGS.learning_rate,
                        FLAGS.learning_rate_decay_factor, forward_only=True,
                        optimizer=FLAGS.optimizer, num_pred=FLAGS.num_cand)
    model.build_model(vocab_size_char, model=FLAGS.model, flag_bidirect=FLAGS.flag_bidirect, model_sum=FLAGS.flag_sum)
    ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
    if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
        logging.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
        num_epoch = int(ckpt.model_checkpoint_path.split('-')[1])
        print (num_epoch)
    else:
        raise ValueError("No trained Model Found")


def decode_string(list_tokens, revocab):
    if FLAGS.flag_word:
        return map(lambda tokens: ' '.join([id2word[ele] if revocab[ele] != ' ' else '<space>' for ele in tokens[1:]]), list_tokens)
    else:
        return map(lambda tokens: ''.join([id2char[tok] for tok in tokens[1:]]), list_tokens)


def get_mask(len_input, max_len):
    mask = map(lambda ele: [1] * min(ele, max_len) + [0] * (max_len - min(ele, max_len)), len_input)
    return np.asarray(mask).T


def decode_batch(sess, source_tokens, source_mask, target_tokens):
    cur_max_len = max(np.sum(source_mask, axis=0))
    cur_chunk_len = np.sum(source_mask, axis=0)
    batch_size = source_tokens.shape[0]

    cur_recall = np.zeros((2, batch_size, cur_max_len))
    cur_mrr = np.zeros((2, batch_size, cur_max_len))
    cur_perplex = np.zeros((batch_size, cur_max_len))

    if FLAGS.model == "lstm":
        if FLAGS.flag_bidirect:
            #perplex, top_seqs = model.decode_lstm(sess, src_probs, tgt_toks, cur_mask)
            #top_seqs = np.transpose(top_seqs, [1, 2, 0])
            #truth = np.transpose(tgt_toks)
            #cur_mrr[0, :, :] = batch_mrr(top_seqs, truth, 10)
            #cur_recall[0, :, :] = batch_recall_at_k(top_seqs, truth, 10)
            #cur_perplex = np.transpose(perplex)
            encoder_out_fw = model.encode_fw(sess, source_tokens, source_mask)
            for k in range(cur_max_len):
                print('\t' + str(k))
                cur_mask = source_mask[:k+1,:]
                encoder_out_bw = model.encode_bw(sess, source_tokens[:k + 1, :, :], cur_mask)
                perplex, top_seqs = model.decode_bilstm(sess, encoder_out_fw[:k+1,:,:],
                                                        encoder_out_bw[:k + 1,:,:],
                                                        target_tokens[:k + 1, :], cur_mask)
                cur_truth = np.transpose(target_tokens[k, :])
                cur_pred = top_seqs[k, :, :]
                cur_mrr[0, :, k] = batch_mrr(cur_pred, cur_truth, FLAGS.num_cand)
                cur_recall[0, :, k] = batch_recall_at_k(cur_pred, cur_truth, FLAGS.num_cand)
                cur_perplex[:, k] = perplex[k, :]
        else:
            perplex, top_seqs = model.decode_lstm(sess, source_tokens, target_tokens, source_mask)
            top_seqs = np.transpose(top_seqs, [1, 2, 0])
            truth = np.transpose(target_tokens)
            cur_mrr[0, :, :] = batch_mrr(top_seqs, truth, 10)
            cur_recall[0, :, :] = batch_recall_at_k(top_seqs, truth, 10)
            cur_perplex = np.transpose(perplex)[:, :-1]
    elif FLAGS.model == "seq2seq":
        if FLAGS.flag_bidirect:
            encoder_out_fw = model.encode_fw(sess, source_tokens, source_mask)
        else:
            encoder_out = model.encode(sess, source_tokens, source_mask)
        for k in range(cur_max_len):
            print('\t' + str(k))
            cur_mask = source_mask[:k+1,:]
            if FLAGS.flag_bidirect:
                encoder_out_bw = model.encode_bw(sess, source_tokens[:k+1,:,:], cur_mask)
                encoder_out = np.concatenate((encoder_out_fw[:k+1,:,:], encoder_out_bw[:k+1,:,:]), 2)
            outputs, prediction, perplex, top_seqs = model.decode(
                sess, encoder_out, cur_mask, target_tokens[:k + 1, :], FLAGS.beam_size)
            cur_truth = np.transpose(target_tokens[k, :])
            for pred_id in range(2):
                cur_pred = top_seqs[pred_id][:, :, k + 1]
                cur_mrr[pred_id, :, k] = batch_mrr(cur_pred, cur_truth, FLAGS.num_cand)
                cur_recall[pred_id, :, k] = batch_recall_at_k(cur_pred, cur_truth, FLAGS.num_cand)
            cur_perplex[:, k] = perplex[:, k + 1]
    str_mrr = ['\n'.join(map(lambda a, len: '\t'.join(map(str, a[:len])), cur_mrr[pred_id], cur_chunk_len)) + '\n' for pred_id in range(2)]
    str_recall = ['\n'.join(map(lambda a, len: '\t'.join(map(str, a[:len])), cur_recall[pred_id], cur_chunk_len)) + '\n' for pred_id in range(2)]
    str_perplex = '\n'.join(map(lambda a, len: '\t'.join(map(str, a[:len])), cur_perplex, cur_chunk_len)) + '\n'
    return str_mrr, str_recall, str_perplex


def decode():
    if not os.path.exists(pjoin(FLAGS.data_dir, FLAGS.out_dir)):
       os.makedirs(pjoin(FLAGS.data_dir, FLAGS.out_dir))
    f_mrr = []
    f_recall = []
    if FLAGS.model == 'seq2seq':
        num_file = 2
    else:
        num_file = 1
    for pred_id in range(num_file):
        f_mrr.append(open(pjoin(FLAGS.data_dir, FLAGS.out_dir, 'mrr%d.%d_%d' % (pred_id, FLAGS.start, FLAGS.end)), 'w'))
        f_recall.append(open(pjoin(FLAGS.data_dir, FLAGS.out_dir, 'recall%d.%d_%d') % (pred_id, FLAGS.start, FLAGS.end), 'w'))
    f_perplex = open(pjoin(FLAGS.data_dir, FLAGS.out_dir, 'perplex.%d_%d') % (FLAGS.start, FLAGS.end), 'w')

    with tf.Session() as sess:
        create_model(sess)
        for source_tokens, source_mask, target_tokens in pair_iter(FLAGS.data_dir,
                                                                   FLAGS.dev, FLAGS.num_wit,
                                                                   cur_len=-2,
                                                                   num_top=FLAGS.num_top,
                                                                   max_seq_len=FLAGS.max_seq_len,
                                                                   data_random=FLAGS.random,
                                                                   batch_size=FLAGS.batch_size,
                                                                   prior=FLAGS.prior,
                                                                   prob_high=FLAGS.prob_high,
                                                                   prob_in=FLAGS.prob_in,
                                                                   flag_generate=FLAGS.flag_generate,
                                                                   prob_back=FLAGS.prob_back):

            mrr, recall, perplex = decode_batch(sess, source_tokens, source_mask, target_tokens)
            for pred_id in range(num_file):
                f_mrr[pred_id].write(mrr[pred_id])
                f_recall[pred_id].write(recall[pred_id])
            f_perplex.write(perplex)
        for pred_id in range(num_file):
            f_mrr[pred_id].close()
            f_recall[pred_id].close()
        f_perplex.close()


def main(_):
    decode()

if __name__ == "__main__":
    tf.app.run()
