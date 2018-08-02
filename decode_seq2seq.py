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
from data_simulate_eeg_varlen import create_vector

import logging
logging.basicConfig(level=logging.INFO)


id2word, word2id = None, None
model = None
line_x = None
len_x = None
data = None
index_x = None


def create_model(session):
    global model, word2id
    if FLAGS.flag_word:
        vocab_size_word = len(word2id)
    vocab_size_char = len(char2id)
    model = model_concat.Model(FLAGS.size, FLAGS.num_cand, FLAGS.num_layers,
                        FLAGS.max_gradient_norm, FLAGS.learning_rate,
                        FLAGS.learning_rate_decay_factor, forward_only=True,
                        optimizer=FLAGS.optimizer, num_pred=FLAGS.num_cand)
    model.build_model(vocab_size_char, model=FLAGS.model, flag_bidirect=FLAGS.flag_bidirect)
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


def read_data():
    global line_x, data, len_x, index_x
    logging.info("Get NLC data in %s" % FLAGS.data_dir)
    x_dev = pjoin(FLAGS.data_dir, FLAGS.dev + '.ids')
    filename = '%s.%s.%d' % (FLAGS.dev, FLAGS.random, FLAGS.num_top)
    if FLAGS.random == 'random':
        filename = pjoin(FLAGS.data_dir, '%s_%.1f_%.1f_%.1f' % (filename,
                                                              FLAGS.prob_high,
                                                              FLAGS.prior,
                                                              FLAGS.prob_in))
    fx = open(x_dev)
    fc = open(filename + '.cand')
    fp = open(filename + '.prob')
    line_id = 0
    data, list_tokens, len_line = [], [], []
    linex, linec, linep = fx.readline(), fc.readline(), fp.readline()
    voc_size = len(char2id)
    pad_head = np.zeros(voc_size)
    pad_head[char2id['<sos>']] = 1.
    while line_id < FLAGS.end and len(linex.strip()) > 0:
        if line_id >= FLAGS.start:
            cur_line = map(int, linex.strip('\n').split(' '))[:FLAGS.max_seq_len]
            list_tokens.append(cur_line)
            len_line.append(len(cur_line))
            newc = linec.strip().split('\t')[:FLAGS.max_seq_len]
            cands = map(lambda ele: [int(s) for s in ele.split()[:FLAGS.num_wit]], newc)
            newp = linec.strip().split('\t')[:FLAGS.max_seq_len]
            probs = map(lambda ele: [int(s) for s in ele.split()[:FLAGS.num_wit]], newp)
            vecs = [pad_head] + map(lambda cand, prob: create_vector(cand, prob), cands[:-1], probs[:-1])
            data.append(vecs)
        line_id += 1
        linex, linec, linep = fx.readline(), fc.readline(), fp.readline()
    index_x = np.argsort(len_line)
    line_x = map(lambda ele: list_tokens[ele], index_x)
    len_x = map(lambda ele: len_line[ele], index_x)
    data = map(lambda ele: data[ele], index_x)


def get_mask(len_input, max_len):
    mask = map(lambda ele: [1] * min(ele, max_len) + [0] * (max_len - min(ele, max_len)), len_input)
    return np.asarray(mask).T


def decode_batch(sess, batch_size, s, e):
    global data, model, line_x, len_x, index_x
    cur_chunk_len = len_x[s:e]
    cur_max_len = max(cur_chunk_len)
    cur_index = index_x[FLAGS.start + s: FLAGS.start + e]
    batch_size = min(batch_size, e - s)
    src_probs = data[s:e]
    # padding
    src_probs = padded(src_probs, len(id2char), 0.0)
    src_probs = np.transpose(np.array(src_probs), (1, 0, 2))
    tgt_toks = padded(line_x[s:e], 1)
    tgt_toks = np.asarray(tgt_toks).T
    # results
    cur_recall = np.zeros((2, batch_size, cur_max_len))
    cur_mrr = np.zeros((2, batch_size, cur_max_len))
    cur_perplex = np.zeros((batch_size, cur_max_len))

    if FLAGS.model == "lstm":
        cur_mask = (np.sum(src_probs, -1) > 0).astype(np.int32)
        perplex, top_seqs = model.decode_lstm(sess, src_probs, tgt_toks, cur_mask)
        top_seqs = np.transpose(top_seqs, [1, 2, 0])
        truth = np.transpose(tgt_toks)
        cur_mrr[0, :, :] = batch_mrr(top_seqs, truth, 10)
        cur_recall[0, :, :] = batch_recall_at_k(top_seqs, truth, 10)
        cur_perplex = np.transpose(perplex)[:, :-1]
    elif FLAGS.model == "seq2seq":
        if FLAGS.flag_bidirect:
            src_mask = get_mask(cur_chunk_len, cur_max_len)
            encoder_out_fw = model.encode_fw(sess, src_probs, src_mask)
        else:
            src_mask = get_mask(cur_chunk_len, cur_max_len)
            encoder_out = model.encode(sess, src_probs, src_mask)
        for k in range(cur_max_len):
            print('\t' + str(k))
            cur_mask = get_mask(cur_chunk_len, k + 1)
            if FLAGS.flag_bidirect:
                encoder_out_bw = model.encode_bw(sess, src_probs[:k+1,:,:], cur_mask)
                encoder_out = np.concatenate((encoder_out_fw[:k+1,:,:], encoder_out_bw[:k+1,:,:]), 2)
            outputs, prediction, perplex, top_seqs = model.decode(sess, encoder_out, cur_mask,
                                                                      tgt_toks[:k + 1, :], FLAGS.beam_size)
            cur_truth = np.transpose(tgt_toks[k, :])
            for pred_id in range(2):
                cur_pred = top_seqs[pred_id][:, :, k + 1]
                cur_mrr[pred_id, :, k] = batch_mrr(cur_pred, cur_truth, FLAGS.num_cand)
                cur_recall[pred_id, :, k] = batch_recall_at_k(cur_pred, cur_truth, FLAGS.num_cand)
            cur_perplex[:, k] = perplex[:, k + 1]
    str_mrr = ['\n'.join(map(lambda a, len: '\t'.join(map(str, a[:len])), cur_mrr[pred_id], cur_chunk_len)) + '\n' for pred_id in range(2)]
    str_recall = ['\n'.join(map(lambda a, len: '\t'.join(map(str, a[:len])), cur_recall[pred_id], cur_chunk_len)) + '\n' for pred_id in range(2)]
    str_perplex = '\n'.join(map(lambda a, len: '\t'.join(map(str, a[:len])), cur_perplex, cur_chunk_len)) + '\n'
    str_index = '\n'.join(map(str, cur_index)) + '\n'
    return str_mrr, str_recall, str_perplex, str_index


def decode():
    global word2id, id2word, data, model, line_x, len_x
    #if not os.path.exists(pjoin(FLAGS.data_dir, FLAGS.out_dir)):
    #    os.makedirs(pjoin(FLAGS.data_dir, FLAGS.out_dir))
    read_data()
    num_line = len(line_x)
    num_chunk = int(np.ceil(num_line * 1. / FLAGS.batch_size))
    f_mrr = []
    f_recall = []
    if FLAGS.model == 'seq2seq':
        num_file = 2
    else:
        num_file = 1
    for pred_id in range(num_file):
        f_mrr.append(open(pjoin(FLAGS.data_dir, FLAGS.out_dir, 'mrr%d.%d_%d' % (pred_id, FLAGS.start, FLAGS.end)), 'w'))
        f_recall.append(open(pjoin(FLAGS.data_dir, FLAGS.out_dir, 'recall%d.%d_%d') % (pred_id, FLAGS.start, FLAGS.end), 'w'))
    f_index = open(pjoin(FLAGS.data_dir, FLAGS.out_dir, 'index.%d_%d' % (FLAGS.start, FLAGS.end)), 'w')
    f_perplex = open(pjoin(FLAGS.data_dir, FLAGS.out_dir, 'perplex.%d_%d') % (FLAGS.start, FLAGS.end), 'w')

    with tf.Session() as sess:
        create_model(sess)
        for i in range(num_chunk):
            start = i * FLAGS.batch_size
            end = min(num_line, (i + 1) * FLAGS.batch_size)
            print('batch', i)
            mrr, recall, perplex, index = decode_batch(sess, FLAGS.batch_size, start, end)
            for pred_id in range(num_file):
                f_mrr[pred_id].write(mrr[pred_id])
                f_recall[pred_id].write(recall[pred_id])
            f_perplex.write(perplex)
            f_index.write(index)
        for pred_id in range(num_file):
            f_mrr[pred_id].close()
            f_recall[pred_id].close()
        f_perplex.close()
        f_index.close()


def main(_):
    decode()

if __name__ == "__main__":
    tf.app.run()
