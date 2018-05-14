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
from os.path import exists
import os
import numpy as np
from six.moves import xrange
import tensorflow as tf

import model_concat

from flag import FLAGS
from data_generation import id2char, char2id, load_vocabulary
from evaluate import batch_mrr, batch_recall_at_k

import logging
logging.basicConfig(level=logging.INFO)


id2word, word2id = None, None
data = None
model = None
line_x = None
len_x = None


def create_model(session):
    global model, word2id
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

def get_mask(len_input, max_len):
    num_line = len(len_input)
    mask = map(lambda ele: [1] * min(ele, max_len) + [0] * (max_len - min(ele, max_len)), len_input)
    return np.asarray(mask).T


def read_data():
    global data, line_x, len_x, word2id, id2word
    logging.info("Get NLC data in %s" % FLAGS.data_dir)
    x_dev = pjoin(FLAGS.data_dir, FLAGS.dev + '.ids')
    vocab_size_char = len(id2char)
    word2id, id2word = load_vocabulary(FLAGS.data_dir)
    if FLAGS.flag_word:
        rev_vocab = id2word
    else:
        rev_vocab = id2char
    lines = []
    len_line = []
    line_id = 0
    for line in file(x_dev):
        if line_id == FLAGS.end:
            break
        if line_id >= FLAGS.start:
            cur_line = map(int, line.strip('\n').split(' '))
            lines.append(cur_line)
            len_line.append(len(cur_line))
        line_id += 1
    sorted_index = np.argsort(len_line)
    line_x = map(lambda ele: lines[ele], sorted_index)
    len_x = map(lambda ele: len_line[ele], sorted_index)
    data = np.load(pjoin(FLAGS.data_dir, FLAGS.dev + '.prob.npy'))
    data = data[:, FLAGS.start:FLAGS.end, :]
    data = data[:, sorted_index, :]

def decode_batch(sess, batch_size, s, e):
    global data, model, line_x, len_x
    cur_chunk_len = len_x[s:e]
    cur_max_len = max(cur_chunk_len)
    src_probs = data[:cur_max_len, s:e, :]
    tgt_toks = map(lambda line: [char2id['<sos>']] + line + [char2id['<pad>']] * (cur_max_len - len(line)), line_x[s:e])
    tgt_toks = np.asarray(tgt_toks).T
    batch_size = min(batch_size, e - s)
    cur_recall = np.zeros((2, batch_size, cur_max_len))
    cur_mrr = np.zeros((2, batch_size, cur_max_len))
    cur_perplex = np.zeros((batch_size, cur_max_len))
    if FLAGS.model == "lstm":
        cur_mask = np.ones_like(tgt_toks) * (tgt_toks > 0)
        outputs, prediction, perplex, top_seqs = model.decode_beam(sess, src_probs, tgt_toks, cur_mask, FLAGS.beam_size)
        truth = tgt_toks[1:, :]
        for pred_id in range(2):
            cur_mrr[pred_id,:,:] = batch_mrr(top_seqs[pred_id][:, :, 1:], truth, 10)
            cur_recall[pred_id, :, :] = batch_recall_at_k(top_seqs[pred_id][:, :, 1:], truth, 10)
        cur_perplex = perplex[:, 1:]
    elif FLAGS.model == "seq2seq":
        if not FLAGS.flag_bidirect:
            src_mask = get_mask(cur_chunk_len, cur_max_len + 1)
            encoder_out = model.encode(sess, src_probs, src_mask)
        for k in range(cur_max_len):
            cur_mask = get_mask(cur_chunk_len, k + 1)
            if FLAGS.flag_bidirect:
                outputs, prediction, perplex, top_seqs = model.decode_beam(sess, src_probs[:k+1,:,:], tgt_toks[:k+2,:], cur_mask, FLAGS.beam_size)
            else:
                outputs, prediction, perplex, top_seqs = model.decode(sess, encoder_out[:k + 1, :, :], cur_mask,
                                                                      tgt_toks[:k + 2, :], FLAGS.beam_size)
            cur_truth = tgt_toks[k + 1, :]
            for pred_id in range(2):
                cur_pred = top_seqs[pred_id][:, :, k+1]
                cur_mrr[pred_id, :, k] = batch_mrr(cur_pred, cur_truth, FLAGS.num_cand)
                cur_recall[pred_id, :, k] = batch_recall_at_k(cur_pred, cur_truth, FLAGS.num_cand)
            cur_perplex[:, k] = perplex[:, k + 1]
    str_mrr = ['\n'.join(map(lambda a, len: '\t'.join(map(str, a[:len])), cur_mrr[pred_id], cur_chunk_len)) + '\n' for pred_id in range(2)]
    str_recall = ['\n'.join(map(lambda a, len: '\t'.join(map(str, a[:len])), cur_recall[pred_id], cur_chunk_len)) + '\n' for pred_id in range(2)]
    str_perplex = '\n'.join(map(lambda a, len: '\t'.join(map(str, a[:len])), cur_perplex, cur_chunk_len)) + '\n'
    return str_mrr, str_recall, str_perplex


def decode():
    global word2id, id2word, data, model, line_x, len_x
    if not exists(pjoin(FLAGS.data_dir, FLAGS.out_dir)):
        os.makedirs(pjoin(FLAGS.data_dir, FLAGS.out_dir))
    read_data()
    num_line = len(line_x)
    num_chunk = int(np.ceil(num_line * 1. / FLAGS.batch_size))
    f_mrr = []
    f_recall = []
    for pred_id in range(2):
        f_mrr.append(open(pjoin(FLAGS.data_dir, FLAGS.out_dir, 'mrr%d.%d_%d' % (pred_id, FLAGS.start, FLAGS.end)), 'w'))
        f_recall.append(open(pjoin(FLAGS.data_dir, FLAGS.out_dir, 'recall%d.%d_%d') % (pred_id, FLAGS.start, FLAGS.end), 'w'))
    f_perplex = open(pjoin(FLAGS.data_dir, FLAGS.out_dir, 'perplex.%d_%d') % (FLAGS.start, FLAGS.end), 'w')

    with tf.Session() as sess:
        model, epoch = create_model(sess)
        print('epoch', epoch)
        for i in range(FLAGS.start_batch, num_chunk):
            start = i * FLAGS.batch_size
            end = min(num_line, (i + 1) * FLAGS.batch_size)
            cur_len = len_x[start:end]
            max_len = max(cur_len)
            if max_len > 125:
                chunk_size = 1
                cur_num_chunk = int(FLAGS.batch_size / chunk_size)
                for j in range(cur_num_chunk):
                    cur_start = i * FLAGS.batch_size + j * chunk_size
                    cur_end = min(num_line, cur_start + chunk_size)
                    mrr, recall, perplex = decode_batch(sess, chunk_size, cur_start, cur_end)
            else:
                mrr, recall, perplex = decode_batch(sess, FLAGS.batch_size, start, end)
            for pred_id in range(2):
                f_mrr[pred_id].write(mrr[pred_id])
                f_recall[pred_id].write(recall[pred_id])
            f_perplex.write(perplex[pred_id])
        for pred_id in range(4):
            f_mrr[pred_id].close()
            f_recall[pred_id].close()
        f_perplex.close()



def main(_):
    decode()

if __name__ == "__main__":
    tf.app.run()
