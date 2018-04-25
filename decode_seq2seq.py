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

import numpy as np
from six.moves import xrange
import tensorflow as tf

import model_concat
from flag import FLAGS
from data_generation import id2char, char2id, get_confusion_concat_distributed, load_vocabulary
from evaluate import batch_mrr, batch_recall_at_k

import logging
logging.basicConfig(level=logging.INFO)


id2word, word2id = None, None


def create_model(session,  vocab_size_char, vocab_size_word):
    model = model_concat.Model(FLAGS.size, FLAGS.num_cand, FLAGS.num_layers,
                        FLAGS.max_gradient_norm, FLAGS.learning_rate,
                        FLAGS.learning_rate_decay_factor, forward_only=True,
                        optimizer=FLAGS.optimizer, num_pred=FLAGS.num_cand)
    model.build_model(vocab_size_char, model=FLAGS.model, flag_bidirect=True)
    ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
    num_epoch = 0
    if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
        logging.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
        num_epoch = int(ckpt.model_checkpoint_path.split('-')[1])
        print (num_epoch)
    else:
        logging.info("Created model with fresh parameters.")
        session.run(tf.initialize_all_variables())
        logging.info('Num params: %d' % sum(v.get_shape().num_elements() for v in tf.trainable_variables()))
    return model, num_epoch


def decode_string(list_tokens, revocab):
    if FLAGS.flag_word:
        return map(lambda tokens: ' '.join([id2word[ele] if revocab[ele] != ' ' else '<space>' for ele in tokens[1:]]), list_tokens)
    else:
        return map(lambda tokens: ''.join([id2char[tok] for tok in tokens[1:]]), list_tokens)

def get_mask(len_input, max_len):
    num_line = len(len_input)
    # mask = np.zeros((num_line, max_len))
    mask = map(lambda ele: [1] * min(ele, max_len) + [0] * (max_len - min(ele, max_len)), len_input)
    return np.asarray(mask).T

def decode():
    global word2id, id2word
    logging.info("Get NLC data in %s" % FLAGS.data_dir)
    x_dev = pjoin(FLAGS.data_dir, FLAGS.dev + '.ids')
    vocab_size_char = len(id2char)
    word2id, id2word = load_vocabulary(FLAGS.data_dir)
    vocab_size_word = len(word2id)
    if FLAGS.flag_word:
        rev_vocab  = id2word
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
    num_line = len(line_x)
    data = np.load(pjoin(FLAGS.data_dir, FLAGS.dev + '.prob.npy'))
    data = data[:,FLAGS.start:FLAGS.end,:]
    data = data[:, sorted_index, :]
    num_chunk = int(np.ceil(num_line * 1. / FLAGS.batch_size))
    f_mrr =  open(pjoin(FLAGS.data_dir, FLAGS.out_dir, 'mrr.%d_%d' % (FLAGS.start, FLAGS.end)), 'w')
    f_recall = open(pjoin(FLAGS.data_dir, FLAGS.out_dir, 'recall.%d_%d') % (FLAGS.start, FLAGS.end), 'w')
    f_perplex = open(pjoin(FLAGS.data_dir, FLAGS.out_dir, 'perplex.%d_%d') % (FLAGS.start, FLAGS.end), 'w')
    with tf.Session() as sess:
        model, epoch = create_model(sess, vocab_size_char, vocab_size_word)
        print('epoch', epoch)
        for i in range(num_chunk):
            start = i * FLAGS.batch_size
            end = min(num_line, (i + 1) * FLAGS.batch_size)
            cur_len = len_x[start:end]
            max_len = max(cur_len)
            src_probs = data[:max_len, start:end, :]
            tgt_toks = map(lambda line: [char2id['<sos>']] + line + [char2id['<pad>']] * (max_len - len(line)), line_x[start:end])
            tgt_toks = np.asarray(tgt_toks).T
            cur_recall = np.zeros((FLAGS.batch_size, max_len))
            cur_mrr = np.zeros((FLAGS.batch_size, max_len))
            cur_perplex = np.zeros((FLAGS.batch_size, max_len))
            for k in range(max_len):
                print(i, k, max_len)
                cur_mask = get_mask(cur_len, k + 1)
                outputs, prediction, top_seqs, perplex = model.decode_beam(sess, src_probs[:k+1,:,:], tgt_toks[:k+2,:], cur_mask, FLAGS.beam_size)
                cur_pred = top_seqs[:, :, k+1]
                cur_truth = tgt_toks[k + 1, :]
                cur_mrr[:, k] = batch_mrr(cur_pred, cur_truth)
                cur_recall[:, k] = batch_recall_at_k(cur_pred, cur_truth, 10)
                cur_perplex[:, k] = perplex[:, k + 1]
            f_mrr.write('\n'.join(map(lambda a, len: '\t'.join(map(str, a[:len])), cur_mrr, cur_len)) + '\n')
            f_recall.write('\n'.join(map(lambda a, len: '\t'.join(map(str, a[:len])), cur_recall, cur_len)) + '\n')
            f_perplex.write('\n'.join(map(lambda a, len: '\t'.join(map(str, a[:len])), cur_perplex, cur_len)) + '\n')
        f_mrr.close()
        f_recall.close()
        f_perplex.close()



def main(_):
    decode()

if __name__ == "__main__":
    tf.app.run()
