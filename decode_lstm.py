from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join as pjoin
import os
import numpy as np
import tensorflow as tf
import model_concat as model_concat
from flag import FLAGS
from data_generate import id2char, char2id
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
    model = model_concat.Model(FLAGS.size, FLAGS.num_layers,
                        FLAGS.max_gradient_norm, FLAGS.learning_rate,
                        FLAGS.learning_rate_decay_factor, forward_only=True,
                        optimizer=FLAGS.optimizer, num_pred=FLAGS.num_cand, num_wit=FLAGS.num_wit)
    model.build_lstm(vocab_size_char, model_sum=FLAGS.flag_sum)
    ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
    if ckpt:
        logging.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
        num_epoch = int(ckpt.model_checkpoint_path.split('-')[1])
        print (num_epoch)
    else:
        raise ValueError("No trained Model Found")


def decode_string(list_tokens):
    return map(lambda tokens: ''.join([id2char[tok] for tok in tokens[1:]]), list_tokens)


def get_mask(len_input, max_len):
    mask = map(lambda ele: [1] * min(ele, max_len) + [0] * (max_len - min(ele, max_len)), len_input)
    return np.asarray(mask).T


def decode_batch(sess, source_tokens, source_probs, source_mask, target_tokens):
    cur_max_len = max(np.sum(source_mask, axis=0))
    cur_chunk_len = np.sum(source_mask, axis=0)
    batch_size = source_tokens.shape[1]

    cur_recall = np.zeros((batch_size, cur_max_len))
    cur_mrr = np.zeros((batch_size, cur_max_len))

    perplex, top_seqs = model.decode_lstm(sess, source_tokens, source_probs, target_tokens, source_mask, FLAGS.flag_sum)
    top_seqs = np.transpose(top_seqs, [1, 2, 0])
    truth = np.transpose(target_tokens)
    cur_mrr[:, :] = batch_mrr(top_seqs, truth, 10)
    cur_recall[:, :] = batch_recall_at_k(top_seqs, truth, 10)
    cur_perplex = np.transpose(perplex)[:, :-1]
    str_mrr = '\n'.join(list(map(lambda a, len: '\t'.join(list(map(str, a[:len]))),
                                 cur_mrr, cur_chunk_len))) + '\n'
    str_recall = '\n'.join(list(map(lambda a, len: '\t'.join(list(map(str, a[:len]))),
                                    cur_recall, cur_chunk_len))) + '\n'
    str_perplex = '\n'.join(list(map(lambda a, len: '\t'.join(list(map(str, a[:len]))),
                                     cur_perplex, cur_chunk_len))) + '\n'
    str_predict = '\n'.join(list(map(lambda a, len: '\t'.join(list(map(lambda b:
                                                                       ' '.join(list(map(str, b))),
                                                                       a[:len]))),
                                     np.transpose(top_seqs, [0, 2, 1]),cur_chunk_len))) + '\n'
    return str_mrr, str_recall, str_perplex, str_predict


def decode():
    if not os.path.exists(pjoin(FLAGS.data_dir, FLAGS.out_dir)):
       os.makedirs(pjoin(FLAGS.data_dir, FLAGS.out_dir))
    f_mrr = open(pjoin(FLAGS.data_dir, FLAGS.out_dir, 'mrr.%d_%d' % (FLAGS.start, FLAGS.end)), 'w')
    f_recall = open(pjoin(FLAGS.data_dir, FLAGS.out_dir, 'recall.%d_%d') % (FLAGS.start, FLAGS.end), 'w')
    f_perplex = open(pjoin(FLAGS.data_dir, FLAGS.out_dir, 'perplex.%d_%d') % (FLAGS.start, FLAGS.end), 'w')
    f_predict = open(pjoin(FLAGS.data_dir, FLAGS.out_dir, 'top.%d_%d') % (FLAGS.start, FLAGS.end), 'w')
    with tf.Session() as sess:
        create_model(sess)
        for source_tokens, source_probs, source_mask, target_tokens \
                in pair_iter(FLAGS.data_dir, FLAGS.dev, FLAGS.num_wit,
                             cur_len=-2, num_top=FLAGS.num_top,
                             max_seq_len=FLAGS.max_seq_len,
                             data_random=FLAGS.random,
                             batch_size=FLAGS.batch_size,
                             prior=FLAGS.prior, prob_high=FLAGS.prob_high,
                             prob_in=FLAGS.prob_in,
                             flag_generate=FLAGS.flag_generate,
                             prob_back=FLAGS.prob_back,
                             start=FLAGS.start, end=FLAGS.end,
                             flag_vector=not FLAGS.flag_sum):
            mrr, recall, perplex, predict = decode_batch(sess, source_tokens,
                                                         source_probs, source_mask,
                                                         target_tokens)
            f_mrr.write(mrr)
            f_recall.write(recall)
            f_perplex.write(perplex)
            f_predict.write(predict)
        f_mrr.close()
        f_recall.close()
        f_perplex.close()
        f_predict.close()


def main(_):
    decode()

if __name__ == "__main__":
    tf.app.run()
