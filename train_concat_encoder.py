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

import math
import os
import sys
import time
import random
import json
from os.path import join as pjoin

import numpy as np
from six.moves import xrange
import tensorflow as tf

import model_concat_encoder as model_concat
from flag import FLAGS
from data_generate import pair_iter_distributed as pair_iter, id2char

import logging
logging.basicConfig(level=logging.INFO)


def create_model(session, vocab_size_char, vocab_size_word):
    model = model_concat.Model(FLAGS.size, FLAGS.num_wit, FLAGS.num_layers,
                        FLAGS.max_gradient_norm, FLAGS.learning_rate,
                        FLAGS.learning_rate_decay_factor, forward_only=False,
                        optimizer=FLAGS.optimizer)
    model.build_model(vocab_size_char, model=FLAGS.model, flag_bidirect=FLAGS.flag_bidirect)
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


def validate(model, sess, x_dev, flag_seq2seq):
    valid_costs, valid_lengths = [], []
    for source_tokens, source_mask, target_tokens in pair_iter(x_dev, FLAGS.num_wit,
                                                               data_random=FLAGS.random,
                                                               flag_seq2seq=flag_seq2seq,
                                                               batch_size=FLAGS.batch_size,
                                                               prob_high=FLAGS.prob_high,
                                                               prob_noncand=FLAGS.prob_noncand,
                                                               prior=FLAGS.prior,
                                                               sort_and_shuffle=False):
        cost = model.test(sess, source_tokens, source_mask, target_tokens)
        valid_costs.append(cost * source_mask.shape[1])
        valid_lengths.append(np.sum(source_mask[1:, :]))
    valid_cost = sum(valid_costs) / float(sum(valid_lengths))
    return valid_cost


def train():
    """Train a translation model using NLC data."""
    # Prepare NLC data.
    logging.info("Get NLC data in %s" % FLAGS.data_dir)
    x_train = pjoin(FLAGS.data_dir, 'train.ids')
    x_dev = pjoin(FLAGS.data_dir, FLAGS.dev + '.ids')
    vocab_size = len(id2char)
    logging.info("Vocabulary size: %d" % vocab_size)
    if not os.path.exists(FLAGS.train_dir):
        os.makedirs(FLAGS.train_dir)
    if FLAGS.model == 'seq2seq':
        flag_seq2seq = 1
    else:
        flag_seq2seq = 0
    file_handler = logging.FileHandler("{0}/log.txt".format(FLAGS.train_dir))
    logging.getLogger().addHandler(file_handler)
    with open(os.path.join(FLAGS.train_dir, "flags.json"), 'w') as fout:
        json.dump(FLAGS.__flags, fout)

    with tf.Session() as sess:
        logging.info("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))
        model, epoch = create_model(sess, vocab_size, False)

        if False:
            tic = time.time()
            params = tf.trainable_variables()
            num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
            toc = time.time()
            print ("Number of params: %d (retreival took %f secs)" % (num_params, toc - tic))

        best_epoch = 0
        previous_losses = []
        exp_cost = None
        exp_length = None
        exp_norm = None
        total_iters = 0
        start_time = time.time()
        while (FLAGS.epochs == 0 or epoch < FLAGS.epochs):
            epoch += 1
            print(epoch)
            current_step = 0

            ## Train
            epoch_tic = time.time()
            for source_tokens, source_mask, target_tokens in pair_iter(x_train, FLAGS.num_wit,
                                                                       data_random=FLAGS.random,
                                                                       flag_seq2seq=flag_seq2seq,
                                                                       batch_size=FLAGS.batch_size,
                                                                       prob_high=FLAGS.prob_high,
                                                                       prob_noncand=FLAGS.prob_noncand,
                                                                       prior=FLAGS.prior):
                # Get a batch and make a step.
                tic = time.time()

                grad_norm, cost, param_norm = model.train(sess, source_tokens, source_mask, target_tokens, FLAGS.keep_prob)

                toc = time.time()
                iter_time = toc - tic
                total_iters += np.sum(source_mask)
                tps = total_iters / (time.time() - start_time)
                current_step += 1
                print(current_step)
                lengths = np.sum(source_mask, axis=0)
                mean_length = np.mean(lengths)
                std_length = np.std(lengths)

                if not exp_cost:
                    exp_cost = cost
                    exp_length = mean_length
                    exp_norm = grad_norm
                else:
                    exp_cost = 0.99*exp_cost + 0.01*cost
                    exp_length = 0.99*exp_length + 0.01*mean_length
                    exp_norm = 0.99*exp_norm + 0.01*grad_norm
                    exp_norm = 0.99*exp_norm + 0.01*grad_norm

                cost = cost / mean_length

                if current_step % FLAGS.print_every == 0:
                    logging.info('epoch %d, iter %d, cost %f, exp_cost %f, grad norm %f, param norm %f, tps %f, length mean/std %f/%f' %
                                 (epoch, current_step, cost, exp_cost / exp_length, grad_norm, param_norm, tps, mean_length, std_length))
            epoch_toc = time.time()

            ## Checkpoint
            checkpoint_path = os.path.join(FLAGS.train_dir, "best.ckpt")

            ## Validate
            valid_cost = validate(model, sess, x_dev, flag_seq2seq)

            logging.info("Epoch %d Validation cost: %f time: %f" % (epoch, valid_cost, epoch_toc - epoch_tic))

            if len(previous_losses) > 2 and valid_cost > previous_losses[-1]:
                logging.info("Annealing learning rate by %f" % FLAGS.learning_rate_decay_factor)
                sess.run(model.lr_decay_op)
                model.saver.restore(sess, checkpoint_path + ("-%d" % best_epoch))
            else:
                previous_losses.append(valid_cost)
                best_epoch = epoch
                model.saver.save(sess, checkpoint_path, global_step=epoch)
            sys.stdout.flush()


def main(_):
    train()

if __name__ == "__main__":
    tf.app.run()
