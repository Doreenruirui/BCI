from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import time
import numpy as np
import tensorflow as tf

import model_concat_lstm as model_concat
from flag import FLAGS
from data_generate import pair_iter, id2char

import logging
logging.basicConfig(level=logging.INFO)


def create_model(session, vocab_size_char):
    model = model_concat.Model(FLAGS.size, FLAGS.num_wit, FLAGS.num_layers,
                               FLAGS.max_gradient_norm, FLAGS.learning_rate,
                               FLAGS.learning_rate_decay_factor,
                               forward_only=False,
                               optimizer=FLAGS.optimizer)
    model.build_lstm(vocab_size_char, model=FLAGS.model,
                      flag_bidirect=FLAGS.flag_bidirect,
                      model_sum=FLAGS.flag_sum)
    ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
    num_epoch = 0
    if ckpt:
        logging.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
        num_epoch = int(ckpt.model_checkpoint_path.split('-')[1])
        print (num_epoch)
    else:
        logging.info("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
        logging.info('Num params: %d' % sum(v.get_shape().num_elements() for v in tf.trainable_variables()))
    return model, num_epoch


def validate(model, sess, cur_len):
    valid_costs, valid_lengths = [], []
    for source_tokens, source_mask, target_tokens in pair_iter(FLAGS.data_dir,
                                                               FLAGS.dev, FLAGS.num_wit,
                                                               cur_len=cur_len,
                                                               num_top=FLAGS.num_top,
                                                               max_seq_len=FLAGS.max_seq_len,
                                                               data_random=FLAGS.random,
                                                               batch_size=FLAGS.batch_size,
                                                               prior=FLAGS.prior,
                                                               prob_high=FLAGS.prob_high,
                                                               prob_in=FLAGS.prob_in,
                                                               flag_generate=FLAGS.flag_generate,
                                                               prob_back=FLAGS.prob_back):
        cost = model.test(sess, source_tokens, source_mask, target_tokens)
        valid_costs.append(cost * source_mask.shape[1])
        valid_lengths.append(np.sum(source_mask))
    valid_cost = sum(valid_costs) / float(sum(valid_lengths))
    return valid_cost


def train():
    """Train a translation model using NLC data."""
    # Prepare NLC data.
    logging.info("Get NLC data in %s" % FLAGS.data_dir)
    vocab_size = len(id2char)
    logging.info("Vocabulary size: %d" % vocab_size)
    if not os.path.exists(FLAGS.train_dir):
        os.makedirs(FLAGS.train_dir)
    file_handler = logging.FileHandler("{0}/log.txt".format(FLAGS.train_dir))
    logging.getLogger().addHandler(file_handler)
    # with open(os.path.join(FLAGS.train_dir, "flags.json"), 'w') as fout:
    #     json.dump(FLAGS.__flags, fout)
    with tf.Session() as sess:
        logging.info("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))
        model, epoch = create_model(sess, vocab_size)

        if False:
            tic = time.time()
            params = tf.trainable_variables()
            num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
            toc = time.time()
            print ("Number of params: %d (retreival took %f secs)" % (num_params, toc - tic))

        best_epoch = epoch
        previous_losses = []
        exp_cost = None
        exp_length = None
        exp_norm = None
        total_iters = 0
        start_time = time.time()
        cur_len = -2

        while (FLAGS.epochs == 0 or epoch < FLAGS.epochs):
            epoch += 1
            current_step = 0

            ## Train
            epoch_tic = time.time()
            if FLAGS.flag_varlen:
                cur_len = epoch - 1

            print('epoch', epoch, cur_len)
            for source_tokens, source_mask, target_tokens in pair_iter(FLAGS.data_dir, 'train',
                                                                       FLAGS.num_wit,
                                                                       cur_len=cur_len,
                                                                       num_top=FLAGS.num_top,
                                                                       max_seq_len=FLAGS.max_seq_len,
                                                                       batch_size=FLAGS.batch_size,
                                                                       data_random=FLAGS.random,
                                                                       prior=FLAGS.prior,
                                                                       prob_high=FLAGS.prob_high,
                                                                       prob_in=FLAGS.prob_in,
                                                                       flag_generate=FLAGS.flag_generate,
                                                                       prob_back=FLAGS.prob_back,
                                                                       sort_and_shuffle=True):
                # Get a batch and make a step.
                tic = time.time()
                grad_norm, cost, param_norm = model.train(sess, source_tokens, source_mask, target_tokens, FLAGS.keep_prob)
                toc = time.time()
                iter_time = toc - tic
                total_iters += np.sum(source_mask)
                tps = total_iters / (time.time() - start_time)
                current_step += 1
                print('iter', current_step)
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
            valid_cost = validate(model, sess, cur_len)

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
