from flag import FLAGS
from data_simulate_eeg_varlen import load_eegs, generate_simulation
from os.path import join as pjoin
import numpy as np
import tensorflow as tf


def generate_prob():
    if FLAGS.random == 'random':
        filename = pjoin(FLAGS.data_dir, '%s.%s.%d_%.2f_%.2f_%.2f' % (FLAGS.dev,
                                                                     FLAGS.random,
                                                                     FLAGS.num_top,
                                                                     FLAGS.prob_high,
                                                                     FLAGS.prior,
                                                                     FLAGS.prob_in))
    else:
        load_eegs()
        filename = pjoin(FLAGS.data_dir, '%s.%s.%d' % (FLAGS.dev,
                                                       FLAGS.random,
                                                       FLAGS.num_top))

    file_cand = filename + '.cand'
    file_prob = filename + '.prob'
    f_cand = open(file_cand, 'w')
    f_prob = open(file_prob, 'w')

    x_dev = pjoin(FLAGS.data_dir, FLAGS.dev + '.ids')

    for line in file(x_dev):
        line = map(int, line.strip('\n').split(' '))
        src_toks = line[:FLAGS.max_seq_len]
        res = map(lambda ele: generate_simulation(ele, FLAGS.num_wit,
                                                  FLAGS.num_top,
                                                  prior=FLAGS.prior,
                                                  prob_in = FLAGS.prob_in,
                                                  prob_high=FLAGS.prob_high,
                                                  flag_vec=False,
                                                  simul=FLAGS.random),
                  src_toks)
        f_prob.write('\t'.join([' '.join(map(str, ele[1])) for ele in res]) + '\n')
        f_cand.write('\t'.join([' '.join(map(str, ele[0])) for ele in res]) +'\n')
    f_prob.close()
    f_cand.close()


def main(_):
    generate_prob()

if __name__ == "__main__":
    tf.app.run()
