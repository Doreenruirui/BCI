from flag import FLAGS
from data_simulate import load_eegs, add_backspace, generate_simulation
from os.path import join as pjoin
import tensorflow as tf


def generate_prob():
    filename = pjoin(FLAGS.data_dir, '%s.%s.%d' % (FLAGS.dev,
                                                   FLAGS.random,
                                                   FLAGS.num_top))
    if FLAGS.random == 'random':
        filename =  '%s_%.2f_%.2f_%.2f' % (filename, FLAGS.prob_high,
                                           FLAGS.prior, FLAGS.prob_in)
    else:
        load_eegs()

    if FLAGS.prob_back > 0:
        filename = '%s_%.2f' % (filename, FLAGS.prob_back)
        f_out = open(filename + '.ids', 'w')

    f_cand = open(filename + '.cand', 'w')
    f_prob = open(filename + '.prob', 'w')

    with open(pjoin(FLAGS.data_dir, FLAGS.dev + '.ids')) as f_in:
        for line in f_in:
            line = list(map(int, line.strip('\n').split(' ')))
            src_toks = line[:FLAGS.max_seq_len]
            if FLAGS.prob_back > 0.:
                new_ids  = add_backspace(src_toks, FLAGS.prob_back)
                f_out.write(' '.join(map(str, new_ids)) + '\n')
                src_toks = new_ids
            res = list(map(lambda ele: generate_simulation(ele, FLAGS.num_wit,
                                                      FLAGS.num_top,
                                                      prior=FLAGS.prior,
                                                      prob_in = FLAGS.prob_in,
                                                      prob_high=FLAGS.prob_high,
                                                      flag_vec=False,
                                                      simul=FLAGS.random),
                      src_toks))

            f_prob.write('\t'.join([' '.join(list(map(str, ele[1]))) for ele in res]) + '\n')
            f_cand.write('\t'.join([' '.join(list(map(str, ele[0]))) for ele in res]) +'\n')

    f_prob.close()
    f_cand.close()
    if FLAGS.prob_back > 0:
        f_out.close()


def main(_):
    generate_prob()

if __name__ == "__main__":
    tf.app.run()
