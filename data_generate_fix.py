from flag import FLAGS
from data_generate import id2char, char2id
from data_simulate_eeg import generate_eeg, load_eegs, generate_direchlet, generate_clean
from os.path import join as pjoin
import  numpy as np
import tensorflow as tf


def generate_prob():
    if FLAGS.random == 'eeg':
        load_eegs(nbest=FLAGS.num_wit)
    x_dev = pjoin(FLAGS.data_dir, FLAGS.dev + '.ids')
    line_x = []
    max_len = 0
    for line in file(x_dev):
        strs = map(int, line.strip('\n').split(' '))
        line_x.append(strs)
        if len(strs) > max_len:
            max_len = len(strs)
    num_line = len(line_x)
    max_len = min(max_len, 300)
    prob_mat = np.zeros((max_len, num_line, len(char2id)))
    for i in range(num_line):
        line = line_x[i]
        src_toks = [char2id['<sos>']] + line[:-1]
        src_toks = src_toks[:max_len]
        if FLAGS.random == 'clean':
            src_probs = [generate_clean(ele) for ele in src_toks]
        elif FLAGS.random == 'eeg':
            src_probs = [generate_eeg(ele, FLAGS.num_wit) for ele in src_toks]
        else:
            src_probs = [generate_direchlet(ele, FLAGS.num_wit,
                                                prior=FLAGS.prior,
                                                prob_noncand=FLAGS.prob_noncand,
                                                prob_high=FLAGS.prob_high)
                            for ele in src_toks]
        src_probs = np.asarray(src_probs)
        prob_mat[:len(line),i,:] = src_probs
    np.save(pjoin(FLAGS.data_dir, FLAGS.dev + '.' + FLAGS.random + '.' + str(FLAGS.num_wit) + '.prob'), prob_mat)

def main(_):
    generate_prob()

if __name__ == "__main__":
    tf.app.run()
