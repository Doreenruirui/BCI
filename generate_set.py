from flag import FLAGS
from data_generation import id2char, char2id, get_confusion_concat_distributed, load_vocabulary
from os.path import join as pjoin
import  numpy as np
import tensorflow as tf


def generate_prob():
    x_dev = pjoin(FLAGS.data_dir, FLAGS.dev + '.ids')
    line_x = []
    max_len = 0
    for line in file(x_dev):
        strs = map(int, line.strip('\n').split(' '))
        line_x.append(strs)
        if len(strs) > max_len:
            max_len = len(strs)
    num_line = len(line_x)
    prob_mat = np.zeros((max_len, num_line, len(char2id)))
    for i in range(num_line):
        line = line_x[i]
        src_toks = [char2id['<sos>']] + line[:-1]
        src_probs = [get_confusion_concat_distributed(ele, FLAGS.num_wit, prior=FLAGS.prior,
                                                   prob_noncand=FLAGS.prob_noncand,
                                                   prob_high=FLAGS.prob_high) for ele in src_toks]
        src_probs = np.asarray(src_probs)
        prob_mat[:len(line),i,:] = src_probs
    np.save(pjoin(FLAGS.data_dir, FLAGS.dev + '.prob'), prob_mat)


def main(_):
    generate_prob()

if __name__ == "__main__":
    tf.app.run()
