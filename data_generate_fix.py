from data_simulate import load_eegs, add_backspace, simulate_one
from os.path import join as pjoin
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='tmp', help='folder of data.')
    parser.add_argument('--dev', type=str, default='train', help='devlopment dataset.')
    parser.add_argument('--random', type=str, default='random', help='simulation method.')
    parser.add_argument('--num_top', type=int, default=5, help='number of candidates')
    parser.add_argument('--num_wit', type=int, default=5, help='number of witnesses')
    parser.add_argument('--prob_high', type=float, default=0.7, help='probability of correct input ranked as highest among the candidates')
    parser.add_argument('--prior', type=float, default=3.0, help='prior for the highest element in Dirichlet Distribution')
    parser.add_argument('--prob_in', type=float, default=1.0, help='probability that the correct input is in the candidates')
    parser.add_argument('--prob_back', type=float, default=0.0, help='probability of backspace in the corpus')
    args = parser.parse_args()
    return args


def generate_prob(FLAGS):
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
            src_toks = line
            if FLAGS.prob_back > 0.:
                new_ids  = add_backspace(src_toks, FLAGS.prob_back)
                f_out.write(' '.join(map(str, new_ids)) + '\n')
                src_toks = new_ids
            res = list(map(lambda ele: simulate_one(ele, FLAGS.num_wit,
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


def main():
    FLAGS = get_args()
    generate_prob(FLAGS)

if __name__ == "__main__":
    main()
