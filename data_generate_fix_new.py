from data_simulate import load_eegs, add_backspace, generate_dirichlet_vector
from os.path import join as pjoin
import argparse
from multiprocessing import Pool

prior_vec = None
prob_vec = None

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='tmp', help='folder of data.')
    parser.add_argument('--dev', type=str, default='train', help='devlopment dataset.')
    parser.add_argument('--prior', type=str, default='3.0_1.0', help='prior for the highest element in Dirichlet Distribution')
    parser.add_argument('--prob_back', type=float, default=0.0, help='probability of backspace in the corpus')
    parser.add_argument('--prob_vec', type=str, default='0.8_1.0', help='probability that the correct input is in the candidates')
    args = parser.parse_args()
    return args

def initialize(prior, prob):
    global prior_vec, prob_vec
    prior_vec = prior
    prob_vec = prob

def generate_sentence(sentence):
    global prior_vec, prob_vec
    res = list(map(lambda ele: generate_dirichlet_vector(ele,
                                                         prior=prior_vec,
                                                         prob_vec=prob_vec),
                   sentence))
    cands = '\t'.join([' '.join(list(map(str, ele[0]))) for ele in res])
    probs = '\t'.join([' '.join(list(map(str, ele[1]))) for ele in res])
    return cands, probs

def generate_prob(FLAGS):
    filename = pjoin(FLAGS.data_dir, '%s' % FLAGS.dev)
    filename =  '%s_prior_%s_prob_%s' % (filename, FLAGS.prior, FLAGS.prob_vec)

    f_cand = open(filename + '.cand', 'w')
    f_prob = open(filename + '.prob', 'w')

    prob = list(map(float, FLAGS.prob_vec.split('_')))
    prior = list(map(float, FLAGS.prior.split('_')))

    pool = Pool(processes=50, initializer=initialize(prior, prob))
    chunk_size = 10000
    total_line = 0
    with open(pjoin(FLAGS.data_dir, FLAGS.dev + '.ids')) as f_in:
        lno = 0
        chunk_lines = []
        for line in f_in:
            lno += 1
            line = list(map(int, line.strip().split(' ')))
            # generate_sentence(line)
            # print(lno)
            chunk_lines.append(line)
            if lno % chunk_size == 0:
                res = pool.map(generate_sentence, chunk_lines)
                f_cand.write('\n'.join([ele[0] for ele in res]) + '\n')
                f_prob.write('\n'.join([ele[1] for ele in res]) + '\n')
                chunk_lines = []
        if len(chunk_lines) != 0:
            res = pool.map(generate_sentence, enumerate(chunk_lines))
            f_cand.write('\n'.join([ele[0] for ele in res]) + '\n')
            f_prob.write('\n'.join([ele[1] for ele in res]) + '\n')
    f_prob.close()
    f_cand.close()


def main():
    FLAGS = get_args()
    generate_prob(FLAGS)

if __name__ == "__main__":
    main()
