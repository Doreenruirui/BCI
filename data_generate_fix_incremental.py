from os.path import join as pjoin
import argparse
from multiprocessing import Pool
import numpy as np
from random import shuffle

num_wit = 0
prior = 0

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='tmp', help='folder of data.')
    parser.add_argument('--dev', type=str, default='train', help='devlopment dataset.')
    parser.add_argument('--prior', type=float, default=3.0, help='prior for the highest element in Dirichlet Distribution')
    parser.add_argument('--prob_back', type=float, default=0.0, help='probability of backspace in the corpus')
    parser.add_argument('--prob', type=float, default=0.5, help='probability that the correct input is in the candidates')
    args = parser.parse_args()
    return args

def initialize(nwit, prior_vec):
    global num_wit, prior
    num_wit = nwit
    prior = prior_vec

def generate_cands_char(ch, rank, num_wit):
    index = np.arange(30).tolist()
    index = index[3:]
    index.remove(ch)
    shuffle(index)
    return index[:rank] + [ch] + index[rank:num_wit - 1]

def generate_sentence(paras):
    global num_wit, prior
    sen, sen_rank = paras
    sen_cand = []
    rank_str = []
    sen_prob = []
    for ele, rank in zip(sen, sen_rank):
        cand = generate_cands_char(ele, rank, num_wit)
        sen_cand.append(' '.join(list(map(str, cand))))
        prior_vec = np.ones(5)
        prior_vec[0] = prior
        prob = np.random.dirichlet(prior_vec, size=1)[0, :]
        prob = np.sort(prob)[::-1]
        sen_prob.append(' '.join(list(map(str, prob))))
    rank_str.append('\t'.join(list(map(str, sen_rank))))
    return '\t'.join(sen_cand), '\t'.join(sen_prob), '\t'.join(rank_str)


def get_noisy(data_dir, dev, prior, prob_high):
    lno = 0
    lines = []
    sen2cid = []
    nchar = 0
    with open(pjoin(data_dir, '0.0', dev + '.ids')) as f_clean:
        for line in f_clean:
            items = list(map(int, line.strip().split(' ')))
            lines.append(items)
            sen2cid.append(np.arange(len(items)) + nchar)
            nchar += len(items)
            lno += 1
    num_chunk = int((1 - prob_high) / 0.025)
    high_size = int(np.around(nchar * prob_high))
    chunk_size = int(np.around(nchar * 0.025))
    index = np.arange(nchar)
    shuffle(index)
    id2rank = np.zeros(nchar, dtype=int)
    for i in range(high_size, nchar):
        cur_index = index[i]
        bin_no = int(np.floor((i - high_size) / chunk_size)) + 1
        id2rank[cur_index] = bin_no
    sen2rank = []
    num_sen = len(lines)
    for i in range(num_sen):
        cur_cid = sen2cid[i]
        cur_rank = [id2rank[ele] for ele in cur_cid]
        sen2rank.append(cur_rank)
    pool = Pool(processes=50, initializer=initialize(num_chunk + 1, float(prior)))
    sen_chunk_size = 10000
    num_sen_chunk = int(np.ceil(num_sen / sen_chunk_size))
    f_cand = open(pjoin(data_dir, '0.5', dev + '.cand'), 'w')
    f_prob = open(pjoin(data_dir, '0.5', dev + '.prob'), 'w')
    f_rank = open(pjoin(data_dir, '0.5', dev + '.rank'), 'w')
    for i in range(num_sen_chunk):
        start = sen_chunk_size * i
        end = min(sen_chunk_size + start, num_sen)
        res = pool.map(generate_sentence, zip(lines[start:end], sen2rank[start:end]))
        for cand, prob, rank in res:
            f_cand.write(cand + '\n')
            f_prob.write(prob + '\n')
            f_rank.write(rank + '\n')
    f_cand.close()
    f_prob.close()
    f_rank.close()

def main():
    FLAGS = get_args()
    get_noisy(FLAGS.data_dir, FLAGS.dev, FLAGS.prior, FLAGS.prob)

if __name__ == "__main__":
    main()
