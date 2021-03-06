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
    parser.add_argument('--prior', type=str, default='3.0', help='prior for the highest element in Dirichlet Distribution')
    parser.add_argument('--prob_back', type=float, default=0.0, help='probability of backspace in the corpus')
    parser.add_argument('--prob', type=str, default='0.5', help='probability that the correct input is in the candidates')
    parser.add_argument('--task', type=str, default='base', help='base/cand/prob')

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


def initialize_fix(arg_bin, arg_nwit):
    global bin, nwit
    bin = arg_bin
    nwit = arg_nwit


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

def get_bin(prob_vec):
    prob_vec = list(map(float, prob_vec.split('_')))
    num_wit = len(prob_vec)
    bin = [[] for _ in range(num_wit)]
    bin[0].append(0)
    bin_no = 0
    for i in range(num_wit):
        cur_prob = prob_vec[i]
        if i == 0:
            num_bin = int(np.around((cur_prob - 0.5) / 0.025))
        else:
            num_bin = int(np.around(cur_prob / 0.025))
        bin[i] += [ele + bin_no + 1 for ele in range(num_bin)]
        bin_no += num_bin
    dict_bin = {}
    for i in range(num_wit):
        for ele in bin[i]:
            dict_bin[ele] = i
    print(dict_bin)
    return dict_bin

def process_line(paras):
    global bin, nwit
    linex, linec, liner = paras
    sen = list(map(int, linex.strip('\n').split(' ')))
    newc = linec.strip().split('\t')
    newr = liner.strip().split('\t')
    cands = list(map(lambda ele: [int(s) for s in ele.split()], newc))
    ranks = list(map(int, newr))
    res_cand = []
    for tok, rank, cand in zip(sen, ranks, cands):
        if rank in bin:
            cur_bin = bin[rank]
            new_cand = [ele for ele in cand]
            if rank < nwit:
                new_cand[rank] = new_cand[cur_bin]
            new_cand[cur_bin] = tok
            new_cand = new_cand[:nwit]
        else:
            new_cand = cand[:nwit]
        res_cand.append(new_cand)
    return '\t'.join([' '.join(list(map(str, ele))) for ele in res_cand])

def generate_new(FLAGS):
    dict_bin = get_bin(FLAGS.prob)
    num_wit = len(FLAGS.prob.split('_'))
    pool = Pool(processes=50, initializer=initialize_fix(dict_bin, num_wit))
    fx = open(pjoin(FLAGS.data_dir, '0.0', FLAGS.dev + '.ids'))
    fc = open(pjoin(FLAGS.data_dir, '0.5', '%s.cand' % FLAGS.dev))
    fr = open(pjoin(FLAGS.data_dir, '0.5', '%s.rank' % FLAGS.dev))
    fout_c = open(pjoin(FLAGS.data_dir, '0.0',
                        '%s_prior_%s_prob_%s.cand' % (FLAGS.dev, FLAGS.prior, FLAGS.prob)), 'w')
    lines = []
    linex, linec, liner = fx.readline(), fc.readline(), fr.readline()
    while linex and linec and liner:
        lines.append((linex, linec, liner))
        if len(lines) == 10000:
            res = pool.map(process_line, lines)
            print(len(res))
            fout_c.write('\n'.join(res) + '\n')
            lines = []
            #print(len(lines))
        linex, linec, liner = fx.readline(), fc.readline(), fr.readline()
    print(len(lines))
    if len(lines) > 0:
        res = pool.map(process_line, lines)
        fout_c.write('\n'.join(res) + '\n')
    pool.close()
    fx.close()
    fr.close()
    fc.close()


def main():
    FLAGS = get_args()
    #get_noisy(FLAGS.data_dir, FLAGS.dev, 3, 0.5)
    generate_new(FLAGS)


if __name__ == "__main__":
    main()
