from __future__ import division
import numpy as np
from random import shuffle, randint
import string
import math
from bitweight import *
import sys

eeg = None
id2char = [b'<pad>', b'<sos>', b'<eos>', b' '] + list(string.ascii_lowercase) + ['<backspace>']
char2id = {k: v for v, k in enumerate(id2char)}


def create_vector(cand, prob):
    a = np.zeros(len(char2id))
    sum_p = sum(prob)
    prob = np.asarray(prob) / sum_p
    a[cand] = prob
    return a


def load_eegs(path='EEGEvidence.txt-high'):
    """
    load EEG simulations from txt file
    """
    global eeg
    eeg = []
    a_sample = []
    for line in open(path).readlines():
        line = line.split()
        if line:
            a_sample.append(float(line[0]))
        else:
            a_sample = np.array(a_sample)
            sorted_list = [a_sample[0]] + sorted(a_sample[1:], reverse=True)
            transformed_dist = [BitWeight(math.e ** (ele)) for ele in sorted_list]
            total = sum(transformed_dist, BitWeight(0))
            normalized_dist = [
                (prob / total).real() for prob in transformed_dist]
            eeg.append(normalized_dist)
            a_sample = []


def generate_eeg(ch, index, num_wit=3):
    """
    generate according to target and non-target
    symbols a simulated EEG distribution
    """
    # generate a tuple with all symbols (currently does not include "<")
    global eeg
    sample_id = randint(0, 999)
    sample = eeg[sample_id]
    rank_ch = 0
    for i in range(1, num_wit + 1):
        if sample[i] <= sample[0]:
            break
        rank_ch = i
    if rank_ch < 10:
        return index[:rank_ch] + [ch] + index[rank_ch: num_wit - 1], \
               sample[1: rank_ch + 1] + [sample[0]] + sample[rank_ch + 1: num_wit]
    else:
        return index[:num_wit], sample[1 :num_wit + 1]


def generate_candidate(ch):
    index = np.arange(len(id2char)).tolist()
    index = index[3:]
    index.remove(ch)
    shuffle(index)
    return index


def dirichlet_distribution(num_wit, prior):
    prior_vec = np.ones(num_wit)
    prior_vec[0] = prior
    prob = np.random.dirichlet(prior_vec, size=1)[0, :]
    prob = np.sort(prob)[::-1]
    return prob


def generate_dirichlet(ch, index, num_wit, prior=1., prob_high=0.7, prob_in=0.78):
    prob = dirichlet_distribution(num_wit, prior)
    flag_in = np.random.binomial(1, prob_in)
    if flag_in:
        flag_high = np.random.binomial(1, prob_high)
        if flag_high:
            return [ch] + index[:num_wit - 1], prob
        else:
            ins_id = np.random.choice(num_wit - 1, 1)[0]
            return index[:ins_id + 1] + [ch] + index[ins_id + 1: num_wit - 1], prob
    else:
        return index[:num_wit], prob


def generate_simulation(ch, num_wit, top=10, flag_vec=True, simul='eeg',
                        prior=1., prob_high=0.7, prob_in=0.78):
    if simul != 'clean':
        index = generate_candidate(ch)
        if simul == 'eeg':
            cand, prob = generate_eeg(ch, top)
        else:
            cand, prob = generate_dirichlet(ch, index, top, prior=prior,
                                            prob_high=prob_high,
                                            prob_in=prob_in)
    else:
        cand, prob = [ch], [1.0]
    if flag_vec:
        return create_vector(cand[:num_wit], prob[:num_wit])
    else:
        return cand, prob


def generate_backspace(ch, num_wit, top=10, flag_vec=True,
                       prior=1., prob_high=0.7, prob_in=0.78):
    index = generate_candidate(ch)
    cur_err = index[0]
    err_cand, err_prob = generate_dirichlet(cur_err,
                                            generate_candidate(cur_err),
                                            top,
                                            prior=prior,
                                            prob_high=prob_high,
                                            prob_in=prob_in)
    back_id = char2id['<backspace>']
    back_cand, back_prob = generate_dirichlet(back_id,
                                              generate_candidate(back_id),
                                              top,
                                              prior=prior,
                                              prob_high=prob_high,
                                              prob_in=prob_in)
    ch_cand, ch_prob = generate_dirichlet(ch, generate_candidate(ch),
                                          top,
                                          prior=prior,
                                          prob_in=prob_in,
                                          prob_high=prob_high)
    if not flag_vec:
        return [cur_err, err_cand, err_prob], [back_id, back_cand, back_prob], \
               [ch, ch_cand, ch_prob]
    else:
        return (cur_err, create_vector(err_cand[:num_wit], err_prob[:num_wit])), \
               (back_id, create_vector(back_cand[:num_wit], back_prob[:num_wit])), \
               (ch, create_vector(ch_cand[:num_wit], ch_prob[:num_wit]))


def add_backspace(line, prob_back):
    nc = len(line)
    nback = int(np.round(nc * prob_back))
    rand_index = np.arange(nc)
    np.random.shuffle(rand_index)
    insert_index = rand_index[:nback]
    new_line = []
    for cid in range(nc):
        if cid in insert_index:
            cand = generate_candidate(line[cid])
            new_line.append(cand[0])
            new_line.append(char2id['<backspace>'])
        new_line.append(line[cid])
    return new_line

# def generate_line(line, num_wit, num_top=10, prior=1., prob_high=0.7,
#                   prob_in=1.0, prob_back=0.0, simul="eeg", flag_vec=False):
#     probs = []
#     tokens = []
#     if not flag_vec:
#         cands = []
#     for ele in line:
#         flag_back = np.random.binomial(1, prob_back)
#         if flag_back:
#             err, back, input = generate_backspace(ele, num_wit, top=num_top,
#                                                   prior=prior, prob_high=prob_high,
#                                                   prob_in=prob_in, flag_vec=flag_vec)
#             if flag_vec:
#                 probs.extend([err[1], back[1], input[1]])
#                 tokens.extend([err[0], back[0], input[0]])
#             else:
#                 cands.extend([err[1], back[1], input[1]])
#                 probs.extend([err[2], back[2], input[2]])
#                 tokens.extend([err[0], back[0], input[0]])
#         else:
#             input = generate_simulation(ele, num_wit,
#                                         num_top,
#                                         prior=prior,
#                                         prob_in=prob_in,
#                                         prob_high=prob_high,
#                                         flag_vec=flag_vec,
#                                         simul=simul)
#             if flag_vec:
#                 probs.append(input)
#                 tokens.append(ele)
#             else:
#                 tokens.append(ele)
#                 cands.append(input[0])
#                 probs.append(input[1])
#     if flag_vec:
#         return probs, tokens
#     else:
#         return cands, probs, tokens

#===========================================test======================================================
# for i in range(100):
   # generate_backspace(char2id['a'], 1, top=5, flag_vec=False, prior=1., prob_high=0.8, prob_in=1.0)
# a = [3, 4, 5, 6, 7, 8]
# for i in range(10):
#     generate_line(a,  5, num_top=5, prior=1., prob_high=0.8, prob_in=1.0, prob_back=0.2, flag_vec=True, simul='random')
# line = [3, 4, 5, 6, 7, 8, 9, 10]
# print add_backspace(line, 0.2)