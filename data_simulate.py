from __future__ import division
import numpy as np
from random import shuffle, randint
import string
import math
from bitweight import *
import sys

eeg = None
id2char = [b'<pad>', b'<sos>', b'<eos>', b' '] + list(string.ascii_lowercase)
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
        return index[:rank_ch] + [ch] + index[rank_ch: num_wit - 1], sample[1: rank_ch + 1] + [sample[0]] + sample[rank_ch + 1: num_wit]
    else:
        return index[:num_wit], sample[1 :num_wit + 1]

def generate_direchlet(ch, index, num_wit, prior=1, prob_high=0.7, prob_in=0.78):
    prior_vec = np.ones(num_wit)
    prior_vec[0] = prior
    prob = np.random.dirichlet(prior_vec, size=1)[0, :]
    flag_high = np.random.binomial(1, prob_high)
    prob = np.sort(prob)[::-1]
    flag_in = np.random.binomial(1, prob_in)
    if flag_in:
        if flag_high:
            return [ch] + index[:num_wit - 1], prob
        else:
            ins_id = np.random.choice(num_wit - 1, 1)[0]
            return index[:ins_id + 1] + [ch] + index[ins_id + 1: num_wit - 1], prob
    else:
        return index[:num_wit], prob

def generate_simulation(ch, num_wit, top=10, flag_vec=True, simul='eeg', prior=1,
                        prob_high=0.7, prob_in=0.78):
    if simul != 'clean':
        index = np.arange(len(id2char)).tolist()
        del index[ch]
        index = index[3:]
        shuffle(index)
        if simul == 'eeg':
            cand, prob = generate_eeg(ch, index, top)
        else:
            cand, prob = generate_direchlet(ch, index, top, prior=prior, prob_high=prob_high, prob_in=prob_in)
    else:
        cand, prob = [ch], [1.0]
    if flag_vec:
        return create_vector(cand[:num_wit], prob[:num_wit])
    else:
        return cand, prob

# load_eegs(nbest=28)
# for i in range(100):
#     generate_eeg(2, num_wit=10)
    # generate_direchlet(2, num_wit=10, prior=3, prob_high=0.5, prob_in=0.5)
