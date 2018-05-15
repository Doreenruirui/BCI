from __future__ import division
import numpy as np
from random import shuffle, randint
import string
import math
from bitweight import *

eeg = None
id2char = [b'<pad>', b'<sos>', b'<eos>', b' '] + list(string.ascii_lowercase)
char2id = {k: v for v, k in enumerate(id2char)}


def load_eegs(path='EEGEvidence.txt-high', nbest=3):
    """
    load EEG simulations from txt file
    """
    global eeg
    sample = 0
    eeg = []
    a_sample = []
    for line in open(path).readlines():
        line = line.split()
        if line:
            a_sample.append(float(line[0]))
        else:
            a_sample = np.array(a_sample)
            sorted_list = [a_sample[0]] + sorted(a_sample[1:], reverse=True)
            transformed_dist = [BitWeight(math.e ** (ele)) for ele in sorted_list[:nbest]]
            total = sum(transformed_dist, BitWeight(0))
            normalized_dist = [
                (prob / total).real() for prob in transformed_dist]
            eeg.append(normalized_dist)
            a_sample = []


def generate_clean(ch):
    prob_vec = np.zeros(len(id2char))
    prob_vec[ch] = 1.0
    return prob_vec


def generate_eeg(ch, nbest=3):
    """
    generate according to target and non-target
    symbols a simulated EEG distribution
    """
    # generate a tuple with all symbols (currently does not include "<")
    global eeg
    if ch == 1:
        prob_vec = np.zeros(len(id2char))
        prob_vec[ch] = 1.0
        return prob_vec
    elif ch == 0:
        return np.zeros(len(id2char))
    index = np.arange(len(id2char)).tolist()
    del index[ch]
    index = index[3:]
    shuffle(index)
    sample_id = randint(0, 999)
    sample = eeg[sample_id]
    res = np.zeros(len(id2char))
    res[ch] = sample[0]
    for i in range(nbest - 1):
        res[index[i]] = sample[i + 1]
    return res


def generate_direchlet(ch, num_cand, prior=1, prob_high=0.7, prob_noncand=0.1):
    if ch == 1:
        prob_vec = np.zeros(len(id2char))
        prob_vec[ch] = 1.0
        return prob_vec
    elif ch == 0:
        return np.zeros(len(id2char))
    index = np.arange(len(id2char)).tolist()
    del index[ch]
    index = index[3:]
    shuffle(index)
    prior_vec = np.ones(num_cand)
    prior_vec[num_cand - 1] = prior
    prob = np.random.dirichlet(prior_vec, size=1)[0, :] * (1. - prob_noncand)
    flag_high = np.random.binomial(1, prob_high)
    max_id = np.argmax(prob)
    max_v = prob[max_id]
    prob_new = prob.tolist()
    del prob_new[max_id]
    if not flag_high:
        shuffle(prob_new)
        prob_tgt = prob_new[0]
        prob_new.append(max_v)
        prob_cand = prob_new[1:]
    else:
        prob_tgt = max_v
        prob_cand = prob_new
    prob_vec = np.ones(len(id2char)) * (prob_noncand * 1. / (len(id2char) - num_cand))
    prob_vec[ch] = prob_tgt
    for ind, item in enumerate(index[:num_cand-1]):
        prob_vec[item] = prob_cand[ind]
    return prob_vec


