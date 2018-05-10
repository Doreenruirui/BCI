from __future__ import division
import numpy as np
from random import shuffle, randint
import math
from bitweight import *


def eegs(path):
    """
    load EEG simulations from txt file
    """
    sample = 0
    num_sym = 0
    eeg_dict = {}
    a_sample = []
    for line in open(path).readlines():
        line = line.split()
        if line:
            a_sample.append(float(line[0]))
            num_sym += 1
        else:
            num_sym = 0
            a_sample = np.array(a_sample)
            # assuming it's log likelihood and not negative ll
            transformed_dist = [BitWeight(math.e**(prob)) for prob in a_sample]
            total = sum(transformed_dist, BitWeight(1e6))
            normalized_dist = [
                (prob / total).real() for prob in transformed_dist]
            eeg_dict[sample] = [-math.log(prob) for prob in normalized_dist]
            sample += 1
            a_sample = []
    return eeg_dict


def generate_eeg(eeg, ch, nbest=3):
    """
    generate according to target and non-target
    symbols a simulated EEG distribution
    """
    # generate a tuple with all symbols (currently does not include "<")
    syms = syms_set[:]
    idx = syms.index(ch)
    del syms[idx]
    shuffle(syms)
    sample_id = randint(0, 999)
    sample = eeg[sample_id]
    dist = []
    dist.append((ch, sample[0]))
    for i in xrange(len(syms)):
        dist.append((syms[i], sample[i + 1]))
    #dist = sorted(dist, key=lambda symbol: symbol[1])
    # includes the target
    dist = [dist[0]] + sorted(dist[1:], key=lambda symbol: symbol[1])
    return dist[:nbest]


def simulate_eeg(sym, eeg_path=""):
    """
    generate non deterministic EEG history
    """
    if not eeg_path:
        raise ValueError('An path to EEG simulated samples is missing')

    eeg_smaples = eegs(eeg_path)
    if sym in syms_set or sym == ' ':
        if sym == ' ':
            sym = '#'
        return generate_eeg(eeg_smaples, sym)
    else:
        raise ValueError('An invalid symbol')


syms_set = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o",
        "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "#"]
path = 'EEGEvidence.txt-high'
sentence = "i went to ohsu"
for sym in sentence:
    print simulate_eeg(sym, path)
