import numpy as np
import sys
from os.path import join as pjoin, exists
import os
import string
import operator
from evaluate import batch_mrr, batch_recall_at_k

folder_bci = '/gss_gpfs_scratch/dong.r/Dataset/BCI'
folder_train = sys.argv[1]
folder_test = sys.argv[2]
folder_eval = sys.argv[3]
file_name = sys.argv[4]


id2char = [b'<pad>', b'<sos>', b'<eos>', b' '] + list(string.ascii_lowercase)
char2id = {k: v for v, k in enumerate(id2char)}


def count_frequency():
    dict_char = {}
    for line in file(pjoin(folder_bci, folder_train, 'train.ids')):
        line = map(int, line.strip().split(' '))
        for char in line:
            dict_char[char] = dict_char.get(char, 0) + 1
    vocab = sorted(dict_char, key=dict_char.get, reverse=True)
    candidates = vocab[:10]
    sum_prob = sum(dict_char.values())
    cand_prob = np.zeros(len(id2char))
    for ele in dict_char:
        cand_prob[ele] = dict_char[ele] * 1. / sum_prob
    return candidates, cand_prob

def decode():
    # candidates = [3, 8, 4, 23, 12, 17, 18, 22, 21, 11]
    candidates, cand_prob = count_frequency()
    folder_out = pjoin(folder_bci, folder_eval)
    if not exists(folder_out):
        os.makedirs(folder_out)
    f_mrr = open(pjoin(folder_bci, folder_eval, 'mrr'), 'w')
    f_recall = open(pjoin(folder_bci, folder_eval, 'recall'), 'w')
    f_perplex = open(pjoin(folder_bci, folder_eval, 'perplex'), 'w')
    line_id = 0
    for line in file(pjoin(folder_bci, folder_test, file_name)):
        line = map(int, line.strip().split(' '))
        cur_mrr = batch_mrr(np.asarray([candidates] * len(line)), line, 10)
        cur_recall = batch_recall_at_k(np.asarray([candidates] * len(line)), line ,10)
        cur_perplex = [cand_prob[ele] for ele in line]
        f_mrr.write('\t'.join(map(str, cur_mrr)) + '\n')
        f_recall.write('\t'.join(map(str,cur_recall)) + '\n')
        f_perplex.write('\t'.join(map(str, cur_perplex)) + '\n')
        if line_id > 51200:
            break
        line_id += 1
    f_mrr.close()
    f_recall.close()
    f_perplex.close()

decode()



