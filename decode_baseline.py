import numpy as np
import sys
from os.path import join as pjoin
import string
from evaluate import batch_mrr, batch_recall_at_k

folder_name = '/gss_gpfs_scratch/dong.r/Dataset/BCI_sen'
folder_eval = sys.argv[1]


id2char = [b'<pad>', b'<sos>', b'<eos>', b' '] + list(string.ascii_lowercase)
char2id = {k: v for v, k in enumerate(id2char)}


def count_frequency():
    dict_char = {}
    for line in file(pjoin(folder_name, 'train.ids')):
        line = map(int, line.strip().split(' '))
        for char in line:
            dict_char[char] = dict_char.get(char, 0) + 1
    vocab = sorted(dict_char, key=dict_char.get, reverse=True)
    candidates = vocab[:10]
    return candidates

def decode():
    candidates = [3, 8, 4, 23, 12, 17, 18, 22, 21, 11]
    #candidates = count_frequency()
    f_mrr = open(pjoin(folder_name, folder_eval, 'mrr'), 'w')
    f_recall = open(pjoin(folder_name, folder_eval, 'recall'), 'w')
    line_id = 0
    for line in file(pjoin(folder_name, 'dev.ids')):
        line = map(int, line.strip().split(' '))
        cur_mrr = batch_mrr(np.asarray([candidates] * len(line)), line, 10)
        cur_recall = batch_recall_at_k(np.asarray([candidates] * len(line)), line ,10)
        f_mrr.write('\t'.join(map(str, cur_mrr)) + '\n')
        f_recall.write('\t'.join(map(str,cur_recall)) + '\n')
        if line_id > 51200:
            break
        line_id += 1
    f_mrr.close()
    f_recall.close()

decode()



