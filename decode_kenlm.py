import numpy as np
import sys
from os.path import join as pjoin, exists
import os
import string
from evaluate import mean_reciporal_rank
import kenlm
from multiprocessing import Pool

folder_bci = '/scratch/dong.r/Dataset/BCI'
file_lm = sys.argv[1]
folder_test = sys.argv[2]
folder_eval = sys.argv[3]
file_name = sys.argv[4]
num_cand = int(sys.argv[5])


def initializer():
    global lm, candidates, ncand
    lm = kenlm.LanguageModel(file_lm)
    candidates =  ['#'] + list(string.ascii_lowercase)
    ncand = num_cand


def decode_line(paras):
    global lm, candidates, ncand
    line, line_id = paras
    line = line.strip().lower()
    # line = ' '.join([ele.strip() for ele in line.split(' ')])
    # line = [ele if ele != ' ' else '<space>' for ele in line]
    # perplex = lm.perplexity('<s> ' + ' '.join(line))
    line = [ele if ele != ' ' else '#' for ele in line.strip()]
    perplex =  lm.perplexity(' '.join(line))
    nitem = len(line)
    mrr = []
    recall = []
    nch = len(line)
    cur_sen = ''
    for i in range(nch):
        scores = []
        for ch in candidates:
            if len(cur_sen) == 0:
                scores.append(lm.perplexity(cur_sen))
            else:
                scores.append(lm.perplexity(cur_sen + ' ' + ch))
        index = np.argsort(scores).tolist()
        predicts = [candidates[ele] for ele in index[:ncand]]
        cur_mrr = mean_reciporal_rank(predicts, line[i])
        cur_recall = 1 if line[i] in predicts else 0
        mrr.append(cur_mrr)
        recall.append(cur_recall)
        if len(cur_sen) == 0:
            cur_sen = line[i]
        else:
            cur_sen += ' ' + line[i]
    return mrr, recall, perplex, nitem, line_id


def decode():
    folder_out = pjoin(folder_bci, folder_eval)
    if not exists(folder_out):
        os.makedirs(folder_out)
    f_mrr = open(pjoin(folder_bci, folder_eval, 'mrr'), 'w')
    f_recall = open(pjoin(folder_bci, folder_eval, 'recall'), 'w')
    f_perplex = open(pjoin(folder_bci, folder_eval, 'perplex'), 'w')
    p = Pool(100, initializer=initializer())
    with open(pjoin(folder_bci, folder_test, file_name), 'r') as f_:
        lines = [ele for ele in f_.readlines() if len(ele.strip()) > 0]
        decode_line([lines[0], 0])
        res = p.map(decode_line, zip(lines, np.arange(len(lines))))
        list_res = [None for _ in range(len(lines))]
        for mrr, recall, perplex, nitem, line_id in res:
            list_res[line_id] = [mrr, recall, perplex, nitem]
        for ele in list_res:
            f_mrr.write('\t'.join(map(str, ele[0])) + '\n')
            f_recall.write('\t'.join(map(str,ele[1])) + '\n')
            f_perplex.write(str(ele[2]) + '\t' + str(ele[3]) + '\n')
    f_mrr.close()
    f_recall.close()
    f_perplex.close()

decode()



