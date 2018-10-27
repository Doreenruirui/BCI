import numpy as np
import sys
from os.path import join as pjoin, exists
import os
import string
from evaluate import mean_reciporal_rank
import kenlm
import datetime


lm, candidates, ncand, cand2id = None, None, None, None
folder_bci = '/scratch/dong.r/Dataset/BCI'
file_lm = sys.argv[1]
folder_test = sys.argv[2]
folder_eval = sys.argv[3]
file_name = sys.argv[4]
num_cand = int(sys.argv[5])
start = int(sys.argv[6])
end = int(sys.argv[7])


def initializer(file_model):
    global lm, candidates, ncand, cand2id
    lm = kenlm.LanguageModel(file_model)
    candidates =  ['#'] + list(string.ascii_lowercase)
    cand2id = {cand:cid for cid, cand in enumerate(candidates)}
    ncand = num_cand


def decode_line(paras):
    global lm, candidates, ncand, cand2id
    line, line_id = paras
    line = line.strip().lower()
    line = [ele if ele != ' ' else '#' for ele in line.strip()]
    mrr = []
    recall = []
    nch = len(line)
    cur_sen = ''
    all_score = []
    all_cands = []
    accuracy = []
    for i in range(nch):
        scores = []
        new_score = []
        for ch in candidates:
            if len(cur_sen) == 0:
                tmp_sen = cur_sen + ch
            else:
                tmp_sen = cur_sen + ' ' + ch
            full_path = list(lm.full_scores(tmp_sen))
            scores.append(sum([ele[0] for ele in full_path][:-1]))
            new_score.append(full_path[-2][0])
        index = np.argsort(scores)[::-1]
        predicts = [candidates[ele] for ele in index[:ncand]]
        all_cands.append(predicts)
        all_score.append(new_score[cand2id[line[i]]])
        cur_mrr = mean_reciporal_rank(predicts, line[i])
        cur_recall = 1 if line[i] in predicts else 0
        accuracy.append(1 if line[i] == predicts[0] else 0)
        mrr.append(cur_mrr)
        recall.append(cur_recall)
        if len(cur_sen) == 0:
            cur_sen = line[i]
        else:
            cur_sen += ' ' + line[i]
    # perplex = np.power(10.0, sum(all_score) * -1 / (nch + 1))
    return mrr, recall, all_score, all_cands, accuracy, line_id


def decode():
    folder_out = pjoin(folder_bci, folder_eval)
    if not exists(folder_out):
        os.makedirs(folder_out)
    f_mrr = open(pjoin(folder_bci, folder_eval, 'mrr.%d_%d' % (start, end)), 'w')
    f_recall = open(pjoin(folder_bci, folder_eval, 'recall.%d_%d' % (start, end)), 'w')
    f_perplex = open(pjoin(folder_bci, folder_eval, 'perplex.%d_%d' % (start, end)), 'w')
    f_top = open(pjoin(folder_bci, folder_eval, 'top.%d_%d' % (start, end)), 'w')
    f_acc = open(pjoin(folder_bci, folder_eval, 'acc.%d_%d' % (start, end)), 'w')
    initializer(file_lm)
    with open(pjoin(folder_bci, folder_test, file_name), 'r') as f_:
        print(datetime.datetime.now())
        lid = 0
        lines = []
        for line in f_:
            if lid >= start:
                lines.append(line.strip())
            lid += 1
            if lid == end:
                break
        for line in f_:
            line = line.strip()
            mrr, recall, perplex, cands, accu, line_id = decode_line((line, 0))
            f_mrr.write('\t'.join(list(map(str, mrr))) + '\n')
            f_recall.write('\t'.join(list(map(str, recall))) + '\n')
            f_perplex.write('\t'.join(list(map(str, perplex))) + '\n')
            f_top.write('\t'.join([' '.join(e) for e in cands]) + '\n')
            f_acc.write('\t'.join(list(map(str, accu))) + '\n')
        print(datetime.datetime.now())
    f_mrr.close()
    f_recall.close()
    f_perplex.close()
    f_top.close()
    f_acc.close()

decode()



