import matplotlib.pyplot as plt
from multiprocessing import Pool
import numpy as np
import sys
from plot_curve import plot
allres = None
from os.path import join as pjoin
import os
from data_generate import id2char
from data_load import *

def analyze_sentence():
    x = np.random.normal(size = 1000)
    plt.hist(x, normed=True, bins=30)
    plt.ylabel('Probability')

    x = np.arange(3)
    plt.bar(x, height= [1,2,3])
    plt.xticks(x+.5, ['a','b','c'])
    plt.show()

def initializer(res):
    global allres
    allres = res

def eval_position(pos):
    global allres
    mean_res = 0
    nres = 0
    for line in allres:
        if len(line) > pos:
            mean_res += line[pos]
            nres += 1
    return mean_res / nres, nres, pos

def analyze_position_file(filename):
    if len(filename) == 0:
        return [], 0
    res = []
    len_x = []
    for line in file(filename):
        items = map(float, line.strip().split('\t'))
        res.append(items)
        len_x.append(min(len(items), 300))
    max_len = max(len_x)
    pool = Pool(100, initializer=initializer(res))
    results = pool.map(eval_position, np.arange(300))
    new_res = [None for _ in range(max_len)]
    for mean_res, nres, pos in results:
        new_res[pos] = (mean_res, nres)
    return new_res, max_len

def analyze_position(filename1, filename2='', measure='MRR', corpus='NYT', method='Seq2Seq'):
    res1, len_x1 = analyze_position_file(filename1)
    res2, len_x2 = analyze_position_file(filename2)
    # print new_res
    x = np.arange(300) + 1
    y = [[ele[0] for ele in res1]]
    if len_x2 > 0:
        y.append([ele[0] for ele in res2])
    plot(x, y, 'Position', measure, [0.99, 300.01], [0, 1],
         ['12_Epoch', '19_Epoch'], '%s %s v.s. Position on %s' % (method, measure, corpus), 1,
         'Results/%s_%s_Pos_%s' % (method, measure, corpus))

def analyze_word_break(file_id, filename, end, chunk_size):
    nfile = end / chunk_size
    lines = []
    sum_prob = 0
    nprob = 0
    sum_prob1 = 0
    sum_prob2 = 0
    sum_prob3 = 0
    nprob3 = 0
    for line in file(file_id):
        items = map(int, line.strip().split(' '))
        lines.append(items)
    for fid in range(nfile):
        cur_start = fid * chunk_size
        cur_end = fid * chunk_size + chunk_size
        cur_filename = '%s.%d_%d' % (filename, cur_start, cur_end)
        res = []
        for line in file(cur_filename):
            items = map(float, line.strip().split('\t'))
            res.append(items)
        if len(res) == 0:
            continue
        cur_lines = lines[cur_start: cur_end]
        len_line = [len(ele) for ele in cur_lines]
        sorted_index = np.argsort(len_line)
        line_x = map(lambda ele: cur_lines[ele], sorted_index)
        nline = len(cur_lines)
        for i in range(nline):
            items = line_x[i][:300]
            cur_res = res[i]
            for j in range(len(items)):
                if items[j] == 3 and j < len(items) - 1:
                    sum_prob += cur_res[j + 1]
                    nprob += 1
                    sum_prob1 += cur_res[j]
                    sum_prob2 += cur_res[j - 1]
                    if j + 2 < len(items):
                        sum_prob3 += cur_res[j + 2]
                        nprob3 += 1
    print 'After Space', sum_prob / nprob
    print 'Space', sum_prob1 / nprob
    print 'Before Space', sum_prob2 / nprob
    print 'Second Character After Space', sum_prob3 / nprob3

def reprocess(file_id, filename, end, chunk_size):
    lines = load_int_file(file_id)
    nfile = end / chunk_size
    res = [None for _ in range(end)]
    for fid in range(nfile):
        cur_start = fid * chunk_size
        cur_end = fid * chunk_size + chunk_size
        cur_filename = '%s.%d_%d' % (filename, cur_start, cur_end)
        if not os.path.exists(cur_filename):
            continue
        with open(cur_filename, 'r') as f_:
            cur_res = f_.readlines()
        if len(cur_res) == 0:
            continue
        cur_lines = lines[cur_start: cur_end]
        len_line = [len(ele) for ele in cur_lines]
        sorted_index = np.argsort(len_line)
        for i in range(chunk_size):
            res[cur_start + sorted_index[i]] = cur_res[i]
    with open(filename + '.order', 'w') as f_:
        for ele in res[:end]:
            if ele is not None:
                f_.write(ele)
            else:
                f_.write('\n')


def analyze_methods(file_id, filename1, filename2, end, out_file):
    lines = load_int_file(file_id)
    lines = lines[:end]
    if 'baseline' in filename1:
        res1 = load_float_file(filename1)
        res1 = res1[:end]
    else:
        res1 = load_float_file(filename1)
        res1 = res1[:end]
    res2 = load_float_file(filename2)
    res2 = res2[:end]
    # f_= open(file_id.strip('test.ids') + out_file, 'w')
    dict_char = {}
    dict_char_all = {}
    dict_space = {}
    sum_freq1 = 0
    sum_freq2 = 0
    sum_freq = 0
    for i in range(end):
        if len(res1[i]) > 0 and len(res2[i]) > 0:
            cur_len = min(300, len(lines[i]))
            for j in range(cur_len):
                cur_char = id2char[lines[i][j]]
                if cur_char == ' ':
                    if res2[i][j] < res1[i][j]:
                        sum_freq1 += 1
                    elif res2[i][j] > res1[i][j]:
                        sum_freq2 += 1
                    sum_freq += 1
            #     f_.write(cur_char)
            #     if cur_char == ' ':
            #         dict_char_all['<space>'] = dict_char_all.get('<space>', 0) + 1
            #     else:
            #         dict_char_all[cur_char] = dict_char_all.get(cur_char, 0) + 1
            #     if res2[i][j] < res1[i][j]:
            #         f_.write('(%.3f, %.3f)' % (res1[i][j], res2[i][j]))
            #         if cur_char == ' ':
            #             dict_char['<space>'] = dict_char.get('<space>', 0) + 1
            #         else:
            #             dict_char[cur_char] = dict_char.get(cur_char, 0) + 1
            #         if cur_char == ' ':
            #             dict_space[0] = dict_space.get(0, 0) + 1
            #         else:
            #             if j > 0:
            #                 if lines[i][j - 1] == 3:
            #                     dict_space[1] = dict_space.get(1, 0) + 1
            #             if j < cur_len - 1:
            #                 if lines[i][j + 1] == 3:
            #                     dict_space[-1] = dict_space.get(-1, 0) + 1
            #             if j < cur_len - 2:
            #                 if lines[i][j + 2] == 3:
            #                     dict_space[-2] = dict_space.get(-2, 0) + 1
            #             if j > 1:
            #                 if lines[i][j - 2] == 3:
            #                     dict_space[2] = dict_space.get(2, 0) + 1
            # f_.write('\n')
    # f_.close()
    # sum_freq = sum(dict_char.values())
    # # cands = ['<space>', 'e', 't', 'a', 'i', 'n', 'o', 's', 'r', 'h']
    # for ele in dict_char:
    #     print ele, dict_char[ele] * 1. / sum_freq
    #
    # print sum_freq
    # sum_freq_all = sum(dict_char_all.values())
    # for ele in dict_char:
    #     print ele, dict_char_all[ele] * 1./ sum_freq_all
    # print dict_space
    # for ele in [-1, 0, 1]:
    #     print dict_space[ele] * 1. / sum_freq
    print sum_freq, sum_freq1 * 1. / sum_freq, sum_freq2 * 1. / sum_freq


def analyze_random(filename1, file_id):
    rd = np.load(filename1)
    lines = []
    for line in file(file_id):
        items = map(int, line.strip().split(' '))
        lines.append(items)
    nline = len(lines)
    print nline, rd.shape
    mean_prob = 0
    npred = 0
    mean_high = 0
    nhigh = 0
    non_high = 0
    for i in range(nline):
        prob_line = rd[1:, i, :]
        items = lines[i]
        nitem = min(len(items), 300)
        for j in range(nitem - 1):
            prob_c = prob_line[j, :]
            maxid = np.argmax(prob_c)
            if maxid == items[j]:
                mean_high += prob_c[maxid]
                nhigh += 1
            else:
                non_high += 1
            prob_c = prob_line[j, items[j]]
            mean_prob += prob_c
        npred += nitem - 1
    print mean_prob / npred
    print non_high * 1. / npred
    print mean_high / nhigh



#result_file, measure='mrr', corpus='NYT', method='Seq2seq'
analyze_position(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])

#random data file, test id file
#  analyze_random(sys.argv[1], sys.argv[2])

#Result file prefix, num of lines, chunk size, test id file
# analyze_word_break(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))

#file_id, file_name, end, chunk_size
#reprocess(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))

#file_id, file_name1, file_name2, end, out_file
#analyze_methods(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5])
