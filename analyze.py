import numpy as np
#from plot_curve import plot
import os
import sys


###TODO: Evaluate overall MRR, Recall_at_k
###TODO: Evaluate overall Perplexity
###TODO: Plot MRR when length of input increase


def overall_recall(filename):
    num_res = 0.
    sum_res = 0.
    if os.path.exists(filename):
        for line in file(filename):
            if len(line.strip()) == 0:
                continue
            items = map(float, line.strip().split('\t'))
            num_res += len(items)
            sum_res += sum(items)
    print sum_res / num_res


def overall_perplex_neural(filename):
    num_res = 0.
    sum_res = 0.
    if os.path.exists(filename):
        for line in file(filename):
            items = map(float, line.strip().split('\t'))
            num_res += len(items)
            sum_res += -sum(items)
            #sum_res += np.exp(-sum(items)/len(items))
            #sum_res += 1./(np.prod(items) ** (1./ len(items)))
    print sum_res / num_res
    print np.exp(- sum_res / num_res)

def overall_perplex_kenlm(filename):
    num_res = 0.
    sum_res = 0.
    unnormal = 0
    if os.path.exists(filename):
        for line in file(filename):
            items = map(float, line.strip().split('\t'))
            perplex = items[0]
            nitem = items[1]
            #sum_res += perplex * nitem
            #num_res += nitem
            if 10 ** (-perplex * nitem) == 0:
                unnormal += 1
            else:
                sum_res += np.log(10 ** (-perplex * nitem)) * -1    
                num_res += nitem
    print sum_res / num_res
    print unnormal
    #print 10 ** (- sum_res / num_res)

def overall_perplex_baseline(filename):
    num_res = 0.
    sum_res = 0.
    if os.path.exists(filename):
        for line in file(filename):
            items = map(float, line.strip().split('\t'))
            sum_res += np.sum(-1 * np.log(np.asarray(items)))
            num_res += len(items)
    print sum_res / num_res
    print np.exp(- sum_res / num_res)

def eval_length(filename, metric, lenlabel):
    list_res = []
    max_len = 0
    if os.path.exists(filename):
        for line in file(filename):
            items = map(float, line.strip().split('\t'))
            cur_len = len(items)
            if cur_len > max_len:
                max_len = cur_len
            list_res.append(items)
    max_len = min(max_len, 100)
    num_input = len(list_res)
    res_at_pos = map(lambda pos: [sum(ele[:pos]) for ele in list_res if len(ele) >= pos], np.arange(max_len) + 1)
    res = [np.mean(ele) / (i + 1) if len(ele) > 0 else 0. for i, ele in enumerate(res_at_pos)]
    plot(np.arange(max_len), res,
         "length", metric,
         [0., max_len], [0.0, 1.0],
         lenlabel,
         "Performance VS Length of Input",
         1, "./Results/%s_Length" % metric)

filename = sys.argv[1]
flag = int(sys.argv[2])
if flag == 1:
    overall_perplex_neural(filename)
elif flag == 2:
    overall_perplex_kenlm(filename)
elif flag==3:
    overall_perplex_baseline(filename)
elif flag == 0:
    overall_recall(filename)
#eval_length(filename, "MRR", "Seq2Seq")





