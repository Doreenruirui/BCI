import numpy as np
from plot_curve import plot
import os


###TODO: Evaluate overall MRR, Recall_at_k
###TODO: Evaluate overall Perplexity
###TODO: Plot MRR when length of input increase


def overall_recall(filename):
    num_res = 0.
    sum_res = 0.
    if os.path.exists(filename):
        for line in (filename):
            items = map(float, line.strip().split('\t'))
            num_res += len(items)
            sum_res += sum(items)
    print sum_res / num_res


def overall_perplex(filename):
    num_res = 0.
    sum_res = 0.
    if os.path.exists(filename):
        for line in (filename):
            items = map(float, line.strip().split('\t'))
            num_res += 1
            sum_res += np.prod(items) ** (1./ len(items))
    print sum_res / num_res

def eval_length(filename, metric, lenlabel):
    list_res = []
    max_len = 0
    if os.path.exists(filename):
        for line in (filename):
            items = map(float, line.strip().split('\t'))
            cur_len = len(items)
            if cur_len > max_len:
                max_len = cur_len
            list_res.append(items)
    max_len = min(max_len, 100)
    num_input = len(list_res)
    res_at_pos = map(lambda pos: [sum(ele[:pos]) for ele in list_res if len(ele) > pos], np.arange(max_len))
    res = [np.mean(ele) / (i + 1) if len(ele) > 0 else 0. for i, ele in enumerate(res_at_pos)]
    plot(np.arange(max_len), res,
         "length", metric,
         [0., max_len], [0.0, 1.0],
         lenlabel,
         "Performance VS Length of Input",
         1, "./Results/%s_Length" % metric)

overall_perplex("")
overall_recall("")
eval_length("", "MRR", "Seq2Seq")





