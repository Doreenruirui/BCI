import numpy as np


def mean_reciporal_rank(output, groundtruth):
    if groundtruth == 0:
        return 0.
    res = 0.
    for i in range(len(output)):
        if groundtruth == output[i]:
            res += 1. / (i + 1)
            break
    return res

def batch_mrr(output, groundtruth, k):
    # output: batch_size * num_pred * len_inp
    # groundtruth: batch_size * len_inp
    if len(output.shape) > 2:
        output = np.transpose(output, [0, 2, 1])
        num_pred = output.shape[-1]
        batch_size = output.shape[0]
        res = list(map(lambda x, y: mean_reciporal_rank(x[:k], y), np.reshape(output, [-1, num_pred]), np.reshape(groundtruth,[-1])))
        res = np.reshape(res, [batch_size, -1])
    else:
        res = list(map(lambda x, y: mean_reciporal_rank(x[:k], y), output, groundtruth))
    return res


def batch_recall_at_k(output, groundtruth, k):
    # output: batch_size  * num_pred * len_inp
    # groundtruth: batch_size * len_inp
    if len(output.shape) == 2:
        return list(map(lambda x, y: 1 if y in x[:k] else 0, output, groundtruth))
    else:
        output = np.transpose(output, [0, 2, 1])
        num_pred = output.shape[-1]
        batch_size = output.shape[0]
        res = list(map(lambda x, y: 1 if y in x[:k] else 0, np.reshape(output, [-1, num_pred]), np.reshape(groundtruth, [-1])))
        return np.reshape(res, [batch_size, -1])



def batch_acc_at_k(output, groundtruth, k):
    # output: batch_size  * num_pred * len_inp
    # groundtruth: batch_size * len_inp
    if len(output.shape) == 2:
        return list(map(lambda x, y: 1 if y == x[0] else 0, output, groundtruth))
    else:
        output = np.transpose(output, [0, 2, 1])
        num_pred = output.shape[-1]
        batch_size = output.shape[0]
        res = list(map(lambda x, y: 1 if y == x[0] else 0, np.reshape(output, [-1, num_pred]), np.reshape(groundtruth, [-1])))
        return np.reshape(res, [batch_size, -1])