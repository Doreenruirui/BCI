import string
import random
import numpy as np


id2char = [b'<pad>', b'<sos>', b'<eos>', b' '] + list(string.ascii_lowercase)
char2id = {k: v for v, k in enumerate(id2char)}


def get_confusion(cur_char_id, num_cand):
    index = [ele for ele in range(3, len(id2char))]
    random.shuffle(index)
    a = index[:num_cand-1]
    if cur_char_id in a:
        a.append(index[num_cand])
    else:
        a.append(cur_char_id)
    random.shuffle(a)
    return a


def get_confusion_concat(cur_char_id, num_cand):
    index = [ele for ele in range(3, len(id2char))]
    random.shuffle(index)
    a = index[:num_cand-1]
    if cur_char_id in a:
        a.append(index[num_cand])
    else:
        a.append(cur_char_id)
    # random.shuffle(a)
    new_index = np.zeros(len(id2char), dtype=int)
    for item in a:
        new_index[item] = item
    return new_index


def generate_confusion(num_cand, cand_prob):
    list_char = ['_'] + list(string.ascii_lowercase)
    f = open('confusion_matrix', 'w')
    for i in range(len(list_char)):
        cur_char = list_char[i]
        list_new = list_char[:]
        random.shuffle(list_new)
        cur_cand = [ele for ele in list_new[:num_cand] if ele != cur_char]
        if len(cur_cand) < num_cand:
            cur_cand.append(list_new[num_cand])
        cur_prob = [cand_prob] + [(1 - cand_prob) / (num_cand - 1)] * (num_cand - 1)
        f.write(cur_char + '\t' + str(num_cand)
                + '\t' + '\t'.join(cur_cand)
                + '\t' + '\t'.join(map(str, cur_prob)) + '\n')
    f.close()


def load_confusion(file_confusion, num_char):
    global char2id, id2char
    dict_confusion = {}
    for line in file(file_confusion):
        line = line.split('\t')
        cur_char = char2id[line[0]]
        num_char = int(line[1])
        cur_cand = [char2id[ele] for ele in line[2:2 + num_char]]
        cur_prob = map(float, line[2 + num_char:])
        dict_confusion[cur_char] = [cur_cand, cur_prob]
    return dict_confusion




